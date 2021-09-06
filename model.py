import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from einops import rearrange, repeat

from functools import partial
from building_blocks.axial_attention import AxialLongShortTransformer
from building_blocks.graph_transformer_pytorch import GraphTransformer, GatedResidual
from building_blocks.en_transformer import EnTransformer

import torch.nn.functional as F
from utils import soft_one_hot_linspace

class Dense(nn.Module):
    def __init__(self, layer_structure, checkpoint=False):
        super().__init__()
        layers = []
        for idx, (back, front) in enumerate(zip(layer_structure[:-1],
                                            layer_structure[1:])):
            layers.append(nn.Linear(back, front))
            if idx < len(layer_structure) - 2: layers.append(nn.LeakyReLU())
        self.layers = nn.Sequential(*layers)
        self.checkpoint = checkpoint

    def block_checkpoint(self, layers):
        def checkpoint_forward(x):
            return layers(x)
        return checkpoint_forward

    def forward(self, x):
        return checkpoint.checkpoint(self.block_checkpoint(self.layers), x) if self.checkpoint else self.layers(x)


def create_model(config):
    axial_transformer = AxialLongShortTransformer(
        dim=config.at_dim,
        depth=config.at_depth,
        heads=config.at_heads,
        dim_heads=config.at_dim_head,
        dim_index=1,
        checkpoint=config.at_checkpoint,
        window_size=config.at_window_size
    )

    graph_transformer = GraphTransformer(
        dim=config.gt_dim,
        edge_dim=config.gt_edim,
        depth=config.gt_depth,
        heads=config.gt_heads,
        dim_head=config.gt_dim_head,
        checkpoint=config.gt_checkpoint
    )

    en_transformer = EnTransformer(
        dim = config.et_dim,
        edge_dim = config.et_edim,
        depth = config.et_depth,
        dim_head = config.et_dim_head,
        heads = config.et_heads,
        coors_hidden_dim = config.et_coors_hidden_dim,
        checkpoint=config.et_checkpoint,
        neighbors=config.en_num_neighbors,
        rel_pos_emb=True,
    )

    model = StructureExtractor(
        axial_transformer=axial_transformer,
        graph_transformer=graph_transformer,
        en_transformer=en_transformer,
        config=config
    )

    return model


class StructureExtractor(nn.Module):
    def __init__(
            self,
            axial_transformer,
            graph_transformer,
            en_transformer,
            config
        ):
        super().__init__()
        self.config = config

        if config.use_msa:
            self.node_msa_distiller = Dense(config.node_msa_distill_layers)
            self.edge_msa_distiller = Dense(config.edge_msa_distill_layers)
            node_dim, edge_dim = (config.node_msa_distill_layers[-1],
                                  config.edge_msa_distill_layers[-1])

        if config.use_seq:
            self.node_seq_distiller = Dense(config.node_seq_distill_layers)
            self.edge_seq_distiller = Dense(config.edge_seq_distill_layers)
            node_dim, edge_dim = (config.node_seq_distill_layers[-1],
                                  config.edge_seq_distill_layers[-1])

        if config.use_msa == config.use_seq:
            self.node_ens = Dense(config.node_ens_distill_layers)
            self.edge_ens = Dense(config.edge_ens_distill_layers)
            node_dim, edge_dim = (config.node_ens_distill_layers[-1],
                                  config.edge_ens_distill_layers[-1])


        self.node_edge_mixer = Dense([
            (2 * node_dim
               + edge_dim),
            2 * edge_dim,
            edge_dim
        ])

        if config.use_at:
            self.to_at_edges = Dense([edge_dim, config.at_dim])
            self.axial_transformer = axial_transformer
            self.to_distance = Dense([config.at_dim, 32, config.distance_number_of_bins])
            self.to_angle = Dense([config.at_dim, 32, 3 * config.angle_number_of_bins])
            self.to_dihedrals_at = Dense([config.at_dim, config.at_dim, 24])

        self.residual = GatedResidual(config.gt_dim)
        self.to_gt_nodes = Dense([config.at_dim if config.use_at else node_dim, config.gt_dim])
        self.to_gt_edges = Dense([config.at_dim if config.use_at else edge_dim, config.gt_edim])

        self.graph_transformer = graph_transformer
        self.to_position = Dense([config.gt_dim, 32, 3])

        self.to_en_nodes = Dense([config.gt_dim, config.et_dim])
        self.to_en_edges = Dense([config.gt_edim, config.et_edim])

        self.en_transformer = en_transformer
        self.to_lddt = Dense([config.et_dim, 1])
        self.to_dihedrals_en = Dense([config.et_dim, 32, 24])

        self.gaussian_noise = config.gaussian_noise
        self.unroll_steps = config.unroll_steps

        self.msa_wipe_out_prob = config.msa_wipe_out_prob
        self.msa_wipe_out_dropout = config.msa_wipe_out_dropout

        self.msa_embeddings = config.use_msa
        self.seq_embeddings = config.use_seq
        self.topography_giveaway = config.topography_giveaway

        self.edge_norm_emb = partial(soft_one_hot_linspace, start=0,
                    end=config.distance_max_radius, number=config.distance_number_of_bins, basis='gaussian', cutoff=True)

        if self.topography_giveaway:
            self.distance_embedder = Dense([config.giveaway_distance_resolution, 32, 32])
            self.angular_embedder = Dense([config.giveaway_angle_resolution * 3, 32, 32])
            self.edge_embedder = Dense([64, 64, edge_dim])
            self.node_token_emb = nn.Embedding(20, node_dim)

        self.train_fold_steps = config.train_fold_steps
        self.eval_fold_steps = config.eval_fold_steps


    def forward(self, batch, is_training):
        output = {}

        node_pad_mask = batch.node_pad_mask
        edge_pad_mask = batch.edge_pad_mask
        batch_size, seq_len = batch.seqs.size()

        if self.msa_embeddings:
            msa_nodes = self.node_msa_distiller(batch.node_msa_features)
            msa_edges = self.edge_msa_distiller(batch.edge_msa_features)

            if is_training:
                chosen = torch.rand(batch_size) < self.msa_wipe_out_prob
                msa_nodes[chosen] = F.dropout(msa_nodes[chosen], p=self.msa_wipe_out_dropout)
                msa_edges[chosen] = F.dropout(msa_edges[chosen], p=self.msa_wipe_out_dropout)

        if self.seq_embeddings:
            seq_nodes = self.node_seq_distiller(batch.node_seq_features)
            seq_edges = self.edge_seq_distiller(batch.edge_seq_features)

        if self.msa_embeddings and self.seq_embeddings:
            nodes_source = self.node_ens(msa_nodes + seq_nodes)
            edges_source = self.edge_ens(msa_edges + seq_edges)

            nodes_cross = repeat(nodes_source, 'b i c -> b i j c', j=seq_len), repeat(nodes_source, 'b i c -> b j i c', j=seq_len)
            edges_source = self.node_edge_mixer(torch.cat([*nodes_cross, edges_source], dim=-1))

        elif self.msa_embeddings or self.seq_embeddings:
            nodes_source = msa_nodes if self.msa_embeddings else seq_nodes
            edges_source = msa_edges if self.msa_embeddings else seq_edges
            nodes_cross = (repeat(nodes_source, 'b i c -> b i j c', j=seq_len), repeat(nodes_source, 'b i c -> b j i c', j=seq_len))
            edges_source = self.node_edge_mixer(torch.cat([*nodes_cross, edges_source], dim=-1))

        elif (not self.msa_embeddings) and (not self.seq_embeddings):
            dist_signal = soft_one_hot_linspace(batch.edge_distance, start=3, end=25, number=self.config.giveaway_distance_resolution, basis='gaussian', cutoff=True)
            dist_emb = self.distance_embedder(dist_signal)

            angle_embeds = soft_one_hot_linspace(batch.edge_angles, start=-1, end=1, number=self.config.giveaway_angle_resolution, basis='gaussian', cutoff=True)
            angle_embeds = rearrange(angle_embeds, 'b s z a c -> b s z (a c)')
            angs_emb = self.angular_embedder(angle_embeds)

            edges_source = self.edge_embedder(torch.cat((dist_emb, angs_emb), dim=-1))
            nodes_source = self.node_token_emb(batch.seqs)

        if self.config.wipe_edge_information:
            edges_source = F.dropout(edges_source, p=1.0)

        if self.config.use_at:
            at_edges = self.to_at_edges(edges_source)
            at_edges = rearrange(at_edges, 'b i j c -> b c i j')
            at_edges = self.axial_transformer(at_edges)
            at_edges = rearrange(at_edges, 'b c i j -> b i j c')

            at_nodes = rearrange(torch.diagonal(at_edges, dim1=1, dim2=2), 'b c i -> b i c')

            output['predicted_dihedrals_at'] = rearrange(self.to_dihedrals_at(at_nodes), 'b s (c n) -> b s c n', n=2)
            output['distance_logits'] = self.to_distance(at_edges)
            output['angle_logits'] = rearrange(self.to_angle(at_edges), 'b s z (a c) -> b s z a c', a=3)

            gt_nodes, gt_edges = self.residual(self.to_gt_nodes(at_nodes), nodes_source), self.to_gt_edges(at_edges)
        else:
            gt_nodes, gt_edges = self.to_gt_nodes(nodes_source), self.to_gt_edges(edges_source)

        if self.config.use_gt:
            gt_nodes, gt_edges = self.graph_transformer(nodes=gt_nodes, edges=gt_edges, mask=node_pad_mask)
            coors = self.to_position(gt_nodes)
            output['pose_prediction'] = coors.unsqueeze(1)
            coors = coors.detach()
        else:
            coors = batch.crds[..., 1, :].clone()
            coors = coors + torch.normal(0, self.gaussian_noise, coors.size(), device=coors.device)

        if self.config.use_en:
            en_nodes, en_edges = self.to_en_nodes(gt_nodes), self.to_en_edges(gt_edges)

            if self.config.en_num_seq_neighbors > 0:
                idxs = torch.arange(0, seq_len).to(edge_pad_mask.device)
                adj_mat = (idxs[:, None] - idxs[None, :]).abs() < int(self.config.en_num_seq_neighbors / 2)
            else:
                adj_mat = None

            if is_training:
                # unroll so network learns to fix its own previous mistakes
                batch_steps = torch.randint(0, self.unroll_steps, [batch_size])
                with torch.no_grad():
                    max_step = torch.max(batch_steps).item()
                    for step in range(max_step):
                        chosen = batch_steps >= step
                        _, coors[chosen] = self.en_transformer(
                            feats=en_nodes[chosen], coors=coors[chosen],
                            # edges=en_edges[chosen], mask=node_pad_mask[chosen],
                            mask=node_pad_mask[chosen], # edges=en_edges[chosen],
                            adj_mat=adj_mat
                        )

            coors = coors.detach()
            trajectory = []

            for _ in range(self.train_fold_steps if is_training else self.eval_fold_steps):
                nodes_, coors = self.en_transformer(
                    feats=en_nodes, coors=coors,
                    # edges=en_edges, mask=node_pad_mask,
                    mask=node_pad_mask,
                    adj_mat=adj_mat
                )
                trajectory.append(coors)
                coors = coors.detach()

            trajectory = rearrange(torch.stack(trajectory, dim=0), 't b s e -> b t s e')

            output['trajectory'] = trajectory
            output['predicted_dihedrals_en'] = rearrange(self.to_dihedrals_en(nodes_), 'b s (c n) -> b s c n', n=2)
            output['lddt_prediction'] = self.to_lddt(nodes_)

        return output
