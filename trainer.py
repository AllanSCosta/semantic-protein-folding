import os

from functools import partial
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import wandb
os.environ["WANDB_MODE"] = "dryrun"

from tqdm import tqdm
from visualization import plot_predicted_angles, plot_aligned_timeseries, plot_aligned_backbones, plot_distogram_predictions

from data import VALIDATION_DATASETS, TRAIN_DATASETS, TEST_DATASETS
from utils import torsion_angle_loss, discretize, wrap, point_in_circum_to_angle, logit_expectation, get_alignment_metrics
from building_blocks.protein_utils import get_protein_metrics
from copy import deepcopy
from einops import repeat, rearrange



def unbatch(batch, attr, type='node'):
    features = []
    for slice, feature in zip(batch.__num_nodes_list__, attr):
        if type == 'node':
            feature = feature[:slice]
        elif type == 'edge':
            feature = feature[:slice, :slice]
        features.append(feature)
    return features



class Trainer():
    def __init__(self, hparams, model, loaders):
        super().__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.config = hparams
        self.model = model
        self.model.to(self.device)
        self.loaders = loaders

        if wandb.run:
            self.model_path = os.path.join(wandb.run.dir, 'checkpoint.pt')
            print(f'model path: {self.model_path}')
            torch.save(self.model, self.model_path)
            wandb.watch(self.model)

        self.best_val_loss = float('inf')

        self.angle_binner = partial(discretize, start=-1, end=1,
                                    number_of_bins=hparams.angle_number_of_bins)
        self.distance_binner = partial(discretize, start=3, end=hparams.distance_max_radius,
                                    number_of_bins=hparams.distance_number_of_bins)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)


    def train(self):
        for epoch in range(self.config.max_epochs):
            epoch_metrics = self.evaluate('train', epoch)

            if epoch > self.config.validation_start:
                if epoch % self.config.validation_check_rate == 0:
                    for split in VALIDATION_DATASETS:
                        epoch_metrics.update(self.evaluate(split, epoch))

            if wandb.run: wandb.log(epoch_metrics)
            print('saving model')
            torch.save(self.model.state_dict(), self.model_path)


    def test(self):
        for test_set in TEST_DATASETS:
            test_metrics = self.evaluate(test_set)
            if wandb.run: wandb.log(test_metrics)


    def test_logging(self, results):
        for id, result in results.items():
            media = {}

            if self.config.use_at:
                prediction_figure = plot_distogram_predictions(
                    result['at_distance_map'][1], result['at_angle_map'][1],
                    result['at_distance_map'][0], result['at_angle_map'][0]
                )

                at_dihedrals = plot_predicted_angles(
                    name=id,
                    seq=result['sequence'],
                    anchor=result['at_anchor'],
                    gnd_angs=result['at_dihedrals'][0],
                    pred_angs=result['at_dihedrals'][1]
                )

                media.update({
                    f'{id} at topography prediction': prediction_figure,
                    f'{id} at dihedral prediction': wandb.Html(at_dihedrals._make_html())
                })

            if self.config.use_en or self.config.use_gt:
                alignment_fig = plot_aligned_backbones(
                    result['en_trajectory'][-1][1],
                    result['en_trajectory'][-1][0],
                    result['en_alignment_metrics'],
                )
                media.update({ f'{id} alignment': alignment_fig })

            if self.config.use_en:
                timeseries_html = plot_aligned_timeseries(result['en_trajectory'])
                if wandb.run: timeseries_html.write_html(os.path.join(wandb.run.dir, f'{id}.html'))

                en_dihedrals = plot_predicted_angles(
                    name=id,
                    seq=result['sequence'],
                    anchor=result['en_anchor'],
                    gnd_angs=result['en_dihedrals'][0],
                    pred_angs=result['en_dihedrals'][1]
                )

                media.update({ f'{id} en dihedral predictions': wandb.Html(en_dihedrals._make_html()) })

            if wandb.run: wandb.log(media)


    def evaluate(self, split_name, epoch=0):
        is_training = split_name in TRAIN_DATASETS
        is_testing = split_name in TEST_DATASETS

        epoch_metrics = defaultdict(list)
        loader = self.loaders[split_name]

        with torch.set_grad_enabled(is_training):
            with tqdm(enumerate(loader), total=len(loader)) as bar:
                bar.set_description(f'[{epoch}] {split_name}')
                for batch_idx, batch in bar:
                    torch.cuda.empty_cache()
                    batch = batch.to(self.device)

                    batch_predictions = self.model(batch, is_training=is_training)
                    metrics, results = self.batch_metrics(batch, batch_predictions)

                    loss = metrics['loss'] = (
                       (self.config.at_loss_coeff * (metrics['at_dihedral_loss']) if self.config.use_at else 0) +
                       (self.config.gt_loss_coeff * (metrics['gt_rmsd']) if self.config.use_gt else 0)  +
                       (self.config.et_loss_coeff * (metrics['en_rmsd'] + (metrics['en_drmsd'] if self.config.et_drmsd else 0) + metrics['en_dihedral_loss']) if self.config.use_en else 0)
                    )

                    if is_training:
                        self.optim.zero_grad()
                        loss.backward()
                        self.optim.step()

                    if is_testing:
                        self.test_logging(results)

                    for k, v in metrics.items(): epoch_metrics[k].append(v.item() if type(v) is not float else v)

                    if batch_idx % self.config.report_frequency == 0:
                        report = ', '.join([f'{k}={np.mean(epoch_metrics[k]):.3e}' for k, v in epoch_metrics.items()])
                        print(report)

                    bar.set_postfix(loss = f'{np.mean(epoch_metrics["loss"][-100:]):.3e}')

        epoch_metrics = { f'{split_name} {k}': np.mean(v)
                         for k, v in epoch_metrics.items() }

        return epoch_metrics


    def batch_metrics(self, batch, predictions, fetch_results=True):
        metrics = defaultdict(float)

        # collector intercepts batches to collect per-datum information
        collector = None
        if fetch_results:
            collector = defaultdict(lambda: defaultdict(dict))
            for id, seq in zip(batch.ids, batch.str_seqs): collector[id]['sequence'] = seq

        # =========================
        # AXIAL TRANSFORMER PERFORMANCE
        # =========================
        if self.config.use_at:
            metrics.update(
                self.batched_topographical_metrics(
                    batch,
                    predictions['distance_logits'],
                    predictions['angle_logits'],
                    collector,
                    prefix='at'
                )
            )

            metrics.update(
                self.batched_dihedral_metrics(
                    batch, predictions['predicted_dihedrals_at'], collector, prefix='at'
                )
            )

        # =========================
        # GRAPH TRANSFORMER PERFORMANCE
        # =========================
        if self.config.use_gt:
            metrics.update(
                self.batched_structural_metrics(
                    batch, predictions['pose_prediction'], collector, prefix='gt'
                )
            )

        # =========================
        # EN TRANSFORMER PERFORMANCE
        # =========================
        if self.config.use_en:
            metrics.update(
                self.batched_structural_metrics(
                    batch, predictions['trajectory'], collector, prefix='en'
                )
            )

            metrics.update(
                self.batched_dihedral_metrics(
                    batch, predictions['predicted_dihedrals_en'], collector, prefix='en'
                )
            )

        return metrics, collector

    def batched_topographical_metrics(self, batch, distance_logits, angle_logits, collector=None, prefix=''):
        metrics = defaultdict(int)
        edge_mask = batch.edge_pad_mask & batch.edge_record_mask

        metrics[f'{prefix}_xe_dis'] = F.cross_entropy(
            distance_logits[edge_mask].squeeze(),
            self.distance_binner(batch.edge_distance[edge_mask])
        )

        edge_angles = batch.edge_angles.clone()
        # edge_angles[batch.edge_distance > self.config.distance_max_radius] = 0
        metrics[f'{prefix}_xe_ang'] = F.cross_entropy(
            rearrange(angle_logits[edge_mask], 'e c l -> (e c) l'),
            self.angle_binner(edge_angles[edge_mask]).flatten()
        )

        if collector is not None:
            edge_distance = unbatch(batch, batch.edge_distance, type='edge')
            distance_logits = unbatch(batch, distance_logits, type='edge')
            for id, gnd, pred in zip(batch.ids, edge_distance, distance_logits):
                collector[id][f'{prefix}_distance_map'] = (self.distance_binner(gnd).cpu(), logit_expectation(pred).cpu())

            edge_angles = unbatch(batch, batch.edge_angles, type='edge')
            angle_logits = unbatch(batch, angle_logits, type='edge')
            for id, gnd, pred, dist in zip(batch.ids, edge_angles, angle_logits, edge_distance):
                gnd[dist > self.config.distance_max_radius] = 0
                gnd = self.angle_binner(gnd)
                collector[id][f'{prefix}_angle_map'] = (gnd.permute(2, 0, 1).cpu(),
                                    logit_expectation(pred).permute(2, 0, 1).cpu())

        return metrics


    def batched_dihedral_metrics(self, batch, predictions, collector=None, prefix=''):
        metrics = {}
        metrics[f'{prefix}_dihedral_loss'] = torsion_angle_loss(pred_points=predictions,
                    true_torsions=batch.angs, alt_true_torsions=batch.alts, angle_mask=batch.angle_record_mask)

        if collector is not None:
            for id, gnd, pred in zip(batch.ids, unbatch(batch, batch.angs), unbatch(batch, predictions)):
                collector[id][f'{prefix}_dihedrals'] = (gnd.cpu(), point_in_circum_to_angle(pred).squeeze(-1).cpu())

            for id, anchor in zip(batch.ids, unbatch(batch, batch.crds[:, :, 1, :])):
                collector[id][f'{prefix}_anchor'] = anchor.cpu()

        return metrics


    def batched_structural_metrics(self, batch, traj, collector=None, prefix=''):
        metrics = defaultdict(int)
        batch_size, trajectory_len, _, _ = traj.size()

        node_mask = batch.node_pad_mask & batch.node_record_mask

        if collector is not None:
            for id in batch.ids: collector[id][f'{prefix}_trajectory'] = list()

        for id, gnd_wrap, pred_traj, mask in zip(batch.ids, batch.crds[:, :, 1, :], traj, node_mask):
            gnd_wrap = gnd_wrap[mask]

            for step, pred_wrap in enumerate(pred_traj):
                pred_wrap = pred_wrap[mask]

                alignment_metrics, (align_gnd_coors, align_pred_coors) = get_alignment_metrics(
                    deepcopy(gnd_wrap),
                    pred_wrap,
                )

                if collector is not None:
                    collector[id][f'{prefix}_trajectory'].append((align_gnd_coors.detach().cpu(),
                                                                  align_pred_coors.detach().cpu()))
                    collector[id][f'{prefix}_alignment_metrics'] = {k: v.mean().cpu().item() for k, v in alignment_metrics.items()}

                for k, metric in alignment_metrics.items():
                    metrics[f'{prefix}_{k}'] += metric.mean() / trajectory_len / batch_size

            if trajectory_len > 1:
                for k, metric in alignment_metrics.items():
                    metrics[f'{prefix}_final_{k}'] += metric.mean() / batch_size

        return metrics
