import os, pickle, random

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
torch.multiprocessing.set_sharing_strategy('file_system')

import random
import math

from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph, to_dense_batch, to_dense_adj

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import time
import random
import sidechainnet as scn

import torch.utils
from sidechainnet.utils.sequence import VOCAB, DSSPVocabulary

from collections import defaultdict
from functools import partial

import torch.nn.functional as F

from glob import glob
from tqdm import tqdm

from .mp_nerf_utils import ensure_chirality
from copy import deepcopy

# TRAIN_DATASETS = ['train']
# VALIDATION_DATASETS = ['valid-10', 'valid-70', 'valid-30'] # 'valid-10',  'valid-70' 'valid-90'] # 'valid-20', 'valid-30', 'valid-40', 'valid-50',
# TEST_DATASETS = ['casp13', 'casp12', 'casp14']


TRAIN_DATASETS = ['train']
VALIDATION_DATASETS = [] #['valid-10', 'valid-20', 'valid-30', 'valid-40', 'valid-50', 'valid-70', 'valid-90'] # 'valid-10',  'valid-70' 'valid-90'] # 'valid-20', 'valid-30', ,
TEST_DATASETS = ['casp13', 'casp14'] # ,



DATASETS = TRAIN_DATASETS + VALIDATION_DATASETS + TEST_DATASETS

DSSP_VOCAV = DSSPVocabulary()

import torch.nn.functional as F



def rot_matrix(a, b, c):
    a1, a2 = a - b, c - b
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

res_to_mirror_symmetry = {
    "D": 1,
    "F": 1,
    "Y": 1,
    "E": 2
}

def get_alternative_angles(seq, angles):
    ambiguous = res_to_mirror_symmetry.keys()
    angles = deepcopy(angles)
    for res_idx, res in enumerate(seq):
        if res in ambiguous:
            angles[res_idx][6+res_to_mirror_symmetry[res]] -= math.pi
    return angles

def create_dataloaders(config):
    data = scn.load(local_scn_path=os.path.join(config.dataset_source, 'unified.pkl'))
    loaders = {
        split_name: ProteinDataset(
           data[split_name],
           split_name=split_name,
           dataset_source=config.dataset_source,
           downsample=config.downsample,
           max_seq_len=config.max_seq_len,
           seq_clamp=config.seq_clamp,
           max_workers=max(1, config.num_workers),
           fetch_msa_encodings=(config.use_msa & ~config.topography_giveaway),
           fetch_seq_encodings=(config.use_seq & ~config.topography_giveaway),
        ).make_loader(config.batch_size, config.num_workers)
        for split_name in DATASETS
    }
    return loaders

def load_embedding(name, source, type='seq'):
    encodings_dir = os.path.join(source, f'{type}_encodings')
    filepath = os.path.join(encodings_dir, name + '.pyd')
    if not os.path.exists(filepath): return (None, None)
    with open(filepath, 'rb') as f:
        emb, att = pickle.load(f)
    return emb, att

class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self,
                 scn_data_split,
                 split_name,
                 dataset_source,
                 seq_clamp=184,
                 max_workers=40,
                 downsample=1.0,
                 max_seq_len=256,
                 fetch_msa_encodings=True,
                 fetch_seq_encodings=True,
                 add_sos_eos=False,
                 sort_by_length=False,
                 reverse_sort=True):

        print(f'=============== Loading {split_name}!')
        num_proteins = len(scn_data_split['seq'])
        self.max_seq_len = max_seq_len

        # shuffle
        indices = list(range(num_proteins))
        random.shuffle(indices)
        for key, attribute in scn_data_split.items():
            scn_data_split[key] = list(map(attribute.__getitem__, indices))

        # for training, we consider the downsample flag
        # and prune the dataset by protein size
        if split_name in TRAIN_DATASETS or split_name in VALIDATION_DATASETS:
            random_filter = random.sample(range(num_proteins), int(num_proteins*downsample))
            for key, attribute in scn_data_split.items():
                scn_data_split[key] = list(map(attribute.__getitem__, random_filter))

            length_filter = [idx for idx, seq in enumerate(scn_data_split['seq'])
                                                if len(seq) < self.max_seq_len]
            for key, attribute in scn_data_split.items():
                scn_data_split[key] = list(map(attribute.__getitem__, length_filter))

        # standard SCN approach to handling data
        self.seqs = [VOCAB.str2ints(s, add_sos_eos) for s in scn_data_split['seq']]
        self.str_seqs = scn_data_split['seq']
        self.angs = scn_data_split['ang']
        self.crds = scn_data_split['crd']
        self.ids = scn_data_split['ids']
        self.resolutions = scn_data_split['res']
        self.secs = [DSSP_VOCAV.str2ints(s, add_sos_eos) for s in scn_data_split['sec']]

        self.split_name = split_name
        self.seq_clamp = seq_clamp

        self.fetch_msa_encodings = fetch_msa_encodings
        self.fetch_seq_encodings = fetch_seq_encodings

        # we are keeping all the embeddings in RAM
        if fetch_msa_encodings:
            encodings = process_map(partial(load_embedding, source=dataset_source, type='msa'), self.ids, max_workers=max_workers)
            self.msa_emb, self.msa_att = [list(l) for l in zip(*encodings)]

        if fetch_seq_encodings:
            encodings = process_map(partial(load_embedding, source=dataset_source, type='seq'), self.ids, max_workers=max_workers)
            self.seq_emb, self.seq_att = [list(l) for l in zip(*encodings)]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        new_idx = random.randint(0, len(self)-1)

        if ((self.fetch_msa_encodings and self.msa_emb[idx] is None) or
            (self.fetch_seq_encodings and self.seq_emb[idx] is None)):
            print(f'Warning: {self.ids[idx]} MSA encoding not found')
            return self[new_idx]

        seqs = self.seqs[idx]
        num_nodes = len(seqs)

        crds = torch.FloatTensor(self.crds[idx]).reshape(-1, 14, 3)
        crds = ensure_chirality(crds.unsqueeze(0)).squeeze(0)
        rots = rot_matrix(crds[:, 0], crds[:, 1], crds[:, 2])

        backbone_coords = crds[:, 1, :]
        distance_map = torch.cdist(backbone_coords, backbone_coords)

        upper = torch.triu_indices(num_nodes, num_nodes, offset=1)
        v, u = upper

        # build single directional graph
        datum = Data(
            __num_nodes__=len(seqs),
            ids=self.ids[idx],
            crds=crds,
            rots=rots,
            seqs=torch.LongTensor(seqs),
            angs=torch.FloatTensor(self.angs[idx]),
            str_seqs=self.str_seqs[idx],
            edge_index=upper,
            edge_distance=distance_map[v, u],
        )

        if self.fetch_msa_encodings:
            datum.node_msa_features = self.msa_emb[idx].type(torch.float32)
            datum.edge_msa_features = self.msa_att[idx].type(torch.float32)

            if torch.any(~torch.isfinite(datum.node_msa_features)):
                print(f'Warning: {self.ids[idx]} MSA encoding has NaNs')
                return self[new_idx]

        if self.fetch_seq_encodings:
            datum.node_seq_features=self.seq_emb[idx].type(torch.float32)
            datum.edge_seq_features=self.seq_att[idx].type(torch.float32)
            if torch.any(~torch.isfinite(datum.node_seq_features)):
                print(f'Warning: {self.ids[idx]} SEQ encoding has NaNs')
                return self[new_idx]

        datum.alts = get_alternative_angles(datum.str_seqs, datum.angs)

        if self.seq_clamp < num_nodes and self.split_name not in TEST_DATASETS:
            start = random.randint(0, num_nodes - self.seq_clamp - 1)
            end = start + self.seq_clamp
            nodes_filter = torch.LongTensor(list(range(start, end)))

            if self.fetch_msa_encodings:
                datum.node_msa_features = datum.node_msa_features[nodes_filter]
                _, datum.edge_msa_features = subgraph(nodes_filter, datum.edge_index,
                                        edge_attr=datum.edge_msa_features, relabel_nodes=True)

            if self.fetch_seq_encodings:
                datum.node_seq_features = datum.node_seq_features[nodes_filter]
                _, datum.edge_seq_features = subgraph(nodes_filter, datum.edge_index,
                                        edge_attr=datum.edge_seq_features, relabel_nodes=True)

            edge_index, datum.edge_distance = subgraph(nodes_filter, datum.edge_index,
                                    edge_attr=datum.edge_distance, relabel_nodes=True)
            datum.edge_index = edge_index

            datum.rots = datum.rots[nodes_filter]
            datum.crds = datum.crds[nodes_filter]
            datum.angs = datum.angs[nodes_filter]
            datum.seqs = datum.seqs[nodes_filter]
            datum.alts = datum.alts[nodes_filter]

            datum.__num_nodes__ = self.seq_clamp

        datum = deepcopy(datum)

        if (datum.crds[:, 1, :].norm(dim=-1).gt(1e-6) & datum.crds[:, 1, 0].isfinite()).sum() < 10:
            # we potentially made a slice that doesn't have useful data
            return self[new_idx]

        # add both directions for graph
        datum.edge_distance = torch.cat((datum.edge_distance, datum.edge_distance), dim=0)
        if self.fetch_msa_encodings:
            datum.edge_msa_features = torch.cat((datum.edge_msa_features, datum.edge_msa_features), dim=0)
        if self.fetch_seq_encodings:
            datum.edge_seq_features = torch.cat((datum.edge_seq_features, datum.edge_seq_features), dim=0)


        v, u = datum.edge_index
        datum.edge_index = torch.cat((datum.edge_index, torch.stack((u, v))), dim=-1)

        # compute final 3d angles
        v, u = datum.edge_index
        datum.edge_vectors = torch.einsum('b p, b p q -> b q', backbone_coords[u] - backbone_coords[v],
                                                                datum.rots[v].transpose(-1, -2))
        datum.edge_angles = F.normalize(datum.edge_vectors, dim=-1)

        return datum

    def __str__(self):
        """Describe this dataset to the user."""
        return (f"ProteinDataset( "
                f"split='{self.split_name}', "
                f"n_proteins={len(self)}, ")

    def __repr__(self):
        return self.__str__()

    def make_loader(self, batch_size, num_workers, data_structure='batch'):
        loader = DataLoader(
            self,
            batch_size=batch_size if self.split_name not in TEST_DATASETS else 1,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        return loader


def collate_fn(stream):
    batch = Batch.from_data_list(stream)
    for key in ('seqs', 'node_msa_features', 'node_seq_features', 'crds', 'angs', 'alts'):
        if batch[key] is None: continue
        batch[key], mask = to_dense_batch(batch[key], batch=batch.batch)
    for key in ('edge_msa_features', 'edge_seq_features', 'edge_distance', 'edge_angles'):
        if batch[key] is None: continue
        batch[key] = to_dense_adj(edge_index=batch.edge_index,
                                batch=batch.batch, edge_attr=batch[key])

    # used for both models and losses
    batch.node_pad_mask = mask
    batch.edge_pad_mask = (mask[:, None, :] & mask[:, :, None])

    # used for losses
    batch.node_record_mask = batch.crds[:, :, 1].norm(dim=-1).gt(1e-6) & batch.crds[:, :, 1, 0].isfinite()
    batch.angle_record_mask = batch.angs.ne(0.0) & batch.angs.isfinite()
    batch.edge_record_mask = batch.edge_distance.gt(0) & batch.edge_angles.sum(-1).ne(0)

    return batch
