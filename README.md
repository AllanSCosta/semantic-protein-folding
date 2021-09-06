
# Semantic Protein Folding

<img src="https://github.com/AllanSCosta/semantic-protein-folding/blob/main/img/T1024%20fold.gif" width="300" height="300" /><img src="https://github.com/AllanSCosta/semantic-protein-folding/blob/main/img/T1024%20rear.gif" width="300" height="300" />

## A pipeline for Protein Language Modeling + Protein Folding experimentation


Based on [Distillation of MSA Embeddings to Protein Folded Structures](https://www.biorxiv.org/content/10.1101/2021.06.02.446809v1.full.pdf).

This repository stands on shoulders of giant work by the scientific community:

- [Evolutionary Scale Modeling](https://github.com/facebookresearch/esm)
- [SidechainNet](https://github.com/jonathanking/sidechainnet)
- [Massively Parallel Natural Extension of Reference Frame](https://github.com/EleutherAI/mp_nerf)
- [Axial-Attention](https://github.com/lucidrains/En-transformer/)
- [En-Transformer](https://github.com/lucidrains/En-transformer/)
- [Graph-Transformer](https://github.com/lucidrains/graph-transformer-pytorch)
- [E3NN](https://github.com/e3nn/e3nn)

For experimental and prototypical access to internal code, these repos are collected under `building_blocks` (except sidechainnet). As development progresses they will be incorporated as original `imports`.


### Experimental Arguments

#### General
- `debug`: whether to use wandb logging
- `submit`: submit training to a SLURM scheduler
- `name`: name of experiment
- `note`: experimental note
- `report_frequency`: how often to log metrics in log file

#### Data
- `dataset_source`: path to sidechainnet
- `downsample`: whether to uniformly at random downsample dataset
- `seq_clamp`: clamp data at the sequence level, size of clamp
- `max_seq_len`: throw out data with sequences larger than max_seq_len
- `num_workers`: number of CPU workers for data fetching and loading
- `batch_size`: batch size


#### Architecture

topography_giveaway:
wipe_edge_information:
giveaway_distance_resolution:
giveaway_angle_resolution:

wiring_checkpoint:

use_msa
use_seq

use_at
use_gt
use_en

#### ESM-MSA-1 DISTILLATION
node_msa_distill_layers [768, 256, 256, 128]
edge_msa_distill_layers', nargs='+', type=int, default=[96, 64, 64])

#### ESM-1B DISTILLATION
node_seq_distill_layers', nargs='+', type=int, default=[1280, 256, 128])
edge_seq_distill_layers', nargs='+', type=int, default=[160, 64, 64])

#### ENSEMBLE
node_ens_distill_layers', nargs='+', type=int, default=[128, 128, 128])
edge_ens_distill_layers', nargs='+', type=int, default=[64, 64])

#### AXIAL TRANSFORMER
at_checkpoint',type=int, default=0)
at_dim', type=int, default=32)
at_depth',type=int, default=3)
at_heads',type=int, default=1)
at_dim_head',type=int, default=32)
at_window_size', type=int, default=64)

#### GRAPH TRANSFORMER
gt_checkpoint',type=int, default=1)
gt_dim', type=int, default=64)
gt_edim', type=int, default=32)
gt_depth',type=int, default=3)
gt_heads',type=int, default=1)
gt_dim_head',type=int, default=64)

#### EN TRANSFORMER
gaussian_noise', type=float, default=10)
et_checkpoint',type=int, default=0)
et_dim', type=int, default=64)
et_edim', type=int, default=32)
et_depth',type=int, default=3)
et_heads',type=int, default=4)
et_dim_head',type=int, default=64)
et_coors_hidden_dim',type=int, default=64)
en_num_neighbors', type=int, default=0)
en_num_seq_neighbors', type=int, default=64)


#### FOLDING STEPS
unroll_steps', type=int, default=10)
train_fold_steps', type=int, default=1)
eval_fold_steps', type=int, default=60)



#### PREDICTIONS
angle_number_of_bins', type=int, default=16)
distance_number_of_bins', type=int, default=24)
distance_max_radius', type=float, default=24)

#### OPTIM
lr', type=float, default=1e-4)

at_loss_coeff', type=float, default=30.0)
gt_loss_coeff', type=float, default=5.0)
et_loss_coeff', type=float, default=10.0)

et_drmsd', type=int, default=0)

max_epochs', type=int, default=10)
validation_check_rate', type=int, default=10)
validation_start', type=int, default=20)


#### STOCHASTICITY
coordinate_reset_prob', type=float, default=0.3)
msa_wipe_out_prob', type=float, default=0.8)
msa_wipe_out_dropout', type=float, default=0.1)

test_model', type=str, default='-')
retrain_model', type=str, default='-')