
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


## Experimentation Pipeline 

<img src="https://github.com/AllanSCosta/semantic-protein-folding/blob/main/img/full_pipeline.png" width="900" height="500" />


#### General
- `debug`: whether to use wandb logging
- `submit`: submit training to a SLURM scheduler
- `name`: name of experiment
- `note`: experimental note
- `report_frequency`: how often to log metrics in log file

#### Data
- `dataset_source`: path to sidechainnet-formatted dataset
- `downsample`: whether to uniformly at random downsample dataset
- `seq_clamp`: clamp data at the sequence level, size of clamp
- `max_seq_len`: throw out data with sequences larger than max_seq_len
- `num_workers`: number of CPU workers for data fetching and loading
- `batch_size`: batch size


### Architecture

- `wipe_edge_information`: drops out all hij
- `topography_giveaway`: instead of providing language-model-based hij, produces hij based on ground truth distance and orientation
- `giveaway_distance_resolution`: number of bins of relative distance information to input
- `giveaway_angle_resolution`: number of bins of relative orientation information to input

- `wiring_checkpoint`: checkpoint model-inbetween Dense networks 

- `use_msa`: use ESM-MSA-1 embeddings
- `use_seq`: use ESM-1b embeddings

- `use_at`: process hij with an Axial Transformer after distillation
- `use_gt`: project 3D coordinates with Graph Transformer after distillation
- `use_en`: refine with E(n)-Transformer given coords

#### ESM-MSA-1 Distillation
- `node_msa_distill_layers`: hidden layer enumeration of Dense for msa node information extraction [768, 256, 256, 128]
- `edge_msa_distill_layers`: hidden layer enumeration of Dense for msa edge information extraction [96, 64, 64]

#### ESM-1B Distillation
- `node_seq_distill_layers`: hidden layer enumeration of Dense for msa node information extraction [1280, 256, 128]
- `edge_seq_distill_layers`: hidden layer enumeration of Dense for msa edge information extraction [160, 64, 64]

#### Seq + MSA ensemble
- `node_ens_distill_layers`: hidden layer enumeration of Dense for msa node information extraction [128, 128, 128]
- `edge_ens_distill_layers`: hidden layer enumeration of Dense for msa edge information extraction [64, 64]

#### AXIAL TRANSFORMER
- `at_checkpoint`: if the axial transformer should be checkpointed
- `at_dim`: axial transformer dim
- `at_depth`: axial transformer depth
- `at_heads`: axial transformer number of attention heads
- `at_dim_head`: axial transformer dim head
- `at_window_size`: axial transformer window size (for internal Long-Short optimization)

#### GRAPH TRANSFORMER
- `gt_checkpoint`: graph transformer checkpoint
- `gt_dim`: graph transformer dim
- `gt_edim`: graph transformer edge dim
- `gt_depth`: graph transformer depth
- `gt_heads`: graph transformer number of heads
- `gt_dim_head`: graph trnasformer dim head

#### EN TRANSFORMER
- `gaussian_noise`: if graph transformer is not used, which gaussian noise to be added to backbone as starting point
- `et_checkpoint`: checkpoint en transformer
- `et_dim`: dim of en transformer
- `et_edim`: en transformer edge dim
- `et_depth`: en transformer depth
- `et_heads`: en transformer num heads
- `et_dim_head`: en transformer dim head
- `et_coors_hidden_dim`: hidden dim of internal coordinate-head mixer 
- `en_num_neighbors`: num neighbors to consider in 3d space
- `en_num_seq_neighbors`: num neighbors to consider in sequence space


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
