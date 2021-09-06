import sys
import os
import wandb
os.environ["WANDB_MODE"] = "dryrun"

import sys
sys.path.append('/home/gridsan/allanc/msa-transformer-folding')

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from data import create_dataloaders
from model import create_model
from trainer import Trainer

from utils import submit_script

torch.autograd.set_detect_anomaly(True)

from visualization import plot_predicted_angles

import argparse

if __name__ == '__main__':
    with torch.no_grad():
        torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description='Fold some proteins.')


    # ========================
    # GENERAL
    # ========================

    parser.add_argument('--debug', dest='debug', default=False, action='store_true')
    parser.add_argument('--submit', dest='submit', default=False, action='store_true')
    parser.add_argument('--name', type=str, default='anonymous-glyder')
    parser.add_argument('--note', type=str, default='nonote')
    parser.add_argument('--report_frequency', type=int, default=20)

    # ========================
    # DATA
    # ========================

    parser.add_argument('--dataset_source', type=str, default='/home/gridsan/allanc/data')
    parser.add_argument('--downsample', type=float, default=0.8)
    parser.add_argument('--seq_clamp', type=int, default=128)
    parser.add_argument('--max_seq_len', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)


    # ========================
    # ARCHITECTURE
    # ========================

    parser.add_argument('--wiring_checkpoint', type=int, default=0)
    parser.add_argument('--topography_giveaway', type=int, default=0)
    parser.add_argument('--wipe_edge_information', type=int, default=0)

    parser.add_argument('--giveaway_distance_resolution', type=int, default=32)
    parser.add_argument('--giveaway_angle_resolution', type=int, default=16)

    parser.add_argument('--use_msa', type=int, default=1)
    parser.add_argument('--use_seq', type=int, default=1)

    parser.add_argument('--use_at', type=int, default=0)
    parser.add_argument('--use_gt', type=int, default=1)
    parser.add_argument('--use_en', type=int, default=1)

    # ESM-MSA-1 DISTILLATION
    parser.add_argument('--node_msa_distill_layers', nargs='+', type=int, default=[768, 256, 256, 128])
    parser.add_argument('--edge_msa_distill_layers', nargs='+', type=int, default=[96, 64, 64])

    # ESM-1B DISTILLATION
    parser.add_argument('--node_seq_distill_layers', nargs='+', type=int, default=[1280, 256, 128])
    parser.add_argument('--edge_seq_distill_layers', nargs='+', type=int, default=[160, 64, 64])

    # ENSEMBLE
    parser.add_argument('--node_ens_distill_layers', nargs='+', type=int, default=[128, 128, 128])
    parser.add_argument('--edge_ens_distill_layers', nargs='+', type=int, default=[64, 64])

    # AXIAL TRANSFORMER
    parser.add_argument('--at_checkpoint',type=int, default=0)
    parser.add_argument('--at_dim', type=int, default=32)
    parser.add_argument('--at_depth',type=int, default=3)
    parser.add_argument('--at_heads',type=int, default=1)
    parser.add_argument('--at_dim_head',type=int, default=32)
    parser.add_argument('--at_window_size', type=int, default=64)

    # GRAPH TRANSFORMER
    parser.add_argument('--gt_checkpoint',type=int, default=1)
    parser.add_argument('--gt_dim', type=int, default=64)
    parser.add_argument('--gt_edim', type=int, default=32)
    parser.add_argument('--gt_depth',type=int, default=3)
    parser.add_argument('--gt_heads',type=int, default=1)
    parser.add_argument('--gt_dim_head',type=int, default=64)

    # EN TRANSFORMER
    parser.add_argument('--gaussian_noise', type=float, default=10)
    parser.add_argument('--et_checkpoint',type=int, default=0)
    parser.add_argument('--et_dim', type=int, default=64)
    parser.add_argument('--et_edim', type=int, default=32)
    parser.add_argument('--et_depth',type=int, default=3)
    parser.add_argument('--et_heads',type=int, default=4)
    parser.add_argument('--et_dim_head',type=int, default=64)
    parser.add_argument('--et_coors_hidden_dim',type=int, default=64)
    parser.add_argument('--en_num_neighbors', type=int, default=0)
    parser.add_argument('--en_num_seq_neighbors', type=int, default=64)


    # FOLDING STEPS
    parser.add_argument('--unroll_steps', type=int, default=10)
    parser.add_argument('--train_fold_steps', type=int, default=1)
    parser.add_argument('--eval_fold_steps', type=int, default=60)


    # ========================
    # OPTIMIZATION
    # ========================

    # PREDICTIONS
    parser.add_argument('--angle_number_of_bins', type=int, default=16)
    parser.add_argument('--distance_number_of_bins', type=int, default=24)
    parser.add_argument('--distance_max_radius', type=float, default=24)

    # OPTIM
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--at_loss_coeff', type=float, default=30.0)
    parser.add_argument('--gt_loss_coeff', type=float, default=5.0)
    parser.add_argument('--et_loss_coeff', type=float, default=10.0)

    parser.add_argument('--et_drmsd', type=int, default=0)

    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--validation_check_rate', type=int, default=10)
    parser.add_argument('--validation_start', type=int, default=20)


    # STOCHASTICITY
    parser.add_argument('--coordinate_reset_prob', type=float, default=0.3)
    parser.add_argument('--msa_wipe_out_prob', type=float, default=0.8)
    parser.add_argument('--msa_wipe_out_dropout', type=float, default=0.1)

    # ========================
    # TEST
    # ========================

    parser.add_argument('--test_model', type=str, default='-')
    parser.add_argument('--retrain_model', type=str, default='-')

    config = parser.parse_args()
    if config.submit:
        submit_script(os.path.realpath(__file__), os.getcwd(), config)
        exit()

    if config.test_model != '-' or config.debug: config.downsample = 0.001

    if config.topography_giveaway:
        config.use_msa = 0
        config.use_seq = 0

    if not config.debug:
        wandb.init(
            reinit=True,
            name=config.name,
            config=config,
            project='StructureExtractor',
        )

    loaders = create_dataloaders(config)
    for datum in loaders['casp13']: break

    if config.test_model != '-':
        model = create_model(config)
        model.load_state_dict(torch.load(config.test_model))
        trainer = Trainer(config, model, loaders)
        trainer.test()
    else:
        model = create_model(config)
        if config.retrain_model != '-':
            print(f'Loading weights from {config.retrain_model}')
            model.load_state_dict(torch.load(config.retrain_model))
        trainer = Trainer(config, model, loaders)
        trainer.train()
        trainer.test()

    print('Done.')
