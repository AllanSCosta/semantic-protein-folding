import os
import argparse
import time
# original script by Kalyan Palepu

def submit_job(split, n_splits, n_cpu, source_dir, nseq, no_cuda):
    name = f'msa_enc_{split}'
    script = f'{os.getcwd()}/workspace/{name}.sh'
    with open(script, 'w') as file:
        preamble = f'#!/bin/bash\n'
        preamble += f'#SBATCH --gres=gpu:volta:1\n'
        preamble += f'#SBATCH -o workspace/{name}.sh.log\n'
        preamble += f'#SBATCH --cpus-per-task={3}\n'
        preamble += f'#SBATCH --job-name={name}\n\n'
        preamble += f'module load anaconda/2021a\n'
        file.write(preamble)
        executable = '/home/gridsan/allanc/msa-transformer-folding/msa_encoding/encode_msas.py'
        print(f'submitting split {split}!')
        file.write(f'python -u {executable} --nseq {nseq} --n_splits {n_splits} --split {split} --source_dir {source_dir} {"--no_cuda" if no_cuda else ""}\n')
        file.close()
    os.system(f'LLsub {script}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MSAs from a collection of a3m files")
    parser.add_argument("--n_splits", help="Number of splits to split dataset up into (if using multiple nodes to compute MSAs for portions of dataset)", type=int, default=10)
    parser.add_argument("--n_cpu", help="Number of CPUs to use", type=int, default=1)
    parser.add_argument('--no_cuda', dest='no_cuda', action='store_true')
    parser.add_argument("--nseq", help="Number of MSA sequences to use", type=int, default=64)
    parser.add_argument('--source_dir', default='/home/gridsan/allanc/data/')
    args = parser.parse_args()

    print('Submitting Jobs')
    for split in range(args.n_splits):
        submit_job(split, args.n_splits, args.n_cpu, args.source_dir, args.nseq, args.no_cuda)
        time.sleep(5) # let's not overload the system
    # to recover a dead job execute submit_sob independently
    # with the proper rank and world size
    # for rank in (1, 5, 6, 7, 8, 11):
    #     submit_job(rank, 12, args.n_cpu, args.source_dir, args.nseq, args.no_cuda)
    #     time.sleep(10)
