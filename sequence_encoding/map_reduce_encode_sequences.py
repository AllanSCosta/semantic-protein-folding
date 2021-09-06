import os
import argparse

# original script by Kalyan Palepu

def submit_job(split, n_splits, n_cpu, source_dir):
    name = f'seq_enc_{split}'
    script = f'{os.getcwd()}/workspace/{name}.sh'
    with open(script, 'w') as file:
        preamble = f'#!/bin/bash\n'
        preamble += f'#SBATCH --gres=gpu:volta:1\n'
        preamble += f'#SBATCH -o workspace/{name}.sh.log\n'
        preamble += f'#SBATCH --cpus-per-task={3}\n'
        preamble += f'#SBATCH --job-name={name}\n\n'
        preamble += f'module load anaconda/2021a\n'
        file.write(preamble)
        executable = '/home/gridsan/allanc/msa-transformer-folding/sequence_encoding/encode_sequences.py'
        print(f'submitting split {split}!')
        file.write(f'python -u {executable} --n_splits {n_splits} --split {split} --source_dir {source_dir} \n')
        file.close()
    os.system(f'LLsub {script}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MSAs from a collection of a3m files")
    parser.add_argument("--n_splits", help="Number of splits to split dataset up into (if using multiple nodes to compute MSAs for portions of dataset)", type=int, default=10)
    parser.add_argument("--n_cpu", help="Number of CPUs to use", type=int, default=1)
    parser.add_argument('--source_dir', default='/home/gridsan/allanc/data/')
    args = parser.parse_args()


    print('Submitting Jobs')
    for split in range(args.n_splits):
        submit_job(split, args.n_splits, args.n_cpu, args.source_dir)

    # to recover a dead job execute submit_sob independently
    # with the proper rank and world size
    # submit_job(5, 10, args.n_cpu, args.source_dir)
