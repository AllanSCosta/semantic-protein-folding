import os
import argparse
import time
# original script by Kalyan Palepu

def submit_job(split, n_splits, n_cpu, out_dir):
    name = f'msa_split_{split}_val'
    script = f'{os.getcwd()}/workspace/{name}.sh'
    with open(script, 'w') as file:
        preamble = f'#!/bin/bash\n'
        preamble += f'#SBATCH -o workspace/{name}.sh.log\n'
        preamble += f'#SBATCH --cpus-per-task={n_cpu}\n'
        preamble += f'#SBATCH --job-name={name}\n\n'
        preamble += f'module load anaconda/2021a\n'
        file.write(preamble)
        executable = '/home/gridsan/allanc/msa-transformer-folding/msa_generation/generate_msas.py'
        print(f'submitting split {split}!')
        file.write(f'python -u {executable} --n_cpu {n_cpu} --n_splits {n_splits} --split {split} --out_dir {out_dir}\n')
        file.close()
    os.system(f'LLsub {script}')
    time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MSAs from a collection of a3m files")
    parser.add_argument("--n_splits", help="Number of splits to split dataset up into (if using multiple nodes to compute MSAs for portions of dataset)", type=int, default=2)
    parser.add_argument("--n_cpu", help="Number of CPUs to use for hhblits", type=int, default=10)
    parser.add_argument("--out_dir", help="Directory to direct output", default='/home/gridsan/allanc/data/msas')
    args = parser.parse_args()

    print('Submitting Jobs')
    for split in range(args.n_splits):
        submit_job(split, args.n_splits, args.n_cpu, args.out_dir)

    # numbers = [6, 7]
    # for number in numbers:
        # submit_job(number, 25, args.n_cpu, args.out_dir)
    # submit_job(9, 25, args.n_cpu, args.out_dir)
    # submit_job(10, 25, args.n_cpu, args.out_dir)
    # submit_job(11, 25, args.n_cpu, args.out_dir)
