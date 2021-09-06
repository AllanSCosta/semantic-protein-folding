#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import tempfile
import os
import argparse

from Bio import SeqIO
from time import time

from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MSAs from a collection of fasta files")
    parser.add_argument("--n_splits", help="Number of splits to split dataset up into (if using multiple nodes to compute MSAs for portions of dataset)", type=int, default=1)
    parser.add_argument("--split", help="Number in [0, n_splits - 1], representing which split of the dataset this should compute", type=int, default=0)
    parser.add_argument("--n_cpu", help="Number of CPUs to use for hhblits", type=int, default=10)

    parser.add_argument("--out_dir", help="Directory to direct output", default='/home/gridsan/allanc/data/msas')

    args = parser.parse_args()
    source_fasta = '/home/gridsan/allanc/alphafold2/casp1314seq.fasta'

    records = []
    for record in SeqIO.parse(source_fasta, "fasta"):
        records.append((record.name, record.seq))

    records = records[args.split:len(records):args.n_splits]

    msa_dir = args.out_dir
    workspace_dir = f"workspace"

    if not os.path.exists(msa_dir):
        os.makedirs(msa_dir)

    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)

    tmpdir = os.getenv('TMPDIR')
    executable = "/home/gridsan/allanc/hhblits/bin/hhblits"
    uniclust = local_uniclust = '/home/gridsan/allanc/hhblits/databases/uniclust30_2018_08/'
    tmpdir = None

    if tmpdir:
        # move uniclust to local disk, so we avoid reading from a distributed file system
        local_uniclust = os.path.join(tmpdir, 'uniclust30_2018_08')
        print('copying uniclust')
        start = time()
        os.system(f'rsync --progress -avr {uniclust} {local_uniclust}')
        end = time()
        print(f'{end - start:.3f} seconds to copy uniclust')

    local_uniclust = os.path.join(local_uniclust, 'uniclust30_2018_08')

    first_run = True
    iteration = 0
    for name, seq in tqdm(records):
        # if os.path.exists(f'{msa_dir}/{name}.a3m'):
            # print(f'{name} MSA already computed')
            # continue

        fasta_path = f"{workspace_dir}/{name}.fasta"
        with open(fasta_path, "w") as f:
            f.write(f'>{name}\n')
            f.write(str(seq))

        start = time()
        cmd = f"{executable} -cpu {args.n_cpu} -i {fasta_path} -d {local_uniclust} -oa3m {msa_dir}/{name}.a3m -n 10"
        print(f'executing: \n{cmd}\n')
        result = os.system(cmd)
        end = time()
        print(f'{end - start:.3f} seconds to run')

        if result != 0:
            print(f"Failure on {name}")
