import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
from Bio import SeqIO
import itertools
import string

from random import shuffle
import numpy as np

import os
import pickle

from tqdm.auto import tqdm, trange

import numpy as np

import esm


deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

import random
import argparse


def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized


def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)


def read_sequence(filename):
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to embed msas')
    parser.add_argument('--split', default=0)
    parser.add_argument('--n_splits', default=1)
    parser.add_argument('--source_dir', default='/home/gridsan/allanc/data/')
    parser.add_argument('--nseq', type=int, default=64)
    parser.add_argument('--representation_depth', default=33)
    parser.add_argument('--no_cuda', dest='no_cuda', action='store_true')

    args = parser.parse_args()

    split = int(args.split)
    n_splits = int(args.n_splits)

    nseq = args.nseq
    representation_depth = args.representation_depth

    data_path = str(args.source_dir)
    print(f'Processing {data_path}!')

    dir = os.path.join(data_path, 'msas')
    encoding_dir = os.path.join(data_path, 'trans_encodings')

    if not os.path.exists(encoding_dir): os.mkdir(encoding_dir)

    names = []
    for msa in os.listdir(dir):
        # for msa in os.listdir(os.path.join(dir, section)):
        if msa[-4:] == '.a3m':
            names.append(msa)

    chunks = np.array_split(names, n_splits)
    names = chunks[split]

    print('Aight baby we are loading the ESM-1B')
    sequence_transformer, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    sequence_transformer = sequence_transformer.eval()
    if not args.no_cuda:sequence_transformer = sequence_transformer.cuda()

    batch_converter = alphabet.get_batch_converter()

    torch.set_grad_enabled(False)

    print('Baking some machine learning magic')
    print(names)
    for name in tqdm(names):
        saving_sector = encoding_dir

        if os.path.exists(os.path.join(saving_sector, name[:-4] + '.pyd')):
            print(f'{name} encoding already computed !')
            continue

        seqs = [read_sequence(os.path.join(dir, name))]
        batch_labels, batch_strs, batch_tokens = batch_converter(seqs)

        seq_len = batch_tokens.size(-1) - 1
        if seq_len > 1023: continue

        if not args.no_cuda: batch_tokens = batch_tokens.cuda()

        representations = sequence_transformer(batch_tokens,
                                               repr_layers=[representation_depth],
                                               need_head_weights=True)
        attentions = representations['attentions'][..., 1:-1, 1:-1]
        attentions = attentions[:, -8:, ...]

        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)
        attentions = apc(symmetrize(attentions))
        attentions = attentions.permute(0, 2, 3, 1).squeeze(0)

        upper = torch.triu_indices(seqlen, seqlen, offset=1)

        up_attentions = attentions[upper[0], upper[1]].type(torch.float16)
        embeddings = representations['representations'][representation_depth]
        embeddings = embeddings[0, 1:-1, :].type(torch.bfloat16)

        if not os.path.exists(saving_sector):
            os.makedirs(saving_sector, exist_ok=True)

        with open(os.path.join(saving_sector, name[:-4] + '.pyd'), 'wb') as out:
            pickle.dump((embeddings.cpu(), up_attentions.cpu()), out)
