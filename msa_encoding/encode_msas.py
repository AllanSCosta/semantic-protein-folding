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

def remove_insertions(sequence):
    return sequence.translate(translation)


def hamming_distance(string1, string2):
    return sum(c1 != c2 for c1, c2 in zip(string1, string2))

def read_msa(filename, nseq, type='random'):
    records = [(record.description, remove_insertions(str(record.seq)))
                for record in SeqIO.parse(filename, "fasta")]
    source, queries = records[0], records[1:]

    queries = [(hamming_distance(source[1], query[1]), query) for query in queries]
    queries = [query[1] for query in sorted(queries)]

    if type == 'max_hamming':
        msa = queries[-(nseq-1):]
    elif type == 'min_hamming':
        msa = queries[:nseq-1]
    else:
        shuffle(queries)
        msa = queries[:nseq-1]

    msa = [source] + msa
    return msa

deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

import random
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to embed msas')
    parser.add_argument('--split', default=0)
    parser.add_argument('--n_splits', default=1)
    parser.add_argument('--source_dir', default='/home/gridsan/allanc/data/')
    parser.add_argument('--nseq', type=int, default=64)
    parser.add_argument('--representation_depth', default=11)
    parser.add_argument('--no_cuda', dest='no_cuda', action='store_true')

    args = parser.parse_args()

    split = int(args.split)
    n_splits = int(args.n_splits)

    nseq = args.nseq
    representation_depth = args.representation_depth

    data_path = str(args.source_dir)
    print(f'Processing {data_path}!')

    msa_dir = os.path.join(data_path, 'msas')
    encoding_dir = os.path.join(data_path, 'msa_trans_encodings')

    if not os.path.exists(encoding_dir): os.mkdir(encoding_dir)

    names = []
    for msa in os.listdir(msa_dir):
        # for msa in os.listdir(os.path.join(msa_dir, section)):
        if msa[-4:] == '.a3m':
            names.append(msa)

    chunks = np.array_split(names, n_splits)
    names = chunks[split]

    print('Aight baby we are loading the MSA Transformer')
    msa_transformer, msa_alphabet = esm.pretrained.esm_msa1_t12_100M_UR50S()
    msa_transformer = msa_transformer.eval()
    if not args.no_cuda: msa_transformer = msa_transformer.cuda()

    msa_batch_converter = msa_alphabet.get_batch_converter()

    torch.set_grad_enabled(False)

    print('Baking some machine learning magic')
    print(names)
    for name in tqdm(names):
        saving_sector = encoding_dir

        if os.path.exists(os.path.join(saving_sector, name[:-4] + '.pyd')):
            print(f'{name} encoding already computed !')
            continue

        msa = [read_msa(os.path.join(msa_dir, name), nseq)]
        msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa)

        seq_len = msa_batch_tokens.size(-1) - 1
        if seq_len > 1023: continue

        if not args.no_cuda: msa_batch_tokens = msa_batch_tokens.cuda()

        representations = msa_transformer(msa_batch_tokens,
                                          repr_layers=[representation_depth],
                                          need_head_weights=True)

        attentions = representations['row_attentions'][..., 1:, 1:]
        attentions = attentions[:, -8:, ...]
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

        # attentions = apc(symmetrize(attentions))
        attentions = attentions.permute(0, 2, 3, 1)
        attentions = attentions.squeeze(0)

        upper = torch.triu_indices(seqlen, seqlen, offset=1)

        up_attentions = attentions[upper[0], upper[1]].type(torch.float16)
        embeddings = representations['representations'][representation_depth]
        embeddings = embeddings[:, 0, 1:, :].squeeze(0).type(torch.bfloat16)

        if not os.path.exists(saving_sector):
            os.makedirs(saving_sector, exist_ok=True)

        with open(os.path.join(saving_sector, name[:-4] + '.pyd'), 'wb') as out:
            pickle.dump((embeddings.cpu(), up_attentions.cpu()), out)
