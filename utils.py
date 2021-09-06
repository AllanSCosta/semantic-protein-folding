from copy import deepcopy
from torch_geometric.data import Data, Batch
import torch

from collections import defaultdict

import torch.nn.functional as F
from einops import rearrange

def logit_expectation(logits):
    probs = F.softmax(logits, dim=-1)
    value = torch.arange(0, logits.size(-1), device=logits.device)
    expectation = (probs * value).mean(-1)
    return expectation

def wrap(trace):
    size = list(trace.size()[:-1]) + [14, 3]
    wrapper = torch.zeros(*size, device=trace.device)
    wrapper[..., 1, :] = trace
    return wrapper

def point_in_circum_to_angle(points):
    """ Converts a point in the circumference to an angle
        Inputs:
        * poits: (any, 2)
        Outputs: (any)
    """
    # ensure first dim
    if len(points.shape) == 1:
        points = points.unsqueeze(0)

    return torch.atan2(points[..., points.shape[-1] // 2:],
                       points[..., :points.shape[-1] // 2] )

def angle_to_point_in_circum(angles):
    """ Converts an angle to a point in the unit circumference.
        Inputs:
        * angles: tensor of (any) shape.
        Outputs: (any, 2)
    """
    # ensure no last dummy dim
    if len(angles.shape) == 0:
        angles = angles.unsqueeze(0)
    elif angles.shape[-1] == 1 and len(angles.shape) > 1 :
        angles = angles[..., 0]

    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)


def torsion_angle_loss(pred_torsions=None, true_torsions=None,
                       pred_points=None,  true_points=None,
                       alt_true_points=None, alt_true_torsions=None,
                       coeff=2., norm_coeff=1e-2, angle_mask=None):
    if true_torsions is not None and true_points is None:
        true_points = angle_to_point_in_circum(true_torsions)
    if alt_true_torsions is not None and alt_true_points is None:
        alt_true_points = angle_to_point_in_circum(alt_true_torsions)

    # calc norm of angles
    norm = torch.norm(pred_points, dim=-1)
    angle_norm_loss = norm_coeff * (1-norm).abs()

    # do L2 on unit circle
    pred_points = pred_points / norm.unsqueeze(-1)
    torsion_loss = torch.pow(pred_points - true_points, 2).sum(dim=-1)

    if alt_true_points is not None:
        torsion_loss = torch.minimum(
            torsion_loss,
            torch.pow(pred_points - alt_true_points, 2).sum(dim=-1)
        )
    if coeff != 2.:
        torsion_loss *= coeff/2

    if angle_mask is None:
        angle_mask = torch.ones(*pred_points.shape[:-1], dtype=torch.bool)

    return (torsion_loss + angle_norm_loss)[angle_mask].mean()


def l1_loss_torch(pred_translations, true_translations, edge_index,
                  max_val=10., l_func=None, epsilon=1e-4):
    v, u = edge_index
    induced_distance = norm(pred_translations[v] - pred_translations[u], dim=-1, ord=2)
    gnd_distance = norm(true_translations[v] - true_translations[u], dim=-1, ord=2)
    l1loss = F.smooth_l1_loss(induced_distance, gnd_distance)
    return l1loss

def drmsd_torch(pred_translations, true_translations, edge_index,
                  max_val=10., l_func=None, epsilon=1e-4):
    v, u = edge_index
    induced_distance = norm(pred_translations[v] - pred_translations[u], dim=-1, ord=2)
    gnd_distance = norm(true_translations[v] - true_translations[u], dim=-1, ord=2)
    drmsd = F.mse_loss(induced_distance, gnd_distance).sqrt()
    return drmsd

def discretize(measurements, start, end, number_of_bins):
    values = torch.linspace(start, end, number_of_bins + 2)
    step = (values[1] - values[0])
    bins = (values[1:-1]).to(measurements.device)
    diff = (measurements[..., None] - bins) / step
    return torch.argmin(torch.abs(diff), dim=-1)

#
#
# def backbone_fape_torch(pred_translations, pred_rotations, true_translations, true_rotations, edge_index,
#                         max_val=10., l_func=None, epsilon=1e-4):
#     """ Computes FAPE on C-alphas
#         Inputs:
#         * pred_translations: ((...), b, 3) float. vector positions
#         * pred_rotations: ((...), b, 3, 3) float. rotation matrices
#         * true_translations: ((...), b, 3) float. vector positions
#         * true_rotations: ((...), b, 3, 3) float. rotation matrices
#         * edge_index: (2, E) long. mappings
#         * max_val: float. maximum to clamp loss at.
#         * l_func: function. allows for customization beyond L1
#         * epsilon: float. small const to keep grads stable
#     """
#     if l_func is None: l_func = lambda x, y, eps=1e-7, sup = max_val: (((x-y)**2).sum(dim=-1) + eps).sqrt()
#     i, j = edge_index
#     xij_hat = torch.einsum('b p, b p q-> b q', pred_translations[j] - pred_translations[i], pred_rotations[i].transpose(-1, -2))
#     xij = torch.einsum('b p, b p q-> b q', true_translations[j] - true_translations[i], true_rotations[i].transpose(-1, -2))
#     dij = l_func(xij, xij_hat)
#     l_fape = torch.mean(torch.clamp(dij, min=0, max=10)) / max_val
#     return l_fape

def soft_one_hot_linspace(x, start, end, number, basis=None, cutoff=None):
    r"""Projection on a basis of functions
    """
    if cutoff not in [True, False]:
        raise ValueError("cutoff must be specified")

    if not cutoff:
        values = torch.linspace(start, end, number, dtype=x.dtype, device=x.device)
        step = values[1] - values[0]
    else:
        values = torch.linspace(start, end, number + 2, dtype=x.dtype, device=x.device)
        step = values[1] - values[0]
        values = values[1:-1]

    diff = (x[..., None] - values) / step
    if basis == 'gaussian':
        return diff.pow(2).neg().exp().div(1.12)

    if basis == 'cosine':
        return torch.cos(math.pi/2 * diff) * (diff < 1) * (-1 < diff)

    if basis == 'fourier':
        x = (x[..., None] - start) / (end - start)
        if not cutoff:
            i = torch.arange(0, number, dtype=x.dtype, device=x.device)
            return torch.cos(math.pi * i * x) / math.sqrt(0.25 + number / 2)
        else:
            i = torch.arange(1, number + 1, dtype=x.dtype, device=x.device)
            return torch.sin(math.pi * i * x) / math.sqrt(0.25 + number / 2) * (0 < x) * (x < 1)

    if basis == 'bessel':
        x = x[..., None] - start
        c = end - start
        bessel_roots = torch.arange(1, number + 1, dtype=x.dtype, device=x.device) * math.pi
        out = math.sqrt(2 / c) * torch.sin(bessel_roots * x / c) / x

        if not cutoff:
            return out
        else:
            return out * ((x / c) < 1) * (0 < x)

    raise ValueError(f"basis=\"{basis}\" is not a valid entry")




# ATOMS
valid_elements = ['C', 'O', 'N', 'S']
element_to_integer_map = defaultdict(lambda: len(valid_elements)+1)
element_to_integer_map.update({el: i+1 for i, el in enumerate(valid_elements)})


integer_to_single = ["A", "R", "N", "D", "C", "E", "Q",  "G",  "H", "I", "L", "K", "M", "F",  "P", "S", "T", "W", "Y", "V"]

integer_to_single_map = defaultdict(lambda: 'U')
integer_to_single_map.update({i:val for i, val in enumerate(integer_to_single)})

single_to_integer_map = defaultdict(lambda: 21)
single_to_integer_map.update({val:i for i, val in enumerate(integer_to_single)})


single_to_triple = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "E": "GLU",
    "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL", "U": "UNK"
}

single_to_triple_map = defaultdict(lambda: "UNK")
single_to_triple_map.update(single_to_triple)

triple_to_single_map = defaultdict(lambda: "U")
triple_to_single_map.update({v: k for k, v in single_to_triple_map.items()})

integer_to_triple = {i:single_to_triple_map[integer_to_single_map[i]] for i in range(len(integer_to_single))}

triple_to_integer_map = defaultdict(lambda: 21) # 20 is unknown, 21 is a mask
triple_to_integer_map.update({v : k for k, v in integer_to_triple.items()})

atom_format_str = ("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}"
                           "{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}")
defaults = {
    "alt_loc": "",
    "chain_id": "A",
    "insertion_code": "",
    "occupancy": 1,
    "temp_factor": 0,
    "element_sym": "",
    "charge": ""
}


import randomname
def random_name():
    return randomname.get_name(
        adj=('speed', 'emotions', 'temperature'),
        noun=('astronomy', 'set_theory', 'military_navy', 'infrastructure')
    )

import os
import time

def submit_script(script_path, base_path, params):
    params = dict(vars(params))
    name = random_name()
    params['name'] = name
    worskpace_dir = os.path.join(base_path, 'workspace')
    os.makedirs(worskpace_dir, exist_ok=True)
    script = os.path.join(worskpace_dir, f'{name}.sh')
    with open(script, 'w') as file:
        preamble = f'#!/bin/bash\n'
        preamble += f'#SBATCH --gres=gpu:volta:1\n'
        preamble += f'#SBATCH -o {os.path.join(worskpace_dir, name)}.sh.log\n'
        preamble += f'#SBATCH --cpus-per-task=20\n'
        preamble += f'#SBATCH --job-name={name}\n\n'
        preamble += f'module load anaconda/2021a\n'
        file.write(preamble)
        params = [(key, value) for key, value in params.items() if (key != 'submit' and key != 'debug' and key != 'sample_dataset')]
        params_strings = [f'--{key} {str(value) if type(value) != list else " ".join([str(v) for v in value])}' for key, value in params]
        params_string = ' '.join(params_strings)
        file.write(f'python -u {script_path} {params_string}')
        file.close()
    os.system(f'LLsub {script}')
    print(f'submitted {name}!')


def produce_backbone_pdb(title, coords, seq):
    header = f'REMARK  {title}\n'
    body = ""
    seq, coords = seq.detach().cpu().tolist(), coords.detach().cpu().tolist()
    for index, (res, coord) in enumerate(zip(seq, coords)):
        body +=  atom_format_str.format(
                        "ATOM", index, 'CA', defaults["alt_loc"],
                        integer_to_triple[res], defaults["chain_id"], index,
                        defaults["insertion_code"], coord[0], coord[1],
                        coord[2], defaults['occupancy'], defaults["temp_factor"], 'C',
                        defaults["charge"]
        )
        body += '\n'
    footer = "TER\nEND          \n"
    return header + body + footer


def unbatch(batch):
    batch = deepcopy(batch)
    keys = {key for key in batch.__dict__.keys() if ((('node' in key) or ('edge' in key)) and (key in batch.__slices__))}
    data_list = []
    for i, (num_nodes, cumsum) in enumerate(zip(batch.__num_nodes_list__,
                                                batch.__cumsum__['edge_index'])):
        data = Data()
        for key in keys:
            slices = batch.__slices__[key]
            start, end = slices[i], slices[i+1]
            data[key] = (batch[key][start:end]
                         if key != 'edge_index' else batch[key][...,start:end])
        # data.__num_nodes__ = num_nodes
        data.edge_index -= cumsum
        data_list.append(Batch.from_data_list([data]))
    return data_list


TS_CUTOFFS = [1, 2, 4, 8]
HA_CUTOFFS = [0.5, 1, 2, 4]


def get_alignment_metrics(X, Y):
    X, Y = kabsch_torch(X, Y)
    rmsd = rmsd_torch(X, Y)
    gdt_ts = gdt_torch(X, Y, TS_CUTOFFS)
    gdt_ha = gdt_torch(X, Y, HA_CUTOFFS)

    induced_vectors = rearrange(X, 'i d -> i () d') - rearrange(X, 'j d -> () j d')
    induced_distances = induced_vectors.norm(p=2, dim=-1)
    true_vectors = rearrange(Y, 'i d -> i () d') - rearrange(Y, 'j d -> () j d')
    true_distances = true_vectors.norm(p=2, dim=-1)

    drmsd = F.mse_loss(induced_distances, true_distances).sqrt()
    dl1 = F.smooth_l1_loss(induced_distances, true_distances)

    deviations = (X - Y).norm(p=2, dim=-1)
    beta = 0.3
    cond = deviations < beta
    cl1 = torch.where(cond, 0.5 * deviations ** 2 / beta, deviations - 0.5 * beta).mean()

    return {'rmsd': rmsd, 'gdt_ts': gdt_ts, 'gdt_ha': gdt_ha, 'drmsd': drmsd, 'dl1': dl1, 'cl1': cl1}, (X, Y)


# code below is modified from: https://github.com/lucidrains/alphafold2/blob/main/alphafold2_pytorch/utils.py
def gdt_torch(X, Y, cutoffs, weights=None):
    """ Assumes x,y are both (B x D x N). see below for wrapper.
        * cutoffs is a list of `K` thresholds
        * weights is a list of `K` weights (1 x each threshold)
    """
    device = X.device
    X, Y = X.transpose(0, 1), Y.transpose(0, 1)
    # set zeros and fill with values
    GDT = torch.zeros(len(cutoffs), device=device)
    dist = ((X - Y)**2).sum(dim=0).sqrt()
    # iterate over thresholds
    for i,cutoff in enumerate(cutoffs):
        GDT[i] = (dist <= cutoff).float().mean(dim=-1)
    # weighted mean
    return (GDT).mean(-1)

def rmsd_torch(X, Y):
    """ Assumes x,y are both (N x D). See below for wrapper. """
    return torch.sqrt(torch.mean(torch.sum((X - Y)**2 , dim=-1), dim=-1) + 1e-8)

def kabsch_torch(X, Y, cpu=True):
    # Assumes X,Y are both (N_points x Dims). See below for wrapper.
    device = X.device
    X, Y = X.transpose(0, 1), Y.transpose(0, 1)

    #  center X and Y to the origin
    X_ = X - X.mean(dim=-1, keepdim=True)
    Y_ = Y - Y.mean(dim=-1, keepdim=True)
    # calculate convariance matrix (for each prot in the batch)
    C = torch.matmul(X_, Y_.t()).detach()
    if cpu:
        C = C.cpu()

    # Optimal rotation matrix via SVD
    if int(torch.__version__.split(".")[1]) < 8:
        # warning! int torch 1.<8 : W must be transposed
        V, S, W = torch.svd(C)
        W = W.t()
    else:
        V, S, W = torch.linalg.svd(C)

    # determinant sign for direction correction
    d = (torch.det(V) * torch.det(W)) < 0.0
    if d:
        S[-1]    = S[-1] * (-1)
        V[:, -1] = V[:, -1] * (-1)
    # Create Rotation matrix U
    U = torch.matmul(V, W).to(device)
    # calculate rotations
    X_ = torch.matmul(X_.t(), U).t()
    # return centered and aligned
    return X_.transpose(0, 1), Y_.transpose(0, 1)
