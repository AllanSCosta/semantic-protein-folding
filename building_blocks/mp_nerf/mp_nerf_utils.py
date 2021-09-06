# Author: Eric Alcaide

from functools import wraps

import torch
import numpy as np
from einops import repeat, rearrange

from .kb_proteins import *
import sidechainnet
import prody as pr

# only needed for sparse nth_deg adj calculation
try:
    import torch_sparse
    TORCH_SPARSE = True
except:
    pass


# random hacks - angle conversions

to_zero_two_pi = lambda x: ( x + (2*np.pi) * ( 1 + torch.floor_divide(x.abs(), 2*np.pi) ) ) % (2*np.pi)
def to_pi_minus_pi(x):
    zero_two_pi = to_zero_two_pi(x)
    return torch.where(
        zero_two_pi < np.pi, zero_two_pi, -(2*np.pi - zero_two_pi)
    )

# random hacks - device utils for pyTorch - saves transfers
to_cpu = lambda x: x.cpu() if x.is_cuda else x
to_device = lambda x, device: x.to(device) if x.device != device else x

# avoid conversion to TF32 on newer GPUs
def switch_tf32(target):
    torch.backends.cuda.matmul.allow_tf32 = target


@torch.jit.script
def cdist(x,y):
    """ robust cdist - drop-in for pytorch's. """
    return torch.pow(
        x.unsqueeze(-3) - y.unsqueeze(-2), 2
    ).sum(dim=-1).clamp(min=1e-7).sqrt()

########################
## general decorators ##
########################

def switch_tf32(target):
    torch.backends.cuda.matmul.allow_tf32 = target

def expand_dims_to(t, length = 3):
    if length == 0:
        return t
    return t.reshape(*((1,) * length), *t.shape) # will work with both torch and numpy


def set_backend_kwarg(fn):
    @wraps(fn)
    def inner(*args, backend = 'auto', **kwargs):
        if backend == 'auto':
            backend = 'torch' if isinstance(args[0], torch.Tensor) else 'numpy'
        kwargs.update(backend = backend)
        return fn(*args, **kwargs)
    return inner


def invoke_torch_or_numpy(torch_fn, numpy_fn):
    def outer(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            backend = kwargs.pop('backend')
            passed_args = fn(*args, **kwargs)
            passed_args = list(passed_args)
            if isinstance(passed_args[-1], dict):
                passed_kwargs = passed_args.pop()
            else:
                passed_kwargs = {}
            backend_fn = torch_fn if backend == 'torch' else numpy_fn
            return backend_fn(*passed_args, **passed_kwargs)
        return inner
    return outer


################
## data utils ##
################

def read_pdb(filename, chain="A"):
    """ Extracts sequence and angles from pdb. """
    keys = ["angles_np", "coords_np", "observed_sequence"]
    chain = pr.parsePDB(filename, chain=chain, model=1)
    parsed = sidechainnet.utils.measure.get_seq_coords_and_angles(chain)
    data = {k:v for k,v in zip(keys, parsed)}
    return data


def get_prot(dataloader_=None, vocab_=None, min_len=80, max_len=150,
             verbose=True, subset="train"):
    """ Gets a protein from sidechainnet and returns
        the right attrs for training.
        Inputs:
        * dataloader_: sidechainnet iterator over dataset
        * vocab_: sidechainnet VOCAB class
        * min_len: int. minimum sequence length
        * max_len: int. maximum sequence length
        * verbose: bool. verbosity level
        * subset: str. which subset to load proteins from.
        Outputs: (cleaned, without padding)
        (seq_str, int_seq, coords, angles, padding_seq, mask, pid)
    """
    while True:
        for b,batch in enumerate(dataloader_[subset]):
            for i in range(batch.int_seqs.shape[0]):
                # skip too short
                if batch.int_seqs[i].shape[0] < min_len:
                    continue

                # strip padding - matching angles to string means
                # only accepting prots with no missing residues (mask is 0)
                padding_seq = (batch.int_seqs[i] == 20).sum().item()
                padding_mask = -(batch.msks[i] - 1).sum().item() # find 0s

                if padding_seq == padding_mask:
                    # check for appropiate length
                    real_len = batch.int_seqs[i].shape[0] - padding_seq
                    if max_len >= real_len >= min_len:
                        # strip padding tokens
                        seq = batch.str_seqs[i] # seq is already unpadded - see README at scn repo
                        int_seq = batch.int_seqs[i][:-padding_seq or None]
                        angles  = batch.angs[i][:-padding_seq or None]
                        mask    = batch.msks[i][:-padding_seq or None]
                        coords  = batch.crds[i][:-padding_seq*14 or None]

                        if verbose:
                            print("stopping at sequence of length", real_len)

                        yield seq, int_seq, coords, angles, padding_seq, mask, batch.pids[i]
                    else:
                        if verbose:
                            print("found a seq of length:", batch.int_seqs[i].shape,
                                  "but oustide the threshold:", min_len, max_len)
                else:
                    if verbose:
                        print("paddings not matching", padding_seq, padding_mask)
                    pass
    return None


###################
## Metrics utils ##
###################


def rmsd_torch(X, Y):
    """ Assumes x,y are both (B x D x N). See below for wrapper. """
    return torch.sqrt( ((X - Y)**2).sum(-2).mean(-1) + 1e-8 )


def rmsd_numpy(X, Y):
    """ Assumes x,y are both (B x D x N). See below for wrapper. """
    return np.sqrt( np.mean( ((X - Y)**2).sum(axis=-2), axis=-1) )


def drmsd_torch(X, Y):
    """ Assumes x,y are both (B x D x N). See below for wrapper. """
    X_ = X.transpose(-1, -2)
    Y_ = Y.transpose(-1, -2)
    x_dist = cdist(X_, X_) # (B, N, N)
    y_dist = cdist(Y_, Y_) # (B, N, N)

    return torch.sqrt( torch.pow(x_dist-y_dist, 2).mean(dim=(-1, -2)) )


def drmsd_numpy(X, Y, eps=1e-7):
    """ Assumes x,y are both (B x D x N). See below for wrapper. """
    X_ = np.transpose(X, axes=(-1,-2))
    Y_ = np.transpose(Y, axes=(-1,-2))
    x_dist = np.sqrt((X_[None, :, :] - X_[:, None, :])**2 + 1e-7).sum(axis=-1) # (B, N, N)
    y_dist = np.sqrt((Y_[None, :, :] - Y_[:, None, :])**2).sum(axis=-1) # (B, N, N)

    return np.sqrt( ((x_dist-y_dist)**2).mean(axis=(-1, -2)) )


def gdt_torch(X, Y, cutoffs, weights=None):
    """ Assumes x,y are both (B x D x N). see below for wrapper.
        * cutoffs is a list of `K` thresholds
        * weights is a list of `K` weights (1 x each threshold)
    """
    device = X.device
    if weights is None:
        weights = torch.ones(1,len(cutoffs)).to(device)
    else:
        weights = torch.tensor([weights]).to(device)
    # set zeros and fill with values
    GDT = torch.zeros(X.shape[0], len(cutoffs), device=device)
    dist = ((X - Y)**2).sum(dim=1).sqrt()
    # iterate over thresholds
    for i,cutoff in enumerate(cutoffs):
        GDT[:, i] = (dist <= cutoff).float().mean(dim=-1)
    # weighted mean
    return (GDT*weights).mean(-1)


def gdt_numpy(X, Y, cutoffs, weights=None):
    """ Assumes x,y are both (B x D x N). see below for wrapper.
        * cutoffs is a list of `K` thresholds
        * weights is a list of `K` weights (1 x each threshold)
    """
    if weights is None:
        weights = np.ones( (1,len(cutoffs)) )
    else:
        weights = np.array([weights])
    # set zeros and fill with values
    GDT = np.zeros( (X.shape[0], len(cutoffs)) )
    dist = np.sqrt( ((X - Y)**2).sum(axis=1) )
    # iterate over thresholds
    for i,cutoff in enumerate(cutoffs):
        GDT[:, i] = (dist <= cutoff).mean(axis=-1)
    # weighted mean
    return (GDT*weights).mean(-1)


def tmscore_torch(X, Y):
    """ Assumes x,y are both (B x D x N). see below for wrapper. """
    L = max(15, X.shape[-1])
    d0 = 1.24 * (L - 15)**(1./3.) - 1.8
    # get squared of distance
    dist = ((X - Y)**2).sum(dim=1).sqrt()
    # formula (see wrapper for source):
    return (1 / ( 1 + (dist/d0)**2 )).mean(dim=-1)


def tmscore_numpy(X, Y):
    """ Assumes x,y are both (B x D x N). see below for wrapper. """
    L = max(15, X.shape[-1])
    d0 = 1.24 * np.cbrt(L - 15) - 1.8
    # get distance
    dist = np.sqrt( ((X - Y)**2).sum(axis=1) )
    # formula (see wrapper for source):
    return (1 / (1 + (dist/d0)**2)).mean(axis=-1)


def lddt_torch(true_coords, pred_coords, cloud_mask, select_atom="ca", r_0=15.):
    """ Computes the lddt score for each atom_selector.
        https://academic.oup.com/bioinformatics/article/29/21/2722/195896

        CAUTION: ONLY CA is supported for now

        Inputs:
        * true_coords: (b, l, c, d) in sidechainnet format.
        * pred_coords: (b, l, c, d) in sidechainnet format.
        * cloud_mask : (b, l, c) adapted for scn format.
        * select_atom: str. option for atom selction in `proteins.atom_seclector`
        * r_0: float. maximum inclusion radius in reference struct.
        Outputs:
        * (b, l) lddt for c_alpha scores (ranging between 0 and 1)
        See wrapper below.
    """
    device, dtype = true_coords.device, true_coords.type()
    thresholds = torch.tensor([0.5, 1, 2, 4], device=device).type(dtype)
    # adapt masks
    # target_c_alphas, c_alpha_mask = atom_selector(true_coords, masks=cloud_mask, option="ca")

    # hardcode C_A
    atom_mask = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).to(cloud_mask.device)
    c_alpha_mask = (cloud_mask * atom_mask[None, None, ...]).bool()
    target_c_alphas = true_coords[c_alpha_mask]

    # container for c_alpha scores (between 0,1)
    wrapper = torch.zeros(true_coords.shape[:2], device=device).type(dtype)

    for bi, seq in enumerate(true_coords):
        # select atoms for study
        selected_pred = pred_coords[bi, c_alpha_mask[bi], :]
        selected_target = true_coords[bi, c_alpha_mask[bi], :]
        # get number under distance
        dist_mat_pred = cdist(selected_pred, selected_pred)
        dist_mat_target = cdist(selected_target, selected_target)
        under_r0_target = dist_mat_target < r_0
        compare_dists = torch.abs(dist_mat_pred - dist_mat_target)[under_r0_target]
        # measure diff below threshold
        score = torch.zeros_like(under_r0_target).float()
        max_score = torch.zeros_like(under_r0_target).float()
        max_score[under_r0_target] = 4.
        # measure under how many thresholds
        score[under_r0_target] = thresholds.shape[0] - \
                                 torch.bucketize( compare_dists, boundaries=thresholds ).float()
        # dont include diagonal
        l_mask = c_alpha_mask.float().sum(dim=-1).bool()
        wrapper[bi, l_mask[bi]] = ( score.sum(dim=-1) - thresholds.shape[0] ) / \
                                  ( max_score.sum(dim=-1) - thresholds.shape[0] )

    return wrapper


######################
## structural utils ##
######################

def get_dihedral(c1, c2, c3, c4):
    """ Returns the dihedral angle in radians.
        Will use atan2 formula from:
        https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
        Inputs:
        * c1: (batch, 3) or (3,)
        * c2: (batch, 3) or (3,)
        * c3: (batch, 3) or (3,)
        * c4: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2
    u3 = c4 - c3

    return torch.atan2( ( (torch.norm(u2, dim=-1, keepdim=True) * u1) * torch.cross(u2,u3, dim=-1) ).sum(dim=-1) ,
                        (  torch.cross(u1,u2, dim=-1) * torch.cross(u2, u3, dim=-1) ).sum(dim=-1) )


def get_angle(c1, c2, c3):
    """ Returns the angle in radians. Uses atan2 formula
        Inputs:
        * c1: (batch, 3) or (3,)
        * c2: (batch, 3) or (3,)
        * c3: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2

    # dont use acos since norms involved.
    # better use atan2 formula: atan2(cross, dot) from here:
    # https://johnblackburne.blogspot.com/2012/05/angle-between-two-3d-vectors.html

    # add a minus since we want the angle in reversed order - sidechainnet issues
    return torch.atan2( torch.norm(torch.cross(u1,u2, dim=-1), dim=-1),
                        -(u1*u2).sum(dim=-1) )


def get_cosine_angle(c1, c2, c3, eps=1e-7):
    """ Returns the angle in radians. Uses cosine formula
        Not all angles are possible all the time.
        Inputs:
        * c1: (batch, 3) or (3,)
        * c2: (batch, 3) or (3,)
        * c3: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2

    return torch.acos( (u1*u2).sum(dim=-1)  / (u1.norm(dim=-1)*u2.norm(dim=-1) + eps))


def circular_mean(angles, weights=None):
    """ Circular mean of angles.
        Inputs:
        * angles (..., d) float tensor of any shape
        * weights (..., d) float tensor of same shape as angles
        Outputs: float tensor (..., )
    """
    if weights is None:
        weights = torch.ones_like(angles)
    return torch.atan2( (torch.sin(angles) * weights).sum(dim=-1),
                        (torch.cos(angles) * weights).sum(dim=-1), )


def kabsch_torch(X, Y, rot_mat=False):
    """ Kabsch alignment of X into Y.
        Assumes X,Y are both (D, N) - usually (3, N)
        * rot_mat: bool. whether to return rotation matrix.
    """
    #  center X and Y to the origin
    X_ = X - X.mean(dim=-1, keepdim=True)
    Y_ = Y - Y.mean(dim=-1, keepdim=True)
    # calculate convariance matrix (for each prot in the batch)
    C = torch.matmul(X_, Y_.t())
    # Optimal rotation matrix via SVD - warning! W must be transposed
    if int(torch.__version__.split(".")[1]) < 8:
        V, S, W = torch.svd(C.detach())
        W = W.t()
    else:
        V, S, W = torch.linalg.svd(C.detach())
    # determinant sign for direction correction
    d = (torch.det(V) * torch.det(W)) < 0.0
    if d:
        S[-1]    = S[-1] * (-1)
        V[:, -1] = V[:, -1] * (-1)
    # Create Rotation matrix U
    U = torch.matmul(V, W)
    # calculate rotations
    X_ = torch.matmul(X_.t(), U).t()
    # return centered and aligned
    if rot_mat:
        return X_, Y_, U
    return X_, Y_


def kabsch_numpy(X, Y):
    """ Kabsch alignment of X into Y.
        Assumes X,Y are both (Dims x N_points). See below for wrapper.
    """
    # center X and Y to the origin
    X_ = X - X.mean(axis=-1, keepdims=True)
    Y_ = Y - Y.mean(axis=-1, keepdims=True)
    # calculate convariance matrix (for each prot in the batch)
    C = np.dot(X_, Y_.transpose())
    # Optimal rotation matrix via SVD
    V, S, W = np.linalg.svd(C)
    # determinant sign for direction correction
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1]    = S[-1] * (-1)
        V[:, -1] = V[:, -1] * (-1)
    # Create Rotation matrix U
    U = np.dot(V, W)
    # calculate rotations
    X_ = np.dot(X_.T, U).T
    # return centered and aligned
    return X_, Y_


def center_bins(bins, last_bin=None):
    """ Recovers the central estimate from bin limits.
        Inputs:
        * bins: (N,) tensor. the separations for the different bins
        * last_bin: float or None
        Outputs: (N,) tensor.
    """
    diffs = bins[1:] - bins[:-1]

    if not last_bin:
        last_bin = diffs.mean().item()
    last_bin = torch.tensor([last_bin], device=bins.device)
    new_bins = torch.cat([ diffs, last_bin ], dim=0)
    new_bins = expand_dims_to(new_bins, len(histogram.shape) - len(new_bins.shape))

    return new_bins


def center_distogram_torch(distogram, bins, min_t=1., center="mean", wide="inverse"):
    """ Returns the central estimate of a distogram. Median for now.
        Inputs:
        * distogram: (batch, N, N, B) where B is the number of buckets.
        * bins: (B,) containing the cutoffs for the different buckets
        * min_t: float. lower bound for distances.
        * center: str. strategy for centering (mean, median)
        * wide: str (std, var, argmax, inverse).
                strategy to measure uncertainty
        Outputs:
        * central: (batch, N, N)
        * dispersion: (batch, N, N)
        * weights: (batch, N, N)
    """
    shape, device = distogram.shape, distogram.device
    # threshold to weights and find mean value of each bin
    n_bins = center_bins(bins).to(device)
    max_bin_allowed = torch.tensor(n_bins.shape[0]-1).to(device).long()
    # calculate measures of centrality and dispersion -
    magnitudes = distogram.sum(dim=-1)
    if center == "median":
        cum_dist = torch.cumsum(distogram, dim=-1)
        medium   = 0.5 * cum_dist[..., -1:]
        central  = torch.searchsorted(cum_dist, medium).squeeze()
        central  = n_bins[ torch.min(central, max_bin_allowed) ]
    elif center == "mean":
        central  = (distogram * n_bins).sum(dim=-1) / magnitudes
    # create mask for last class - (IGNORE_INDEX)
    mask = (central <= bins[-2].item()).float()
    # mask diagonal to 0 dist - don't do masked filling to avoid inplace errors
    diag_idxs = np.arange(shape[-2])
    central   = expand_dims_to(central, 3 - len(central.shape))
    central[:, diag_idxs, diag_idxs]  *= 0.
    # provide weights
    if wide == "var":
        dispersion = (distogram * (n_bins - central.unsqueeze(-1))**2).sum(dim=-1) / magnitudes
    elif wide == "std":
        dispersion = ((distogram * (n_bins - central.unsqueeze(-1))**2).sum(dim=-1) / magnitudes).sqrt()
    elif wide == "argmax":
        dispersion  = torch.argmax(distogram, dim=-1)
    elif wide == "inverse":
        dispersion  = central.clone()
    else:
        dispersion = torch.zeros_like(central, device=device)
    # rescale to 0-1. lower std / var  --> weight=1. set potential nan's to 0
    weights = mask / (1+dispersion)
    weights[weights != weights] *= 0.
    weights[:, diag_idxs, diag_idxs] *= 0.
    return central, weights


def mds_torch(pre_dist_mat, weights=None, iters=10, tol=1e-5, eigen=False, verbose=2):
    """ Gets distance matrix. Outputs 3d. See below for wrapper.
        Assumes (for now) distogram is (N x N) and symmetric
        https://github.com/scikit-learn/scikit-learn/blob/42aff4e2e/sklearn/manifold/_mds.py#L279
        Outs:
        * best_3d_coords: (batch x 3 x N)
        * historic_stresses: (batch x steps)
    """
    device, dtype = pre_dist_mat.device, pre_dist_mat.type()
    # ensure batched MDS
    pre_dist_mat = expand_dims_to(pre_dist_mat, length = ( 3 - len(pre_dist_mat.shape) ))
    # start
    batch, N, _ = pre_dist_mat.shape
    diag_idxs = np.arange(N)
    his = [torch.tensor([np.inf]*batch, device=device)]

    # initialize by eigendecomposition: https://www.lptmc.jussieu.fr/user/lesne/bioinformatics.pdf
    # follow : https://www.biorxiv.org/content/10.1101/2020.11.27.401232v1.full.pdf
    D = pre_dist_mat**2
    M =  0.5 * (D[:, :1, :] + D[:, :, :1] - D)
    # do loop svd bc it's faster: (2-3x in CPU and 1-2x in GPU)
    # https://discuss.pytorch.org/t/batched-svd-lowrank-being-much-slower-than-loop-implementation-both-cpu-and-gpu/119336
    svds = [torch.svd_lowrank(mi) for mi in M]
    u, s, v = [ torch.stack([svd[i] for svd in svds], dim=0) for i in range(3) ]
    best_3d_coords = torch.bmm(u, torch.diag_embed(s).abs().sqrt())[..., :3]

    # only eigen - way faster but not weights
    if weights is None and eigen==True:
        return torch.transpose( best_3d_coords, -1, -2), torch.zeros_like(torch.stack(his, dim=0))
    elif eigen==True:
        if verbose:
            print("Can't use eigen flag if weights are active. Fallback to iterative")

    # continue the iterative way
    if weights is None:
        weights = torch.ones_like(pre_dist_mat)

    # iterative updates:
    for i in range(iters):
        # compute distance matrix of coords and stress
        best_3d_coords = best_3d_coords.contiguous()
        dist_mat = cdist(best_3d_coords, best_3d_coords).clone()

        stress   = ( weights * (dist_mat - pre_dist_mat)**2 ).sum(dim=(-1,-2)) * 0.5
        # perturb - update X using the Guttman transform - sklearn-like
        dist_mat[ dist_mat <= 0 ] += 1e-7
        ratio = weights * (pre_dist_mat / dist_mat)
        B = -ratio
        B[:, diag_idxs, diag_idxs] += ratio.sum(dim=-1)

        # update
        coords = (1. / N * torch.matmul(B, best_3d_coords))
        dis = torch.norm(coords, dim=(-1, -2))

        if verbose >= 2:
            print('it: %d, stress %s' % (i, stress))
        # update metrics if relative improvement above tolerance
        if (his[-1] - stress / dis).mean() <= tol:
            if verbose:
                print('breaking at iteration %d with stress %s' % (i,
                                                                   stress / dis))
            break

        best_3d_coords = coords
        his.append( stress / dis )

    return torch.transpose(best_3d_coords, -1,-2), torch.stack(his, dim=0)


def mds_numpy(pre_dist_mat, weights=None, iters=10, tol=1e-5, eigen=False, verbose=2):
    """ Gets distance matrix. Outputs 3d. See below for wrapper.
        Assumes (for now) distrogram is (N x N) and symmetric
        Out:
        * best_3d_coords: (3 x N)
        * historic_stress
    """
    if weights is None:
        weights = np.ones_like(pre_dist_mat)

    # ensure batched MDS
    pre_dist_mat = expand_dims_to(pre_dist_mat, length = ( 3 - len(pre_dist_mat.shape) ))
    # start
    batch, N, _ = pre_dist_mat.shape
    his = [np.inf]
    # init random coords
    best_stress = np.inf * np.ones(batch)
    best_3d_coords = 2*np.random.rand(batch, 3, N) - 1
    # iterative updates:
    for i in range(iters):
        # compute distance matrix of coords and stress
        dist_mat = np.linalg.norm(best_3d_coords[:, :, :, None] - best_3d_coords[:, :, None, :], axis=-3)
        stress   = (( weights * (dist_mat - pre_dist_mat) )**2).sum(axis=(-1, -2)) * 0.5
        # perturb - update X using the Guttman transform - sklearn-like
        dist_mat[dist_mat == 0] = 1e-7
        ratio = weights * (pre_dist_mat / dist_mat)
        B = -ratio
        B[:, np.arange(N), np.arange(N)] += ratio.sum(axis=-1)
        # update - double transpose. TODO: consider fix
        coords = (1. / N * np.matmul(best_3d_coords, B))
        dis = np.linalg.norm(coords, axis=(-1, -2))
        if verbose >= 2:
            print('it: %d, stress %s' % (i, stress))
        # update metrics if relative improvement above tolerance
        if (best_stress - stress / dis).mean() <= tol:
            if verbose:
                print('breaking at iteration %d with stress %s' % (i,
                                                                   stress / dis))
            break

        best_3d_coords = coords
        best_stress = stress / dis
        his.append(best_stress)

    return best_3d_coords, np.array(his)


def calc_phis_torch(pred_coords, N_mask, CA_mask, C_mask=None,
                    prop=True, verbose=0):
    """ Does not work with batches, but pass the extra dim.

        Filters mirrors selecting the 1 with most N of negative phis.

        Used as part of the MDScaling wrapper if arg is passed. See below.
        Angle Phi between planes: (Cterm{-1}, N, Ca{0}) and (N{0}, Ca{+1}, Cterm{+1})
        Inputs:
        * pred_coords: (batch, 3, N) predicted coordinates
        * N_mask: (batch, N) boolean mask for N-term positions
        * CA_mask: (batch, N) boolean mask for C-alpha positions
        * C_mask: (batch, N) or None. boolean mask for C-alpha positions or
                    automatically calculate from N_mask and CA_mask if None.
        * prop: bool. whether to return as a proportion of negative phis.
        * verbose: bool. verbosity level
        Output: (batch, N) containing the phi angles or (batch,) containing
                the proportions.
        Note: use [0] since all prots in batch have same backbone
    """
    # detach gradients for angle calculation - mirror selection
    pred_coords_ = torch.transpose(pred_coords.detach(), -1 , -2).cpu()
    if C_mask is None:
        C_mask = ~ ( N_mask + CA_mask )
    # select points
    n_terms  = pred_coords_[:, N_mask.squeeze()]
    c_alphas = pred_coords_[:, CA_mask.squeeze()]
    c_terms  = pred_coords_[:, C_mask.squeeze()]
    # compute phis for every pritein in the batch
    phis = [get_dihedral_torch(c_terms[i, :-1],
                               n_terms[i,  1:],
                               c_alphas[i, 1:],
                               c_terms[i,  1:]) for i in range(pred_coords.shape[0])]

    # return percentage of lower than 0
    if prop:
        return torch.stack([(x<0).float().mean() for x in phis], dim=0 )
    return phis


def ensure_chirality(coords_wrapper, use_backbone=True):
    """ Ensures protein agrees with natural distribution
        of chiral bonds (ramachandran plots).
        Reflects ( (-1)*Z ) the ones that do not.
        Inputs:
        * coords_wrapper: (B, L, C, 3) float tensor. First 3 atoms
                          in C should be N-CA-C
        * use_backbone: bool. whether to use the backbone (better, more robust)
                              if provided, or just use c-alphas.
        Ouputs: (B, L, C, 3)
    """

    # detach gradients for angle calculation - mirror selection
    coords_wrapper_ = coords_wrapper.detach()
    mask = coords_wrapper_.abs().sum(dim=(-1, -2)) != 0.

    # if BB present: use bb dihedrals
    if coords_wrapper[:, :, 0].abs().sum() != 0. and use_backbone:
        # compute phis for every protein in the batch
        phis = get_dihedral(
            coords_wrapper_[:, :-1, 2], # C_{i-1}
            coords_wrapper_[:, 1: , 0], # N_{i}
            coords_wrapper_[:, 1: , 1], # CA_{i}
            coords_wrapper_[:, 1: , 2], # C_{i}
        )

        # get proportion of negatives
        props = [(phis[i, mask[i, :-1]] > 0).float().mean() for i in range(mask.shape[0])]

        # fix mirrors by (-1)*Z if more (+) than (-) phi angles
        corrector = torch.tensor([ [1, 1, -1 if p > 0.5 else 1]  # (B, 3)
                                   for p in props ], dtype=coords_wrapper.dtype)

    # if only CA trace - similar to : https://arxiv.org/pdf/2105.04771v1.pdf
    else:
        # gen dihedrals from empirical distribution
        sampler = EmpiricalDistribution(
            probs=torch.tensor(CA_TRACE_DIHEDRALS["probs"], device=mask.device),
            centers=torch.tensor(CA_TRACE_DIHEDRALS["centers"], device=mask.device)
        )

        coords_wrapper_mirror = coords_wrapper * \
                                torch.tensor([[[[1, 1, -1]]]]).float().to(coords_wrapper.device)
        dihedrals = get_dihedral(
            coords_wrapper[..., :-3 , 1, :],
            coords_wrapper[..., 1:-2, 1, :],
            coords_wrapper[..., 2:-1, 1, :],
            coords_wrapper[..., 3:  , 1, :],
        )
        dihedrals_mirror = get_dihedral(
            coords_wrapper_mirror[..., :-3 , 1, :],
            coords_wrapper_mirror[..., 1:-2, 1, :],
            coords_wrapper_mirror[..., 2:-1, 1, :],
            coords_wrapper_mirror[..., 3:  , 1, :],
        )
        target_distro = sampler.sample(n=mask.shape[-1]).to(mask.device)
        # calc KL of normal struct || # dihedrals = n_atoms-3
        kl_normal = [ torch.kl_div(
                        input=dihedrals[b, mask[b, 2:-1]],
                        target=target_distro[:mask[b, 2:-1].sum()]
                     ).mean() for b in range(coords_wrapper.shape[0]) ]

        # calc KL of mirrored struct || # dihedrals = n_atoms-3
        kl_mirror = [ torch.kl_div(
                        input=dihedrals_mirror[b, mask[b, 2:-1]],
                        target=target_distro[:mask[b, 2:-1].sum()]
                      ).mean() for b in range(coords_wrapper.shape[0]) ]

        # fix mirrors by (-1)*Z if lower kl in reverse mode
        corrector = torch.tensor([ [1, 1, -1 if mirror < normal else 1]
                                   for mirror,normal in zip(kl_mirror, kl_normal) ],
                                   dtype=coords_wrapper.dtype)

    return coords_wrapper * corrector.to(coords_wrapper.device)[:, None, None, :]


def correct_mirror_torch(structs, N_mask, CA_mask, C_mask=None):
    """ Corrects protein handedness based on dihedral distro.
        * structs: (batch, 3, N)
        * N_mask: (batch, N) boolean mask for N-term positions
        * CA_mask: (batch, N) boolean mask for C-alpha positions
        * C_mask: (batch, N) or None. boolean mask for C-alpha positions or
                    automatically calculate from N_mask and CA_mask if None.
    """
    phi_ratios = calc_phis_torch(structs, N_mask, CA_mask, C_mask, prop=True)
    to_correct = torch.nonzero( (phi_ratios < 0.5)).view(-1)
    # fix mirrors by (-1)*Z if more (+) than (-) phi angles
    structs[to_correct, -1] = (-1)*structs[to_correct, -1]
    return structs


def mdscaling_torch(pre_dist_mat, weights=None, iters=10, tol=1e-5,
                    fix_mirror=True, N_mask=None, CA_mask=None, C_mask=None,
                    eigen=False, verbose=2):
    """ Handles the specifics of MDS for proteins (mirrors, ...) """
    # batched mds for full parallel
    preds, stresses = mds_torch(pre_dist_mat, weights=weights,iters=iters,
                                              tol=tol, eigen=eigen, verbose=verbose)

    if fix_mirror:
        preds = correct_mirror_torch(preds, N_mask, CA_mask, C_mask)

    return preds, stresses


def align_overlap(points_base, points_unord, coords, cloud_mask):
    """ Aligns coords2 onto coords via an overlapped region.
        Effective if region is many points.
        Otherwise opt for frame reorientation.
        Inputs:
        * points_base: (N, 3). Points in the frame to be aligned to.
        * points_unord: (N, 3). Points in the unaligned frame.
        * coords: (L_, C, 3)
        * cloud_mask: (L_, C)

        Outputs:
        * aligned coords: (L_, C, 3)
        * aligned points_unord: (N, 3)
    """
    # align overlapped and get rotation matrix
    pb_, pu_, rot = kabsch_torch(points_base.t(),
                                 points_unord.t(), rot_mat=True)
    # rotate unorderd points
    points_rotated = ( coords.reshape(-1,3) - \
                       points_unord.mean(dim=0, keepdim=True) ) @ rot

    points_rotated = points_rotated.reshape(coords.shape) + \
                     points_base.mean(dim=0, keepdim=True)
    points_rotated[~cloud_mask] = 0.

    return points_rotated, pu_


######################
## Backend Wrappers ##
######################

def MDScaling(pre_dist_mat, **kwargs):
    """ Gets distance matrix (-ces). Outputs 3d.
        Assumes (for now) distrogram is (N x N) and symmetric.
        For support of ditograms: see `center_distogram_torch()`
        Inputs:
        * pre_dist_mat: (1, N, N) distance matrix.
        * weights: optional. (N x N) pairwise relative weights .
        * iters: number of iterations to run the algorithm on
        * tol: relative tolerance at which to stop the algorithm if no better
               improvement is achieved
        * backend: one of ["numpy", "torch", "auto"] for backend choice
        * fix_mirror: int. number of iterations to run the 3d generation and
                      pick the best mirror (highest number of negative phis)
        * N_mask: indexing array/tensor for indices of backbone N.
                  Only used if fix_mirror > 0.
        * CA_mask: indexing array/tensor for indices of backbone C_alpha.
                   Only used if fix_mirror > 0.
        * verbose: whether to print logs
        Outputs:
        * best_3d_coords: (3 x N)
        * historic_stress: (timesteps, )
    """
    pre_dist_mat = expand_dims_to(pre_dist_mat, 3 - len(pre_dist_mat.shape))
    return pre_dist_mat, kwargs

@set_backend_kwarg
@invoke_torch_or_numpy(kabsch_torch, kabsch_numpy)
def Kabsch(A, B):
    """ Returns Kabsch-rotated matrices resulting
        from aligning A into B.
        Adapted from: https://github.com/charnley/rmsd/
        * Inputs:
            * A,B are (3 x N)
            * backend: one of ["numpy", "torch", "auto"] for backend choice
        * Outputs: tensor/array of shape (3 x N)
    """
    # run calcs - pick the 0th bc an additional dim was created
    return A, B


@set_backend_kwarg
@invoke_torch_or_numpy(rmsd_torch, rmsd_numpy)
def RMSD(A, B):
    """ Returns RMSD score as defined here (lower is better):
        https://en.wikipedia.org/wiki/
        Root-mean-square_deviation_of_atomic_positions
        * Inputs:
            * A,B are (B x 3 x N) or (3 x N)
            * backend: one of ["numpy", "torch", "auto"] for backend choice
        * Outputs: tensor/array of size (B,)
    """
    return A, B


@set_backend_kwarg
@invoke_torch_or_numpy(drmsd_torch, drmsd_numpy)
def dRMSD(A, B):
    """ Returns dRMSD score (root mean of the squared diff between
        distance matrices.
        Root-mean-square_deviation_of_atomic_positions
        * Inputs:
            * A,B are (B x 3 x N) or (3 x N)
            * backend: one of ["numpy", "torch", "auto"] for backend choice
        * Outputs: tensor/array of size (B,)
    """
    return A, B


@set_backend_kwarg
@invoke_torch_or_numpy(gdt_torch, gdt_numpy)
def GDT(A, B, *, mode="TS", cutoffs=[1,2,4,8], weights=None):
    """ Returns GDT score as defined here (highre is better):
        Supports both TS and HA
        http://predictioncenter.org/casp12/doc/help.html
        * Inputs:
            * A,B are (B x 3 x N) (np.array or torch.tensor)
            * cutoffs: defines thresholds for gdt
            * weights: list containing the weights
            * mode: one of ["numpy", "torch", "auto"] for backend
        * Outputs: tensor/array of size (B,)
    """
    # define cutoffs for each type of gdt and weights
    cutoffs = [0.5,1,2,4] if mode in ["HA", "ha"] else [1,2,4,8]
    # calculate GDT
    return A, B, cutoffs, {'weights': weights}

@set_backend_kwarg
@invoke_torch_or_numpy(tmscore_torch, tmscore_numpy)
def TMscore(A, B):
    """ Returns TMscore as defined here (higher is better):
        >0.5 (likely) >0.6 (highly likely) same folding.
        = 0.2. https://en.wikipedia.org/wiki/Template_modeling_score
        Warning! It's not exactly the code in:
        https://zhanglab.ccmb.med.umich.edu/TM-score/TMscore.cpp
        but will suffice for now.
        Inputs:
            * A,B are (B x 3 x N) (np.array or torch.tensor)
            * mode: one of ["numpy", "torch", "auto"] for backend
        Outputs: tensor/array of size (B,)
    """
    return A, B


###################
## Other Helpers ##
###################

class EmpiricalDistribution():
    def __init__(self, data=None, probs=None, centers=None, resolution=0.1):
        """ Inputs:
            * data: float tensor. any number of dims. Optional.
            * probs: float tensor. cummulative probabilities.
            * centers: float tensor. centers for the bins.
            * resolution: float. resolution for the empirical distribution
        """
        if data is not None:
            self.resolution = resolution
            self.n_fitted = torch.numel(data)
            self.device = data.device
            self.fit(data)
        elif probs is not None:
            self.probs = probs
            self.centers = centers
            self.device = probs.device
        else:
            raise NotImplementedError("Either data or probs must be provided")

    def fit(self, data):
        self.limits = torch.arange(
            data.min().item(), data.max().item()+self.resolution, self.resolution,
            device = self.device
        )
        self.centers = self.limits[:-1] + ( self.limits[1:] - self.limits[:-1] ) * 0.5
        buckets = torch.bucketize(data, self.limits)
        self.uniques, self.counts = torch.unique(buckets, return_counts=True)
        self.probs = (self.counts / data.bool().sum()).cumsum(dim=0)

    def sample(self, n=1):
        """ Inputs: n: int. number of points to sample. """
        sample = torch.rand(n, device=self.device)
        idxs = torch.searchsorted(sorted_sequence=self.probs,
                                  input=sample).clamp(0, self.centers.shape[0]-1)
        return self.centers[idxs]

    def __repr__(self):
        return "Empirical distribution fitted with {0} data points".format(self.n_fitted)
