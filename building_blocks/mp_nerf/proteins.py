# science
import numpy as np
# diff / ml
import torch
import torch.nn.functional as F
from einops import repeat
# module
from .massive_pnerf import *
from .mp_nerf_utils import *
from .kb_proteins import *
from .protein_utils import *


def protein_fold(cloud_mask, point_ref_mask, angles_mask, bond_mask, hybrid=False):
    """ Calcs coords of a protein given it's
        sequence and internal angles.
        Inputs:
        * cloud_mask: (L, 14) mask of points that should be converted to coords
        * point_ref_mask: (3, L, 11) maps point (except n-ca-c) to idxs of
                                     previous 3 points in the coords array
        * angles_mask: (2, L, 14) maps point to theta and dihedral
        * bond_mask: (L, 14) gives the length of the bond originating that atom

        Output: (L, 14, 3) and (L, 14) coordinates and cloud_mask
    """
    # automatic type (float, mixed, double) and size detection
    device, precise = bond_mask.device, bond_mask.dtype
    length  = cloud_mask.shape[0]
    # create coord wrapper
    coords = torch.zeros(length, 14, 3, device=device, dtype=precise)

    # do first AA
    coords[0, 1] = coords[0, 0] + torch.tensor([1, 0, 0], device=device, dtype=precise) * BB_BUILD_INFO["BONDLENS"]["n-ca"]
    coords[0, 2] = coords[0, 1] + torch.stack([torch.cos(np.pi - angles_mask[0, 0, 2]),
                                               torch.sin(np.pi - angles_mask[0, 0, 2]),
                                               torch.tensor(0., device=device, dtype=precise)], dim=-1) * BB_BUILD_INFO["BONDLENS"]["ca-c"]

    # starting positions (in the x,y plane) and normal vector [0,0,1]
    init_a = repeat(torch.tensor([1., 0., 0.], device=device, dtype=precise), 'd -> l d', l=length)
    init_b = repeat(torch.tensor([1., 1., 0.], device=device, dtype=precise), 'd -> l d', l=length)
    # do N -> CA. don't do 1st since its done already
    thetas, dihedrals = angles_mask[:, :, 1]
    coords[1:, 1] = mp_nerf_torch(init_a,
                                   init_b,
                                   coords[:, 0],
                                   bond_mask[:, 1],
                                   thetas, dihedrals)[1:]
    # do CA -> C. don't do 1st since its done already
    thetas, dihedrals = angles_mask[:, :, 2]
    coords[1:, 2] = mp_nerf_torch(init_b,
                                   coords[:, 0],
                                   coords[:, 1],
                                   bond_mask[:, 2],
                                   thetas, dihedrals)[1:]
    # do C -> N
    thetas, dihedrals = angles_mask[:, :, 0]
    coords[:, 3] = mp_nerf_torch(coords[:, 0],
                                   coords[:, 1],
                                   coords[:, 2],
                                   bond_mask[:, 0],
                                   thetas, dihedrals)

    #########
    # sequential pass to join fragments
    #########
    # part of rotation mat corresponding to origin - 3 orthogonals
    mat_origin  = get_axis_matrix(init_a[0], init_b[0], coords[0, 0], norm=True)
    # part of rotation mat corresponding to destins || a, b, c = CA, C, N+1
    # (L-1) since the first is in the origin already
    mat_destins = get_axis_matrix(coords[:-1, 1], coords[:-1, 2], coords[:-1, 3], norm=True)
    # get rotation matrices from origins
    # https://math.stackexchange.com/questions/1876615/rotation-matrix-from-plane-a-to-b
    rotations  = torch.matmul(mat_origin.t(), mat_destins)

    # do rotation concatenation - do for loop in cpu always - faster. Avoid TF32
    switch_tf32(False)
    rotations = to_cpu(rotations)
    rotations = cumrot(rotations)
    rotations = to_device(rotations, device) if hybrid else rotations
    switch_tf32(True)
    # rotate all
    coords[1:, :4] = torch.matmul(coords[1:, :4], rotations)
    # offset each position by cumulative sum at that position
    coords[1:, :4] = coords[1:, :4] + torch.cumsum(coords[:-1, 3:4], dim=0)


    #########
    # parallel sidechain - do the oxygen, c-beta and side chain
    #########
    for i in range(3,14):
        level_mask = cloud_mask[:, i]
        thetas, dihedrals = angles_mask[:, level_mask, i]
        idx_a, idx_b, idx_c = point_ref_mask[:, level_mask, i-3]

        # to place C-beta, we need the carbons from prev res - not available for the 1st res
        if i == 4:
            # the c requested is from the previous residue - offset boolean mask by one
            # can't be done with slicing bc glycines are inside chain (dont have cb)
            coords_a = coords[(level_mask.nonzero().view(-1) - 1), idx_a] # (L-1), 3
            # if first residue is not glycine,
            # for 1st residue, use position of the second residue's N (1,3)
            if level_mask[0]:
                coords_a[0] = coords[1, 1]
        else:
            coords_a = coords[level_mask, idx_a]

        coords[level_mask, i] = mp_nerf_torch(coords_a,
                                              coords[level_mask, idx_b],
                                              coords[level_mask, idx_c],
                                              bond_mask[level_mask, i],
                                              thetas, dihedrals)

    return coords, cloud_mask


# inspired by: https://www.biorxiv.org/content/10.1101/2021.08.02.454840v1
def ca_from_angles(angles, bond_len=3.80):
    """ Builds a C-alpha trace from a set of 2 angles (theta, chi).
        Inputs:
        * angles: (B, L, 4): float tensor. (cos, sin) · (theta, chi)
                  angles in point-in-unit-circumference format.
        Outputs: (B, L, 3) coords for c-alpha trace
    """
    device = angles.device
    length = angles.shape[-2]
    frames = [ torch.repeat_interleave(
                torch.eye(3, device=device, dtype=torch.double).unsqueeze(0),
                angles.shape[0],
                dim=0
             )]

    rot_mats = torch.stack([
        torch.stack([  angles[...,0] * angles[...,2], angles[...,0] * angles[...,3], -angles[...,1]    ], dim=-1),
        torch.stack([ -angles[...,3]                , angles[...,2]                ,  angles[...,0]*0. ], dim=-1),
        torch.stack([  angles[...,1] * angles[...,2], angles[...,1] * angles[...,3],  angles[...,0]    ], dim=-1),
    ], dim=-2)  # (B, L, 3, 3)

    # to-do: see how this relates to `mp_nerf.massive_pnerf.cumrot`
    # SENSITIVE FOR ACC: do cumulative rotation in DOUBLE precision
    rot_mats = cast_type_to(rot_mats, torch.double)
    switch_tf32(False)

    # iterative update of frames
    for i in range(length-1):
        frames.append( rot_mats[:, i] @ frames[i] ) # could do frames[-1] as well
    frames = torch.stack(frames, dim=1) # (B, L, 3, 3)

    ca_trace = bond_len * frames[..., -1, :].cumsum(dim=-2) # (B, L, 3)

    # convert back to existing types
    frames = cast_type_to(frames, angles.dtype)
    ca_trace = cast_type_to(ca_trace, angles.dtype)
    switch_tf32(True)

    return ca_trace, frames


# inspired by: https://github.com/psipred/DMPfold2/blob/master/dmpfold/network.py#L139
def ca_bb_fold(ca_trace):
    """ Calcs a backbone given the coordinate trace of the CAs.
        Inputs:
        * ca_trace: (B, L, 3) float tensor with CA coordinates.
        Outputs: (B, L, 14, 3) (-N-CA(-CB-...)-C(=O)-)
    """
    wrapper = torch.zeros(ca_trace.shape[0], ca_trace.shape[1]+2, 14, 3, device=ca_trace.device)
    wrapper[:, 1:-1, 1] = ca_trace
    # Place dummy extra Cα atoms on extremenes to get the required vectors
    vecs = ca_trace[ :, [0, 2, -1, -3] ] - ca_trace[ :, [1, 1, -2, -2] ] # (B, 4, 3)
    wrapper[:,  0, 1] = ca_trace[:,  0] + 3.80 * F.normalize(torch.cross(vecs[:, 0], vecs[:, 1]), dim=-1)
    wrapper[:, -1, 1] = ca_trace[:, -1] + 3.80 * F.normalize(torch.cross(vecs[:, 2], vecs[:, 3]), dim=-1)

    # place N and C term
    vec_ca_can = wrapper[:, :-2, 1] - wrapper[:, 1:-1, 1]
    vec_ca_cac = wrapper[:, 2: , 1] - wrapper[:, 1:-1, 1]
    mid_ca_can = (wrapper[:, 1:, 1] + wrapper[:, :-1, 1]) / 2
    cross_vcan_vcac = F.normalize(torch.cross(vec_ca_can, vec_ca_cac, dim=-1), dim=-1)
    wrapper[:, 1:-1, 0] = mid_ca_can[:, :-1] - vec_ca_can / 7.5 + cross_vcan_vcac / 3.33
    # placve all C but last, which is special
    wrapper[:, 1:-2, 2] = (mid_ca_can[:, :-1] + vec_ca_can / 8 - cross_vcan_vcac / 2.5)[:, 1:]
    wrapper[:,   -2, 2] = mid_ca_can[:, -1, :] - vec_ca_cac[:, -1, :] / 8 + cross_vcan_vcac[:, -1, :] / 2.5

    return wrapper[:, 1:-1]


# inspired by: https://github.com/psipred/DMPfold2/blob/master/dmpfold/network.py#L139
def cbeta_fold(wrapper, cloud_mask=None):
    """ Places the Cbeta for a set of backbone coordinates.
        Inputs:
        * wrapper: ( (B), L, C, 3) float tensor. bb coords in sidechainnet format
        * cloud_mask: ((B), L, C) bool tensor.
        Outputs: ( (B), L, C, 3) (-N-CA(-CB-...)-C(=O)-)
    """
    vec_n_ca = wrapper[..., 1, :] - wrapper[..., 0, :]
    vec_c_ca = wrapper[..., 1, :] - wrapper[..., 2, :]
    cross_vn_vc = torch.cross(vec_n_ca, vec_c_ca, dim=-1)
    vec_ca_cb = vec_n_ca + vec_c_ca
    ang = np.pi / 2 - np.arcsin(1 / np.sqrt(3))
    # differs 1.6 vs 1.5 - optimized by gridsearch
    sx = 1.6 * np.cos(ang) / vec_ca_cb.norm(dim=-1, keepdim=True)
    sy = 1.5 * np.sin(ang) / cross_vn_vc.norm(dim=-1, keepdim=True)
    wrapper[..., 4, :] = wrapper[..., 1, :] + 1.526 * \
                         F.normalize(sx * vec_ca_cb + sy * cross_vn_vc, dim=-1)
    # mask out glycine cbetas
    wrapper *= cloud_mask.float().unsqueeze(-1)
    return wrapper


def sidechain_fold(wrapper, cloud_mask, point_ref_mask, angles_mask, bond_mask,
                   c_beta="torsion"):
    """ Calcs coords of a protein given it's sequence and internal angles.
        Inputs:
        * wrapper: (L, 14, 3). coords container (ex: with backbone ([:, :3]) and optionally
                               c_beta ([:, 4])
        * cloud_mask: (L, 14) mask of points that should be converted to coords
        * point_ref_mask: (3, L, 11) maps point (except n-ca-c) to idxs of
                                     previous 3 points in the coords array
        * angles_mask: (2, L, 14) maps point to theta and dihedral
        * bond_mask: (L, 14) gives the length of the bond originating that atom
        * c_beta: None or str ("torsion", "backbone"). whether to place cbeta
                  if its not present and which way to add.
                  None: don't place.
                  "torsion": by torsion regression.
                  "backbone": from backbone as template. more accurate.

        Output: (L, 14, 3) and (L, 14) coordinates and cloud_mask
    """
    device, precise = wrapper.device, wrapper.dtype
    own_c_beta = wrapper[..., 4, :].abs().sum().item()

    # parallel sidechain - do the oxygen, c-beta and side chain
    for i in range(3,14):
        # skip cbeta if arg is set
        if i == 4:
            if c_beta == "torsion":
                pass # gen as normal
            elif c_beta == "backbone":
                wrapper[..., 4, :] = cbeta_fold(wrapper, cloud_mask)[..., 4, :]
                continue # place by custom calc
            else:
                continue # don't place

        # prepare inputs
        level_mask = cloud_mask[:, i]
        thetas = angles_mask[0, level_mask, i]
        dihedrals = angles_mask[1, level_mask, i]
        idx_a, idx_b, idx_c = point_ref_mask[:, level_mask, i-3]

        # to place =O, pick dihedral from oposite of psi of current res, as in scn
        if i == 3:
            dihedrals[:-1] = get_dihedral(wrapper[:-1, 0], wrapper[:-1, 1],
                                          wrapper[:-1, 2], wrapper[1:, 0]) - np.pi
            coords_a = wrapper[level_mask, idx_a]
        # to place C-beta, we need the carbons from prev res - not available for the 1st res
        elif i == 4:
            # the c requested is from the previous residue - offset boolean mask by one
            # can't be done with slicing bc glycines are inside chain (dont have cb)
            coords_a = wrapper[(level_mask.nonzero().view(-1) - 1), idx_a] # (L-1), 3
            # if first residue is not glycine,
            # for 1st residue, use position of the second residue's N (1,3)
            if level_mask[0].item():
                coords_a[0] = wrapper[1, 1]
        else:
            coords_a = wrapper[level_mask, idx_a]

        wrapper[level_mask, i] = mp_nerf_torch(coords_a,
                                               wrapper[level_mask, idx_b],
                                               wrapper[level_mask, idx_c],
                                               bond_mask[level_mask, i],
                                               thetas, dihedrals)

    return wrapper, cloud_mask
