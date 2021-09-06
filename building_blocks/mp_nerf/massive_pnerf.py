import time
import numpy as np
# diff ml
import torch
import torch.nn.functional as F
from einops import repeat


def cast_type_to(x:torch.Tensor, target:torch.dtype):
    """ Casts tensor type to target. """
    return x.to(target) if x.dtype != target else x


# @torch.jit.script
# def cumrot(rotations:torch.Tensor):
#     """ Computes a cummulative rotations from a set of ordered rot mats.
#         Does the op in float64 for numerical precision.
#         Inputs, Outputs: (B, 3, 3)
#     """
#     length, dtype = rotations.shape[0], rotations.dtype
#     rotations = cast_type_to(rotations, torch.double)
#     new_rots = [rotations[0]]
#     for i in range(1, length):
#         new_rots.append( torch.matmul(rotations[i], new_rots[i-1]) )
#     return cast_type_to(torch.stack(new_rots, dim=0), dtype)


def get_axis_matrix(a, b, c, norm=True):
    """ Gets an orthonomal basis as a matrix of [e1, e2, e3].
        Useful for constructing rotation matrices between planes
        according to the first answer here:
        https://math.stackexchange.com/questions/1876615/rotation-matrix-from-plane-a-to-b
        Inputs:
        * a: (batch, 3) or (3, ). point(s) of the plane
        * b: (batch, 3) or (3, ). point(s) of the plane
        * c: (batch, 3) or (3, ). point(s) of the plane
        Outputs: orthonomal basis as a matrix of [e1, e2, e3]. calculated as:
            * e1_ = (c-b)
            * e2_proto = (b-a)
            * e3_ = e1_ ^ e2_proto
            * e2_ = e3_ ^ e1_
            * basis = normalize_by_vectors( [e1_, e2_, e3_] )
        Note: Could be done more by Grahm-Schmidt and extend to N-dimensions
              but this is faster and more intuitive for 3D.
    """
    v1_ = c - b
    v2_ = b - a
    v3_ = torch.cross(v1_, v2_, dim=-1)
    v2_ready = torch.cross(v3_, v1_, dim=-1)
    basis    = torch.stack([v1_, v2_ready, v3_], dim=-2)
    # normalize if needed
    if norm:
        return F.normalize(basis, dim=-1)
    return basis



def mp_nerf_torch(a, b, c, l, theta, chi):
    """ Custom Natural extension of Reference Frame.
        Inputs:
        * a: (batch, 3) or (3,). point(s) of the plane, not connected to d
        * b: (batch, 3) or (3,). point(s) of the plane, not connected to d
        * c: (batch, 3) or (3,). point(s) of the plane, connected to d
        * theta: (batch,) or (float).  angle(s) between b-c-d
        * chi: (batch,) or float. dihedral angle(s) between the a-b-c and b-c-d planes
        Outputs: d (batch, 3) or (float). the next point in the sequence, linked to c
    """
    # safety check
    if not ( (-np.pi <= theta) * (theta <= np.pi) ).all().item():
        raise ValueError(f"theta(s) must be in radians and in [-pi, pi]. theta(s) = {theta}")
    # calc vecs
    ba = b-a
    cb = c-b
    # calc rotation matrix. based on plane normals and normalized
    n_plane  = torch.cross(ba, cb, dim=-1)
    n_plane_ = torch.cross(n_plane, cb, dim=-1)
    rotate   = torch.stack([cb, n_plane_, n_plane], dim=-1)
    rotate  = rotate / torch.norm(rotate, dim=-2, keepdim=True)
    # calc proto point, rotate. add (-1 for sidechainnet convention)
    # https://github.com/jonathanking/sidechainnet/issues/14
    d = torch.stack([-torch.cos(theta),
                     torch.sin(theta) * torch.cos(chi),
                     torch.sin(theta) * torch.sin(chi)], dim=-1).unsqueeze(-1)
    # extend base point, set length
    return c + l.unsqueeze(-1) * torch.matmul(rotate, d).squeeze()
