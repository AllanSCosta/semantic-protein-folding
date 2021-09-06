import torch
from torch import nn
from operator import itemgetter
from .reversible import ReversibleSequence

# helper functions

def exists(val):
    return val is not None

def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))

def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)

# calculates the permutation to bring the input tensor to something attend-able
# also calculates the inverse permutation to bring the tensor back to its original shape

def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)

    return permutations

# helper classes

class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return self.fn(x) * self.g

class Sequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g in self.blocks:
            x = x + f(x) + g(x)
        return x

class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()

        shape = axial.shape
        *_, t, d = shape

        # merge all but axial dimension
        axial = axial.reshape(-1, t, d)

        # attention
        axial = self.fn(axial, **kwargs)

        # restore to original shape and permutation
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial

# axial pos emb

class AxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape, emb_dim_index = 1):
        super().__init__()
        parameters = []
        total_dimensions = len(shape) + 2
        ax_dim_indexes = [i for i in range(1, total_dimensions) if i != emb_dim_index]

        for axial_dim, axial_dim_index in zip(shape, ax_dim_indexes):
            shape = [1] * total_dimensions
            shape[emb_dim_index] = dim
            shape[axial_dim_index] = axial_dim
            parameter = nn.Parameter(torch.randn(*shape))
            parameters.append(parameter)

        self.params = nn.ParameterList(parameters)

    def forward(self, x):
        for param in self.params:
            x = x + param
        return x

# attention

class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads = None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias = False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv = None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out

# axial attention class

class AxialAttention(nn.Module):
    def __init__(self, dim, num_dimensions = 2, heads = 8, dim_heads = None, dim_index = -1, sum_axial_out = True):
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)

        attentions = []
        for permutation in calculate_permutations(num_dimensions, dim_index):
            attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads)))

        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.dim, 'input tensor does not have the correct input dimension'

        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))

        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)
        return out

# axial image transformer
import sys
sys.path.append('/home/gridsan/allanc/msa-transformer-folding')

from building_blocks.long_short_transformer import LongShortAttention
import torch.utils.checkpoint as checkpoint

# class AxialLongShortTransformer(nn.Module):
#     def __init__(self, dim, depth, heads = 8, dim_heads = None, dim_index = 1, checkpoint = True, window_size=128, axial_pos_emb_shape = None):
#         super().__init__()
#         permutations = calculate_permutations(2, dim_index)
#
#         get_ff = lambda: nn.Sequential(
#             nn.Conv2d(dim, dim, 3, padding=1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(dim, dim, 3, padding=1)
#         )
#
#         self.checkpoint = checkpoint
#         self.pos_emb = AxialPositionalEmbedding(dim, axial_pos_emb_shape, dim_index) if exists(axial_pos_emb_shape) else nn.Identity()
#
#         layers = nn.ModuleList([])
#         for _ in range(depth):
#             attn_functions = nn.ModuleList([PermuteToFrom(permutation, Rezero(LongShortAttention(dim=dim, heads=heads, dim_head=dim_heads, window_size=window_size))) for permutation in permutations])
#             conv_functions = nn.ModuleList([Rezero(get_ff())])
#             layers.append(attn_functions)
#             layers.append(conv_functions)
#
#         self.layers = layers
#
#
#     def forward(self, x):
#         x = self.pos_emb(x)
#         for attn_fn, conv_fn in self.layers:
#             row_attn, col_attn = attn_fn
#             x = (checkpoint.checkpoint(self.axial_checkpoint(*attn_fn, conv_fn)) if self.checkpoint else conv_fn(col_attn(row_attn(x))))
#         return x




class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)



class AxialLongShortTransformer(nn.Module):
    def __init__(self, dim, depth, heads = 8, dim_heads = None, dim_index = 1, checkpoint = True, window_size=128, axial_pos_emb_shape = None):
        super().__init__()
        permutations = calculate_permutations(2, dim_index)

        get_ff = lambda: nn.Sequential(
            ChanLayerNorm(dim),
            nn.Conv2d(dim, dim * 4, 3, padding = 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim * 4, dim, 3, padding = 1)
        )

        self.pos_emb = AxialPositionalEmbedding(dim, axial_pos_emb_shape, dim_index) if exists(axial_pos_emb_shape) else nn.Identity()
        self.checkpoint = checkpoint

        layers = nn.ModuleList([])
        for _ in range(depth):
            block = nn.ModuleList([
                *[
                    PermuteToFrom(
                        permutation,
                        PreNorm(
                            dim,
                            LongShortAttention(
                                dim=dim,
                                heads=heads,
                                dim_head=dim_heads,
                                window_size=window_size
                            )
                        )
                    ) for permutation in permutations
                ],
                get_ff()
            ])
            layers.append(block)
        self.layers = layers

    def axial_checkpoint(self, block):
        def custom_forward(x):
            row, col, conv = block
            x = row(x)
            x = col(x)
            x = conv(x)
            return x
        return custom_forward

    def forward(self, x):
        for block in self.layers:
            if self.checkpoint:
                x = checkpoint.checkpoint(self.axial_checkpoint(block), x)
            else:
                row_attn, col_attn, conv = block
                x = row_attn(x)
                x = col_attn(x)
                x = conv(x)
        return x
