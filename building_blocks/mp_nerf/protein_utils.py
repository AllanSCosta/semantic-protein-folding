# Author: Eric Alcaide

from functools import partial

# science
import numpy as np
# diff / ml
import torch
from einops import repeat
# module
from .mp_nerf_utils import *
from .kb_proteins import *


# random utils

def infer_scn_torsions(new_torsions, seq=None, torsion_mask=None):
    """ Completes a torsion mask with inference of sidechains.
        Inputs:
        * new_torsions: (L, 14) float tensor. New torsions for bb / scn
        * seq: str. FASTA sequence.
        * torsion_mask: (L, 14) float tensor. optional (saves time)
                                Default torsions from kb_proteins

        Outputs: (L, 14)
    """
    assert torsion_mask is not None or seq is not None, \
           "Either torsion mask or seq must be passed"

    if torsion_mask is None: # torsion_mask # (L, 14)
        torsion_mask = torch.tensor([SUPREME_INFO[aa]["torsion_mask"] for aa in seq]).to(new_torsions.device)

    # infer torsions dependent on the ones changed - (see kb_proteins.py)
    for i,aa in enumerate(seq):
        # E and Q have torsion 8 dependent on torsion 7
        if aa == "Q" or aa == "E":
            new_torsions[i, 8] = new_torsions[i, 7] - np.pi
        # E and Q have torsion 8 dependent on torsion 7
        elif aa == "N" or aa == "D":
            new_torsions[i, 7] = new_torsions[i, 6] - np.pi

        # special rigid bodies anomalies:
        elif aa == "I": # scn_torsion(CG1) - scn_torsion(CG2) = 2.13 # see KB
            new_torsions[i, 7] = torsion_mask[i, 7] + new_torsions[i, 5]
        elif aa == "L": # see KB
            new_torsions[i, 7] = torsion_mask[i, 7] + new_torsions[i, 6]
        elif aa == "T": # see KB
            new_torsions[i, 6] = torsion_mask[i, 6] + new_torsions[i, 5]
        elif aa == "V": # see KB
            new_torsions[i, 6] = torsion_mask[i, 6] + new_torsions[i, 5]

    return new_torsions


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




###############
#### MASKS ####
###############

def scn_cloud_mask(seq, coords=None, strict=False):
    """ Gets the boolean mask atom positions (not all aas have same atoms).
        Inputs:
        * seqs: (length) iterable of 1-letter aa codes of a protein
        * coords: optional .(batch, lc, 3). sidechainnet coords.
                  returns the true mask (solves potential atoms that might not be provided)
        * strict: bool. whther to discard the next points after a missing one
        Outputs: (length, 14) boolean mask
    """
    if coords is not None:
        start = (( rearrange(coords, 'b (l c) d -> b l c d', c=14) != 0 ).sum(dim=-1) != 0).float()
        # if a point is 0, the following are 0s as well
        if strict:
            for b in range(start.shape[0]):
                for pos in range(start.shape[1]):
                    for chain in range(start.shape[2]):
                        if start[b, pos, chain].item() == 0:
                            start[b, pos, chain:] *= 0
        return start.bool()
    return torch.tensor([SUPREME_INFO[aa]['cloud_mask'] for aa in seq], dtype=torch.bool)


def scn_bond_mask(seq):
    """ Inputs:
        * seqs: (length). iterable of 1-letter aa codes of a protein
        Outputs: (L, 14) maps point to bond length
    """
    return torch.tensor([SUPREME_INFO[aa]['bond_mask'] for aa in seq])


def scn_angle_mask(seq, angles=None, device=None):
    """ Inputs:
        * seq: (length). iterable of 1-letter aa codes of a protein
        * angles: (length, 12).
                  [ phi, psi, omega,
                    b_angle(n_ca_c), b_angle(ca_c_n), b_angle(c_n_ca),
                    6_scn_torsions ]
        Outputs: (L, 14) maps point to theta and dihedral.
                 first angle is theta, second is dihedral
    """
    device = angles.device if angles is not None else torch.device("cpu")
    precise = angles.dtype if angles is not None else torch.get_default_dtype()
    torsion_mask_use = "torsion_mask" if angles is not None else "torsion_mask_filled"
    # get masks
    theta_mask   = torch.tensor([SUPREME_INFO[aa]['theta_mask'] for aa in seq], dtype=precise).to(device)
    torsion_mask = torch.tensor([SUPREME_INFO[aa][torsion_mask_use] for aa in seq], dtype=precise).to(device)

    # adapt general to specific angles if passed
    if angles is not None:
        # fill masks with angle values
        theta_mask[:, 0] = angles[:, 4] # ca_c_n
        theta_mask[1:, 1] = angles[:-1, 5] # c_n_ca
        theta_mask[:, 2] = angles[:, 3] # n_ca_c
        # backbone_torsions
        torsion_mask[:, 0] = angles[:, 1] # n determined by psi of previous
        torsion_mask[1:, 1] = angles[:-1, 2] # ca determined by omega of previous
        torsion_mask[:, 2] = angles[:, 0] # c determined by phi
        # https://github.com/jonathanking/sidechainnet/blob/master/sidechainnet/structure/StructureBuilder.py#L313
        torsion_mask[:, 3] = angles[:, 1] - np.pi

        # add torsions to sidechains - no need to modify indexes due to torsion modification
        # since extra rigid modies are in terminal positions in sidechain
        to_fill = torsion_mask != torsion_mask # "p" fill with passed values
        to_pick = torsion_mask == 999          # "i" infer from previous one

        # fill scn required torsions - "p" in kb_proteins
        torsion_mask[:, 4:9][to_fill[:, 4:9]] = angles[:, 6:-1][to_fill[:, 4:9]]
        # repeat is no issue since updated are terminal scns, not the already mod
        torsion_mask = infer_scn_torsions(torsion_mask, seq=seq, torsion_mask=torsion_mask)

    torsion_mask[-1, 3] += np.pi
    return torch.stack([theta_mask, torsion_mask], dim=0)


def scn_index_mask(seq):
    """ Inputs:
        * seq: (length). iterable of 1-letter aa codes of a protein
        Outputs: (L, 11, 3) maps point to theta and dihedral.
                 first angle is theta, second is dihedral
    """
    idxs = torch.tensor([SUPREME_INFO[aa]['idx_mask'] for aa in seq])
    return rearrange(idxs, 'l s d -> d l s')


def scn_rigid_index_mask(seq, partial=None):
    """ Inputs:
        * seq: (length). iterable of 1-letter aa codes of a protein
        * partial: str or None. one of ["c_alpha", "backbone", "backbone_cb"]
                   part of the chain to compute frames on.
        Outputs: (3, Length * Groups). indexes for 1st, 2nd and 3rd point
                  to construct frames for each group.
    """
    maxi = 3 if (partial=="backbone" or partial=="c_alpha") else \
           4 if partial=="backbone_cb" else None # backbone is 3 frames, bb+cb is 4 frames

    return torch.cat([torch.tensor(SUPREME_INFO[aa]['rigid_idx_mask'])[2:maxi] if i==0 else \
                      torch.tensor(SUPREME_INFO[aa]['rigid_idx_mask'])[0:maxi] + 14*i \
                      for i,aa in enumerate(seq)], dim=0).t()


def build_scaffolds_from_scn_angles(seq, angles=None, coords=None, device="auto"):
    """ Builds scaffolds for fast access to data
        Inputs:
        * seq: string of aas (1 letter code)
        * angles: (L, 12) tensor containing the internal angles.
                  Distributed as follows (following sidechainnet convention):
                  * (L, 3) for torsion angles
                  * (L, 3) bond angles
                  * (L, 6) sidechain angles
        * coords: (L, 3) sidechainnet coords. builds the mask with those instead
                  (better accuracy if modified residues present).
        Outputs:
        * cloud_mask: (L, 14 ) mask of points that should be converted to coords
        * point_ref_mask: (3, L, 11) maps point (except n-ca-c) to idxs of
                                     previous 3 points in the coords array
        * angles_mask: (2, L, 14) maps point to theta and dihedral
        * bond_mask: (L, 14) gives the length of the bond originating that atom
    """
    # auto infer device and precision
    precise = angles.dtype if angles is not None else torch.get_default_dtype()
    if device == "auto":
        device = angles.device if angles is not None else "cpu"

    if coords is not None:
        cloud_mask = scn_cloud_mask(seq, coords=coords)
    else:
        cloud_mask = scn_cloud_mask(seq)

    cloud_mask = cloud_mask.bool().to(device)

    point_ref_mask = scn_index_mask(seq).long().to(device)

    angles_mask = scn_angle_mask(seq, angles).to(device, precise)

    bond_mask = scn_bond_mask(seq).to(device, precise)
    # return all in a dict
    return {"cloud_mask":     cloud_mask,
            "point_ref_mask": point_ref_mask,
            "angles_mask":    angles_mask,
            "bond_mask":      bond_mask }


def scaffolds_from_pdb(filename=None, chain="A", data=None):
    if data is None:
        data = read_pdb(filename, chain=chain)
    return data["observed_sequence"], \
           build_scaffolds_from_scn_angles(
            seq=data["observed_sequence"], angles=torch.from_numpy(data["angles_np"]) )


def atom_selector(x, scn_seq=None, masks=None, option=None, discard_absent=True):
    """ Returns a selection of the atoms in a protein.
        Inputs:
        * scn_seq: (batch, len) sidechainnet format or list of strings
        * x: (batch, (len * n_aa), dims) sidechainnet format
        * masks: (batch, len, n_aa) sidechainnet format
        * option: one of [torch.tensor, 'backbone-only', 'backbone-with-cbeta',
                  'all', 'backbone-with-oxygen', 'backbone-with-cbeta-and-oxygen']
        * discard_absent: bool. Whether to discard the points for which
                          there are no labels (bad recordings)
    """
    # get mask
    if masks is None:
        present = []
        for i,seq in enumerate(scn_seq):
            pass_x = x[i] if discard_absent else None
            if pass_x is None and isinstance(seq, torch.Tensor):
                seq = "".join([INDEX2AAS[x] for x in seq.cpu().detach().tolist()])

            present.append( scn_cloud_mask(seq, coords=pass_x) )

        present = torch.stack(present, dim=0).bool()
    else:
        present = masks.bool()

    # atom mask
    if isinstance(option, str):
        atom_mask = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if "backbone" in option:
            atom_mask[[0, 2]] = 1

        if option == "backbone":
            pass
        elif option == 'backbone-with-oxygen':
            atom_mask[3] = 1
        elif option == 'backbone-with-cbeta':
            atom_mask[5] = 1
        elif option == 'backbone-with-cbeta-and-oxygen':
            atom_mask[3] = 1
            atom_mask[5] = 1
        elif option == 'all':
            atom_mask[:] = 1
        elif option == "ca":
            pass
        else:
            print("Your string doesn't match any option.")

    elif isinstance(option, torch.Tensor):
        atom_mask = option
    else:
        raise ValueError('option needs to be a valid string or a mask tensor of shape (14,) ')

    mask = rearrange(present * atom_mask[None, None, ...], 'b l c -> b (l c)').bool()
    return x[mask], mask.bool()


#############################
####### ENCODERS ############
#############################


def modify_angles_mask_with_torsions(seq, angles_mask, torsions, torsion_mask=None):
    """ Modifies a torsion mask to include variable torsions.
        Re-infers the the rest
        Inputs:
        * seq: (L,) str. FASTA sequence
        * angles_mask: (2, L, 14) float tensor of (angles, torsions)
        * torsions: (L, 4) float tensor (or (L, 5) if it includes torsion for cb)
        * torsion_mask: (L, 14) torsion mask for the whole chain.
        Outputs: (2, L, 14) a new angles mask
    """
    c_beta = torsions.shape[-1] == 5 # whether c_beta torsion is passed as well
    start = 4 if c_beta else 5
    # get mask of to-fill values
    torsion_mask = torch.tensor([SUPREME_INFO[aa]["torsion_mask"] for aa in seq]).to(torsions.device) # (L, 14)
    to_fill = torsion_mask != torsion_mask # "p" fill with passed values

    # undesired outside of margins - only 4-5 torsions are useful
    to_fill[:, :start] = to_fill[:, start+torsions.shape[-1]:] = False
    # replace old by new torsions
    angles_mask[1, to_fill] = torsions[ to_fill[:, start:start+torsions.shape[-1]] ]
    # complete scn torsions with inferred ones
    angles_mask[1] = infer_scn_torsions(angles_mask[1], seq, torsion_mask=torsion_mask)

    return angles_mask


def modify_scaffolds_with_coords(scaffolds, coords):
    """ Gets scaffolds and fills in the right data.
        Inputs:
        * scaffolds: dict. as returned by `build_scaffolds_from_scn_angles`
        * coords: (L, 14, 3). sidechainnet tensor. same device as scaffolds
        Outputs: corrected scaffolds
    """


    # calculate distances and update:
    # N, CA, C
    scaffolds["bond_mask"][1:, 0] = torch.norm(coords[1:, 0] - coords[:-1, 2], dim=-1) # N
    scaffolds["bond_mask"][ :, 1] = torch.norm(coords[ :, 1] - coords[:  , 0], dim=-1) # CA
    scaffolds["bond_mask"][ :, 2] = torch.norm(coords[ :, 2] - coords[:  , 1], dim=-1) # C
    # O, CB, side chain
    selector = np.arange(len(coords))
    for i in range(3, 14):
        # get indexes
        idx_a, idx_b, idx_c = scaffolds["point_ref_mask"][:, :, i-3] # (3, L, 11) -> 3 * (L, 11)
        # correct distances
        scaffolds["bond_mask"][:, i] = torch.norm(coords[:, i] - coords[selector, idx_c], dim=-1)
        # get angles
        scaffolds["angles_mask"][0, :, i] = get_angle(coords[selector, idx_b],
                                                      coords[selector, idx_c],
                                                      coords[:, i])
        # handle C-beta, where the C requested is from the previous aa
        if i == 4:
            # for 1st residue, use position of the second residue's N
            first_next_n     = coords[1, :1] # 1, 3
            # the c requested is from the previous residue
            main_c_prev_idxs = coords[selector[:-1], idx_a[1:]]# (L-1), 3
            # concat
            coords_a = torch.cat([first_next_n, main_c_prev_idxs])
        else:
            coords_a = coords[selector, idx_a]
        # get dihedrals
        scaffolds["angles_mask"][1, :, i] = get_dihedral(coords_a,
                                                         coords[selector, idx_b],
                                                         coords[selector, idx_c],
                                                         coords[:, i])
    # correct angles and dihedrals for backbone
    scaffolds["angles_mask"][0, :-1, 0] = get_angle(coords[:-1, 1], coords[:-1, 2], coords[1: , 0]) # ca_c_n
    scaffolds["angles_mask"][0, 1:,  1] = get_angle(coords[:-1, 2], coords[1:,  0], coords[1: , 1]) # c_n_ca
    scaffolds["angles_mask"][0,  :,  2] = get_angle(coords[:,   0], coords[ :,  1], coords[ : , 2]) # n_ca_c

    # N determined by previous psi = f(n, ca, c, n+1)
    scaffolds["angles_mask"][1, :-1, 0] = get_dihedral(coords[:-1, 0], coords[:-1, 1], coords[:-1, 2], coords[1:, 0])
    # CA determined by omega = f(ca, c, n+1, ca+1)
    scaffolds["angles_mask"][1,  1:, 1] = get_dihedral(coords[:-1, 1], coords[:-1, 2], coords[1:, 0], coords[1:, 1])
    # C determined by phi = f(c-1, n, ca, c)
    scaffolds["angles_mask"][1,  1:, 2] = get_dihedral(coords[:-1, 2], coords[1:, 0], coords[1:, 1], coords[1:, 2])

    return scaffolds


############################
####### METRICS ############
############################


def get_protein_metrics(
        true_coords,
        pred_coords,
        cloud_mask = None,
        return_aligned = True,
        detach = None
    ):

    """ Calculates many metrics for protein structure quality.
        Aligns coordinates.
        Inputs:
        * true_coords: (B, L, 14, 3) unaligned coords (B = 1)
        * pred_coords: (B, L, 14, 3) unaligned coords (B = 1)
        * cloud_mask: (B, L, 14) bool. gotten from pred_coords if not passed
        * return_aligned: bool. whether to return aligned structs.
        * detach: bool. whether to detach inputs before compute. saves mem
        Outputs: dict (k,v)
    """

    metric_dict = {
        "rmsd": RMSD,
        "drmsd": dRMSD,
        "gdt_ts": partial(GDT, mode="TS"),
        "gdt_ha": partial(GDT, mode="HA"),
        "tmscore": TMscore,
        "lddt": lddt_torch,
    }

    if detach:
        true_coords = true_coords.detach()
        pred_coords = pred_coords.detach()

    cloud_mask = pred_coords.abs().sum(dim=-1).bool()
    true_aligned, pred_aligned = Kabsch(
        true_coords[cloud_mask].t(), pred_coords[cloud_mask].t()
    )

    # no need to rebuild true coords since unaffected by kabsch
    true_align_wrap = true_coords.clone()
    pred_align_wrap = torch.zeros_like(pred_coords)
    pred_align_wrap[cloud_mask] = true_aligned.t()

    # compute metrics
    outputs = {}
    for k,f in metric_dict.items():
        # special. works only on ca trace
        if k == "tmscore":
            ca_trace = true_align_wrap[..., 1].transpose(-1, -2)
            ca_pred_trace = pred_align_wrap[..., 1].transpose(-1, -2)
            outputs[k] = f(ca_trace, ca_pred_trace)
        # special. works on full prot
        elif k == "lddt":
            outputs[k] = f(true_align_wrap, pred_align_wrap, cloud_mask=cloud_mask)
        # special. needs batch dim
        elif "gdt" in k:
            outputs[k] = f(true_aligned[None, ...], pred_aligned[None, ...])
        else:
            outputs[k] = f(true_aligned, pred_aligned)

    if return_aligned:
        return outputs, (true_align_wrap, pred_align_wrap)

    return outputs
