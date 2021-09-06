import matplotlib.pyplot as plt
import sidechainnet as scn
import wandb
import py3Dmol
import torch

plt.rcParams.update({
    "figure.facecolor":  (0.0, 0.0, 0.0, 0.0),
    "axes.facecolor":    (0.0, 0.0, 0.0, 0.0),
    "savefig.facecolor": (0.0, 0.0, 0.0, 0.0),
})

from protein_utils import build_scaffolds_from_scn_angles
from proteins import sidechain_fold, ca_bb_fold

def plot_predicted_angles(name, seq, anchor, gnd_angs, pred_angs):
    gnd_bb = ca_bb_fold(anchor.unsqueeze(0))[0]

    gnd_scaffolds = build_scaffolds_from_scn_angles(seq, angles=gnd_angs, device="cpu")
    gnd_coords, _ = sidechain_fold(wrapper = gnd_bb.clone(), **gnd_scaffolds, c_beta = 'torsion')

    pred_scaffolds = build_scaffolds_from_scn_angles(seq, angles=pred_angs, device="cpu")
    pred_coords, _ = sidechain_fold(wrapper = gnd_bb.clone(), **pred_scaffolds, c_beta = 'torsion')

    gnd_struct = scn.StructureBuilder(seq, gnd_coords.reshape(-1, 3))
    pred_struct = scn.StructureBuilder(seq, pred_coords.reshape(-1, 3) + 0.1)

    gnd_pdb = gnd_struct.to_pdbstr(title=f'{name}_sidechain_prediction')
    pred_pdb = pred_struct.to_pdbstr(title=f'{name}_sidechain_prediction')

    view = py3Dmol.view()
    view.setBackgroundColor(0x000000,0)
    view.addModelsAsFrames(gnd_pdb)
    view.setStyle({'model': 0}, {'stick': {'color': 'green', 'radius': .1}})

    view.addModelsAsFrames(pred_pdb)
    view.setStyle({'model': 1}, {'stick': {'color': 'cyan', 'radius': .1}})

    view.zoomTo()
    view.rotate(1, 'y')

    return view

def plot_aligned_backbones(b1, b2, metrics):
    trace = aligned_pair_trace(b1, b2)
    fig = produce_3d_traces_row(trace)
    fig.update_layout(title=' '.join(f'{key}: {value:.4e}' for key, value in metrics.items()))
    return fig


def plot_distogram_predictions(pred_dist, pred_ang, dist, ang):
    fig, axes = plt.subplots(2, 4, figsize=(10, 8))

    for idx, pred in enumerate([pred_dist, *pred_ang]):
        axes[0][idx].imshow(-pred)

    for idx, gnd in enumerate([dist, *ang]):
        axes[1][idx].imshow(-gnd)

    return fig


def visualize_aligned_structures(name, seq, gnd_crd, pred_crd):
    gnd_struct = scn.StructureBuilder(seq, gnd_crd.reshape(-1, 3))
    pred_struct = scn.StructureBuilder(seq, pred_crd.reshape(-1, 3))

    view = py3Dmol.view(viewergrid=(2,2))
    view.setBackgroundColor(0x000000,0)

    gnd_pdb = gnd_struct.to_pdbstr(title=f'{name}_gnd')
    pred_pdb = pred_struct.to_pdbstr(title=f'{name}_pred')

    view.addModelsAsFrames(gnd_pdb, viewer=(0, 0))
    view.setStyle({'model': 0}, {'cartoon': {'color': 'spectrum'}, 'stick': {'radius': .15}}, viewer=(0 ,0))

    view.addModelsAsFrames(pred_pdb, viewer=(0, 1))
    view.setStyle({'model': 0}, {'cartoon': {'color': 'spectrum'}, 'stick': {'radius': .15}}, viewer=(0, 1))

    view.addModelsAsFrames(gnd_pdb, viewer=(1, 0))
    view.setStyle({'model': 0}, {"stick": {'radius': 0.15, 'color': 'green'}}, viewer=(1, 0))

    view.addModelsAsFrames(pred_pdb, viewer=(1, 0))
    view.setStyle({'model': 1}, {"stick": {'radius': 0.15, 'color': 'cyan'}}, viewer=(1, 0))


    view.addModelsAsFrames(gnd_pdb, viewer=(1, 1))
    view.setStyle({'model': 0}, {"sphere": {'hidden': True}}, viewer=(1, 1))
    view.addStyle({'model': 0, 'atom':['CA']}, {'sphere':{'hidden': False, 'color':'green','radius':0.6}}, viewer=(1, 1))

    view.addModelsAsFrames(pred_pdb, viewer=(1, 1))
    view.setStyle({'model': 1}, {"sphere": {'hidden': True}}, viewer=(1, 1))
    view.addStyle({'model': 1, 'atom':['CA']}, {'sphere':{'hidden': False, 'color':'cyan','radius':0.6}}, viewer=(1, 1))

    view.zoomTo()
    view.rotate(1, 'y')
    wandb.log({ f'{name} structures': wandb.Html(view._make_html()) })

    gnd_bb_crd = gnd_crd[:, 1, :]
    pred_bb_crd = pred_crd[:, 1, :]

    gnd_dist_map = torch.cdist(gnd_bb_crd, gnd_bb_crd)
    pred_dist_map = torch.cdist(pred_bb_crd, pred_bb_crd)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
    ax1.imshow(-gnd_dist_map)
    ax2.imshow(-pred_dist_map)
    wandb.log({f'{name} distance map': fig})


import plotly
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splprep, splev
import torch
from plotly.subplots import make_subplots

import gif
gif.options.matplotlib["dpi"] = 300

import plotly.graph_objects as go
from torch.linalg import norm

def aligned_pair_trace(coords1, coords2, coloring1='#1F77B4', coloring2='#E45756', upsample=1000, node_size=2):
    protein1 = protein_backbone_trace(coords1, coloring=coloring1, node_size=node_size, upsample=upsample)
    protein2 = protein_backbone_trace(coords2, coloring=coloring2, node_size=node_size, upsample=upsample)
    return protein1 + protein2

def spline_interpolation(interpolants, interpolation_upsample_size=1000):
    tck, u = splprep(interpolants.T, s=0.0)
    interpolants = splev(np.linspace(0, 1, interpolation_upsample_size),tck)
    interpolants = np.stack(interpolants).T
    return interpolants

def linear_interpolation(interpolants, interpolation_upsample_size=1000):
    upsample = np.linspace(0, len(interpolants)-1, interpolation_upsample_size)
    interp = interp1d(np.arange(0, len(interpolants)), interpolants, axis=0)
    return interp(upsample)

def full_protein_point_cloud_trace(coords, tokens):
    cmap = plt.cm.get_cmap('viridis')
    rbga_ = cmap(tokens / 20)[:, :-1]
    supports = point_cloud_trace(coords, rbga_, node_size=1, coloring=coloring)


def aligned_full_atom_trace(coords1, coords2, tokens):
    trace =  point_cloud_trace(coords1, tokens, node_size=1, coloring='tokens', colorscale='Agsunset', linecolor='rgba(100, 255,255,0)')
    trace += point_cloud_trace(coords2, tokens, node_size=1, coloring='tokens', colorscale='Aggrnyl', linecolor='rgba(100, 255,255,0)')

    x_lines = list()
    y_lines = list()
    z_lines = list()
    colors = []

    #create the coordinate list for the lines
    for coord1, coord2 in zip(coords1, coords2):
        for coord in coord1, coord2:
            x_lines.append(coord[0])
            y_lines.append(coord[1])
            z_lines.append(coord[2])

        colors.extend([norm(coord1 - coord2, ord=2, dim=-1).item()] * 3)

        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)

    trace2 = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode='lines',
        name='lines',
        line=dict(
            color=colors,
            colorscale='Inferno',
            width=0
        )
    )

    return trace + [trace2]


def point_cloud_trace(coords, tokens=None, coloring='spectrum', node_size=5, colorscale='Viridis', linecolor=None):
    if coloring == 'spectrum':
        colormap = np.arange(0, len(coords), 1)
    elif coloring == 'tokens' and tokens is not None:
        colormap = np.array(tokens)
    else:
        colormap = coloring

    x, y, z = coords.T
    node_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        name='atoms',
        marker=dict(
            symbol='circle',
            color=colormap,
            colorscale=colorscale,
            size=node_size,
        ),
        hoverinfo='text',
        line=dict(
            color=linecolor if linecolor else colormap,
            colorscale=colorscale,
            width=0
        )
    )

    return [node_trace]


def protein_backbone_trace(coords, res=None, coloring='spectrum', upsample=1000 ,show_supports=False, node_size=5):
    residues = np.random.randint(0, 20, [len(coords)])
    cmap = plt.cm.get_cmap('viridis')
    residues_rgba = cmap(residues / 20)[:, :-1]
    supports = point_cloud_trace(coords, residues_rgba, node_size=node_size+1, coloring=coloring)

    spline_pos, spline_residues = spline_interpolation(coords, upsample), linear_interpolation(residues_rgba, upsample)
    splines = point_cloud_trace(spline_pos, spline_residues, node_size=node_size, coloring=coloring)

    return splines + supports if show_supports else splines


def plot_aligned_timeseries(trajectory, node_size=1):
    axis = dict(backgroundcolor="rgba(0,0,0,0)", showticklabels=False, showgrid=False, zeroline=False, ticks='')
    trace_list = []
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Iteration:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    for i, (gnd, pred) in enumerate(trajectory):
        trace_list.append(aligned_pair_trace(pred, gnd, upsample=len(gnd) * 3, node_size=1))
        slider_step = {"args": [
            [str(i)],
            {"frame": {"duration": 0, "redraw": True},
             "mode": "e",
             "transition": {"duration": 0}}
        ],
            "label": str(i),
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)


    fig = go.Figure(
        data=trace_list[0],
        layout=go.Layout(
            xaxis=dict(range=[0, 5], autorange=False),
            yaxis=dict(range=[0, 5], autorange=False),
            title="Protein Assemblage",
            updatemenus=[

                {
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": 100, "redraw": True},
                                            "fromcurrent": True, "transition": {"duration": 100
                                                                                }}],
                            "label": "Play",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                              "mode": "immediate",
                                              "transition": {"duration": 0}}],
                            "label": "Pause",
                            "method": "animate"
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }

            ],
            margin=dict(b=20,l=5,r=5,t=40),
            scene = dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
            ),
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        ),
        frames=[go.Frame(data=traces, name=str(i)) for (i, traces) in enumerate(trace_list)],
    )

    fig["layout"]["sliders"] = [sliders_dict]
    return fig





def produce_3d_traces_row(*args):
    fig = make_subplots(rows=1, cols=len(args),
                    specs=[[{'is_3d': True} for _ in range(len(args))]],
                    print_grid=False)

    for i, traces in enumerate(args):
        for trace in traces: fig.append_trace(trace, row=1, col=i+1)
    fig.update_layout(
        **{ f'scene{idx if idx > 1 else ""}' : dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", showticklabels=False, showgrid=False, zeroline=False, ticks=''),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", showticklabels=False, showgrid=False, zeroline=False, ticks=''),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", showticklabels=False, showgrid=False, zeroline=False,ticks='')
        ) for idx in range(1, len(args) + 1)},
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def distogram_trace(distogram):
    return [go.Heatmap(
      z=-distogram,
      type='heatmap',
      colorscale='Inferno')]

def produce_distogram_traces_row(*args):
    fig = make_subplots(rows=1, cols=len(args))
    for i, traces in enumerate(args):
        for trace in traces: fig.append_trace(trace, row=1, col=i+1)
        fig['layout'][f'yaxis{i+1 if i > 0 else ""}'] = dict(
            autorange='reversed',  showgrid=False, zeroline=False,
            scaleanchor='x', scaleratio=1, )
        fig['layout'][f'xaxis{i+1 if i > 0 else ""}'].update(showgrid=False, zeroline=False)

    fig['layout']['autosize'] = False
    fig['layout'].update(plot_bgcolor='rgba(0,0,0,0)')
    fig['layout'].update(paper_bgcolor='rgba(0,0,0,0)')
    return fig
