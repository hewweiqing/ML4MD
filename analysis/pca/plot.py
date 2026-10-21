"""2D and static-3D PCA panels, with one combined PDF. No native model imports."""
import argparse
import gc
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from analyze import load, read

FAMILY_COLORS = dict(zip(
    ('Transition metals', 'Chalcogens', 'Other main group', 'Alkali / alkaline earth',
     'H', 'Halogens', 'Lanthanides / actinides', 'Noble gases'),
    ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f')))


def scatter(metadata, level, view, saved, pair=None):
    """pair=None produces a fixed-camera 3D page, otherwise a 2D projection."""
    keys = saved['checkpoint_keys'].tolist()
    points = dict(zip(keys, saved['coordinates']))
    labels = saved['labels']
    ratios = saved['explained_variance_ratio']
    primary, titles = metadata['primary'], metadata['panel_titles']
    columns = len(primary) // 2
    is_3d = pair is None
    colours = FAMILY_COLORS if level == 'individual' else {'Non-metal': '#1f77b4', 'Metal': '#ff7f0e'}
    xyz = saved['coordinates'].reshape(-1, 3)
    axes_index = [0, 1, 2] if is_3d else list(pair)
    lo, hi = xyz[:, axes_index].min(0), xyz[:, axes_index].max(0)
    pad = np.maximum((hi - lo) * .06, 1e-6)
    if is_3d:
        fig = plt.figure(figsize=(5 * columns, 11))
        axes = []
        for i in range(len(primary)):
            row, col = divmod(i, columns)
            axes.append(fig.add_axes([.025 + col * .94 / columns, .55 - row * .43,
                                      .94 / columns - .05, .34], projection='3d'))
    else:
        fig, grid = plt.subplots(2, columns, figsize=(3.8 * columns, 7.2),
                                 sharex=True, sharey=True, layout='constrained')
        axes = grid.ravel()
    for ax, key, title in zip(axes, primary, titles):
        values = points[key][:, axes_index]
        for label, colour in colours.items():
            mask = labels == label
            if not mask.any():
                continue
            options = dict(s=(7 if level == 'individual' else 15) if is_3d else
                             (8 if level == 'individual' else 16),
                           alpha=.55, color=colour, label=label, edgecolors='none', rasterized=True)
            if is_3d:
                options['depthshade'] = False
            ax.scatter(*values[mask].T, **options)
        ax.set_title(title, fontsize=10 if is_3d else None, pad=12 if is_3d else 6)
        setters = [ax.set_xlim, ax.set_ylim] + ([ax.set_zlim] if is_3d else [])
        label_setters = [ax.set_xlabel, ax.set_ylabel] + ([ax.set_zlabel] if is_3d else [])
        for i, (setter, label_setter) in enumerate(zip(setters, label_setters)):
            setter(lo[i] - pad[i], hi[i] + pad[i])
            options = dict(fontsize=8, labelpad=4) if is_3d else {}
            label_setter(f'PC{axes_index[i] + 1} ({100 * ratios[axes_index[i]]:.1f}%)', **options)
        if is_3d:
            ax.tick_params(labelsize=7, pad=0)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_major_locator(MaxNLocator(4))
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=22, azim=-55)
    term = metadata['individual_term'] if level == 'individual' else 'Material'
    method = str(saved.get('anchor_method', np.array('baseline')))
    description = 'raw (no anchor transform)' if view == 'raw' else f'relative, {method.upper() if method == "direct" else method} anchors'
    if is_3d:
        fig.suptitle(f'{metadata["name"]} {term.lower()} representations - {description}: static 3D\n'
                     f'Seed 0; W5/S20; same saved training-only PCA; PC1-PC3 retain '
                     f'{100 * ratios.sum():.1f}% reference variance', fontsize=13, y=.98)
        handles = [Line2D([], [], marker='o', linestyle='', color=c, label=label)
                   for label, c in colours.items() if np.any(labels == label)]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .045),
                   ncol=min(4, len(handles)), fontsize=9, frameon=False)
        fig.text(.5, .015,
                 'Fixed camera and common limits across checkpoints; percentages refer to the combined training reference.\n'
                 'Static projection only: overlapping points can hide structure; PCA variance is not prediction accuracy.',
                 ha='center', fontsize=9)
    else:
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='outside lower center',
                   ncol=min(4, len(handles)), fontsize=8)
        fig.suptitle(f'{metadata["name"]} {term.lower()} representations - {description}\n'
                     f'Seed 0; W5/S20; one training-only PCA; PC1-PC3 retain {100 * ratios.sum():.1f}% reference variance\n'
                     'Projection is not an information-content or calibration measurement', fontsize=11)
    return fig


def plot_run(run, output):
    run, output = Path(run), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Figure directory is nonempty; choose another --output: {output}')
    config = read(run / 'config.json')
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    with PdfPages(output / 'Cross_model_PCA_only_2D_3D.pdf') as pdf:
        info = pdf.infodict()
        info.update(Title='Cross-model PCA plots: 2D and static 3D',
                    Subject=f'{config.get("anchor_method", "baseline")} anchors; training-only PCA; seed 0; W5/S20')
        for model, metadata in config['models'].items():
            for level in ('individual', 'material'):
                for view in ('raw', 'relative'):
                    saved = load(run / model / f'{level}_{view}.npz')
                    saved['anchor_method'] = np.array(config.get('anchor_method', 'baseline'))
                    for pair in ((0, 1), (0, 2), (1, 2), None):
                        fig = scatter(metadata, level, view, saved, pair)
                        suffix = 'PC123_static_3D' if pair is None else f'PC{pair[0] + 1}_PC{pair[1] + 1}'
                        fig.canvas.draw()
                        extras = list(fig.texts) + list(fig.legends)
                        for ax in fig.axes:
                            extras.extend([ax.xaxis.label, ax.yaxis.label])
                            if pair is None:
                                extras.append(ax.zaxis.label)
                        options = dict(bbox_inches='tight', pad_inches=.25, bbox_extra_artists=extras)
                        pdf.savefig(fig, dpi=180, **options)
                        fig.savefig(output / f'{model}_{level}_{view}_{suffix}.png', dpi=150, **options)
                        plt.close(fig)
                        del fig
                        gc.collect()  # release closed raster canvases before the next page
                        count += 1
            print(f'{metadata["name"]}: 16 PCA pages rendered', flush=True)
    print(f'Created {count} pages: {output / "Cross_model_PCA_only_2D_3D.pdf"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Output folder produced by analyze.py')
    parser.add_argument('--output', type=Path, help='New folder for the PDF and PNGs')
    args = parser.parse_args()
    plot_run(args.run, args.output or args.run / 'figures')
