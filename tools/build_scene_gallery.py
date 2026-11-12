"""Render the documented full-scene examples from prepared experiment geometry."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from experiment_outputs import geometry_fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', nargs=3, type=Path, required=True, metavar=('PARK', 'FIELD', 'ROAD'))
    parser.add_argument('--out', type=Path, required=True, help='Output directory for gallery figures and metadata')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), layout='constrained')
    plan, maps = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#20252b')
    metadata = []
    for path, title, ax, xy in zip(args.experiments, ['Парк', 'Поле', 'Дорога'], axes, maps):
        cfg, fields, dx, frequency, _ = geometry_fields(path)
        g = cfg['grid']; er = fields['epsilon_real']; surface = g['surface_z']
        j = int(1.0/dx[1]); depth = .36; k = int((surface-depth)/dx[2])
        im = ax.imshow(er[:, j, :].T, origin='lower', extent=[0, g['domain'][0], surface, surface-g['domain'][2]],
                       cmap=cmap, vmin=1, vmax=20, interpolation='nearest', aspect='equal')
        boundary = 0
        for layer in cfg['preset']['layers']:
            ax.axhline(boundary, ls='--', lw=.7, color='white'); boundary += layer['thickness']
        ax.set(xlim=(0, 7), ylim=(.9, -.4), xlabel='X, м', ylabel='Глубина, м', title=f'{title} · Y = 1 м')
        xy.imshow(er[:, :, k].T, origin='lower', extent=[0, g['domain'][0], 0, g['domain'][1]], cmap=cmap,
                  vmin=1, vmax=20, interpolation='nearest', aspect='equal')
        with np.load(path/'positions.npz') as p:
            tx, routes = p['tx_m'], p['route_id']
            for r in np.unique(routes):
                points = tx[routes == r]
                xy.plot(points[:, 0], points[:, 1], color='white', lw=1.3)
                xy.scatter(*points[0, :2], c='#f8e63c', edgecolors='black', s=55, zorder=4)
                xy.scatter(*points[-1, :2], c='#ef6048', edgecolors='black', marker='s', s=40, zorder=4)
        for obj in cfg['experiment_objects']:
            x, y, _ = obj['center_m']; xy.add_patch(plt.Circle((x, y), obj['radius_m'], fc='black', ec='white', lw=1))
        xy.set(xlabel='X, м', ylabel='Y, м', title=title)
        metadata.append(dict(scene=cfg['preset']['name'], config=f'examples/scenes/{cfg["preset"]["name"]}.json',
                             generator=cfg['generator'], frequency_hz=frequency, section_y_m=(j+.5)*dx[1],
                             plan_depth_m=surface-(k+.5)*dx[2], trace_count=len(tx), objects=cfg['experiment_objects']))
    fig.colorbar(im, ax=list(axes), label='Относительная проницаемость εr при 600 МГц', shrink=.85)
    fig.suptitle('Полные сцены нового формата · модель полей CRIM/Archie', fontsize=15)
    plan.suptitle('Сцена и маршрут задаются отдельно · фон на глубине около 36 см\nЖёлтый круг — старт; красный квадрат — конец; чёрный — проекции объектов', fontsize=13)
    fig.savefig(args.out/'full_scenes.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    plan.savefig(args.out/'scene_routes.png', dpi=150, bbox_inches='tight'); plt.close(plan)
    (args.out/'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n')
    print(args.out)


if __name__ == '__main__':
    main()
