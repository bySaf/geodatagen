#!/usr/bin/env python3
"""Prepare and resume a frozen scene and arbitrary routes in one experiment folder."""
from pathlib import Path
import argparse
import contextlib
import csv
import fcntl
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
import h5py
import numpy as np
from src.layout import object_commands
from src.physical_constants import INCLUSION_EPSILON_R, INCLUSION_CONDUCTIVITY_S_M
from src.soil import GENERATORS, prepare_scene, sha256
from src import experiment_solver as solver


def destination(path):
    path = Path(path).resolve()
    protected = [ROOT/'data/final', ROOT/'user_review/realism/pilot']
    if any(path == p or path.is_relative_to(p) or p.is_relative_to(path) for p in protected):
        raise ValueError('Choose a NEW experiment directory, outside the completed datasets')
    return path


def settings(path, mode=None):
    """Compatibility wrapper for callers expecting the original three-item result."""
    from src.experiment_config import load_settings
    cfg, positions = load_settings(path, mode=mode)
    return cfg, positions['tx_m'], positions['rx_m']


@contextlib.contextmanager
def locked(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.experiment.lock').open('a') as lock:
        try:fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This experiment is running; use status or wait')
        yield


def route_name(positions, route):
    names = positions.get('route_names', [])
    try:return str(names[int(route)])
    except (ValueError, IndexError, TypeError):return str(route)


def write_trajectory_csv(directory, positions):
    with (directory/'trajectory.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['trace_index', 'route_id', 'route_name', 'segment_id', 'distance_m',
                         'tx_x_m', 'tx_y_m', 'tx_z_m', 'rx_x_m', 'rx_y_m', 'rx_z_m',
                         'requested_tx_x_m', 'requested_tx_y_m', 'requested_tx_z_m',
                         'requested_rx_x_m', 'requested_rx_y_m', 'requested_rx_z_m'])
        for i, route in enumerate(positions['route_id']):
            writer.writerow([i, route, route_name(positions, route), positions['segment_id'][i],
                             positions['distance_m'][i], *positions['tx_m'][i], *positions['rx_m'][i],
                             *positions['requested_tx_m'][i], *positions['requested_rx_m'][i]])


def verify_frozen(directory):
    """New experiments have immutable input hashes; old folders remain resumable."""
    manifest = directory/'frozen.json'
    if manifest.exists():
        for name, checksum in json.loads(manifest.read_text()).items():
            if sha256(directory/name) != checksum:raise ValueError(f'Frozen experiment file changed: {name}')


def prepare(directory, cfg, positions, rx=None):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    if rx is not None:positions = dict(tx_m=positions, rx_m=rx)
    positions = solver.validate_positions(cfg, positions)
    stamp = directory/'config.json'
    if stamp.exists():
        if json.loads(stamp.read_text()) != cfg:
            raise ValueError('Saved settings differ. Choose a NEW --data directory')
        verify_frozen(directory)
    else:
        if any(p.name != '.experiment.lock' for p in directory.iterdir()):
            raise ValueError('Output directory is not empty and has no experiment config')
        stamp.write_text(json.dumps(cfg, indent=2, ensure_ascii=False)+'\n')
    archive = directory/'positions.npz'
    if archive.exists():
        saved = solver.load_positions(directory)
        if set(saved) != set(positions) or any(not np.array_equal(saved[k], positions[k]) for k in saved):
            raise ValueError('Saved positions differ. Choose a NEW --data directory')
    else:np.savez_compressed(archive, **positions)
    scene = directory/'scene'
    if not (scene/'soil.json').exists():
        lines = [f'#material: {INCLUSION_EPSILON_R} {INCLUSION_CONDUCTIVITY_S_M} 1 0 inclusion']
        lines += [command+' n' for command in object_commands(cfg['experiment_objects'])]
        prepare_scene(scene, cfg['preset'], cfg['grid'], cfg['seed'], object_lines=lines, generator=cfg['generator'])
    solver.validate_scene(directory)
    # No full fine-grid volume is built here. Moving mode refines bounded crops on demand.
    objects = directory/'objects.json'
    if not objects.exists():objects.write_text(json.dumps(cfg['experiment_objects'], indent=2, ensure_ascii=False)+'\n')
    if not (directory/'trajectory.csv').exists():write_trajectory_csv(directory, positions)
    if cfg.get('schema_version') and not (directory/'frozen.json').exists():
        (directory/'frozen.json').write_text(json.dumps({name: sha256(directory/name) for name in
            ('config.json', 'positions.npz', 'scene/soil.json')}, indent=2)+'\n')
    layout_plot(directory, cfg, positions)


def layout_plot(directory, cfg, positions, rx=None):
    if rx is not None:positions = dict(tx_m=positions, rx_m=rx)
    positions = solver.normalized_positions(positions)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    tx, rx = positions['tx_m'], positions['rx_m']
    g = cfg['grid']; fig, axs = plt.subplots(2, 1, figsize=(10, 8), layout='constrained')
    for route in dict.fromkeys(positions['route_id'].tolist()):
        ix = np.flatnonzero(positions['route_id'] == route); name = route_name(positions, route)
        line, = axs[0].plot(tx[ix, 0], tx[ix, 1], '.-', label=f'{name}: Tx')
        axs[0].plot(rx[ix, 0], rx[ix, 1], '.--', color=line.get_color(), alpha=.65, label=f'{name}: Rx')
        axs[0].annotate(name, tx[ix[0], :2], xytext=(5, 5), textcoords='offset points', fontsize=8)
        axs[1].plot(positions['distance_m'][ix], g['surface_z']-tx[ix, 2], '.-', label=name)
        for k in range(1, len(ix)):
            if positions['segment_id'][ix[k]] != positions['segment_id'][ix[k-1]]:
                axs[0].plot(*tx[ix[k], :2], 'o', mfc='none', mec=line.get_color(), ms=10)
    colors = {'pec': 'black', 'free_space': 'white', 'inclusion': '#a6a6a6'}
    for obj in cfg['experiment_objects']:
        x, y, _ = obj['center_m']; r = obj['radius_m']
        axs[0].add_patch(Circle((x, y), r, facecolor=colors[obj['material']], edgecolor='black'))
    d = 0
    for layer in cfg['preset']['layers']:
        d += layer['thickness']; axs[1].axhline(d, color='.6', ls='--', lw=.8)
    axs[0].set(xlim=(0, g['domain'][0]), ylim=(0, g['domain'][1]), xlabel='X, м', ylabel='Y, м',
               title='План: маршруты Tx/Rx и проекции объектов; кружки — смена сегмента', aspect='equal')
    axs[0].legend(fontsize=8)
    axs[1].set(ylim=(g['surface_z'], -max(.1, cfg['source']['height_m']*2)),
               xlabel='Путь Tx от начала своего маршрута, м', ylabel='Глубина, м',
               title='Высота Tx вдоль маршрутов; пунктир — номинальные границы грунта')
    axs[1].legend(fontsize=8)
    fig.savefig(directory/'scene_layout.png', dpi=150); plt.close(fig)


def route_background(values, done, route_ids):
    result = np.full_like(values, np.nan, dtype=float)
    for route in dict.fromkeys(route_ids.tolist()):
        ix = np.flatnonzero(route_ids == route); ready = ix[done[ix]]
        if len(ready):result[ix] = values[ix] - np.mean(values[ready], axis=0, keepdims=True)
    return result


def _edges(values, fallback):
    if len(values) == 1:return np.array([values[0]-fallback/2, values[0]+fallback/2])
    middle = (values[:-1]+values[1:])/2
    return np.r_[values[0]-(middle[0]-values[0]), middle, values[-1]+(values[-1]-middle[-1])]


def plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    with h5py.File(directory/'bscan.h5') as h:
        done = h['complete'][:]
        if not done.any():raise ValueError('No completed traces to plot')
        a = h['Ey'][:].astype(float); t = h['time_s'][:]*1e9
    positions = solver.load_positions(directory); a[~done] = np.nan
    if not np.isfinite(a[done]).all():raise ValueError('Nonfinite completed signal')
    cfg = json.loads((directory/'config.json').read_text())
    distance = positions['distance_m']; routes = list(dict.fromkeys(positions['route_id'].tolist()))
    bg = route_background(a, done, positions['route_id'])
    cmap = plt.get_cmap('gray').with_extremes(bad='#b7bdc4')
    te = _edges(t, 1); fallback = cfg.get('profile', {}).get('step_m') or .05
    # Keep each image bounded even when a survey contains many independent routes.
    for page, offset in enumerate(range(0, len(routes), 4)):
        page_routes = routes[offset:offset+4]
        fig, axs = plt.subplots(len(page_routes), 2, figsize=(12, 4*len(page_routes)), squeeze=False, layout='constrained')
        for row, route in enumerate(page_routes):
            ix = np.flatnonzero(positions['route_id'] == route); x = distance[ix]
            xe = _edges(x, fallback); name = route_name(positions, route)
            for ax, values, title in zip(axs[row], [a[ix], bg[ix]], ['Исходное поле Ey', 'Минус средняя трасса этого маршрута']):
                finite = np.abs(values[np.isfinite(values)])
                limit = float(np.percentile(finite, 99)) if finite.size else 1; limit = limit or 1
                im = ax.pcolormesh(xe, te, np.ma.masked_invalid(values.T/limit), cmap=cmap, vmin=-1, vmax=1, shading='flat', rasterized=True)
                ax.set_ylim(te[-1], te[0]); ax.set_facecolor('#b7bdc4')
                ax.set(title=f'{name} · {title}', xlabel='Путь Tx от начала маршрута, м', ylabel='Время, нс')
                for k in range(1, len(ix)):
                    if positions['segment_id'][ix[k]] != positions['segment_id'][ix[k-1]]:
                        ax.axvline(x[k], color='#ba3b30', ls='--', lw=.8)
                fig.colorbar(im, ax=ax, label='Амплитуда / P99 панели')
            if done[ix].sum() < 2:
                axs[row, 1].text(.5, .5, 'Для сравнения нужны хотя бы 2 трассы', transform=axs[row, 1].transAxes, ha='center', bbox=dict(facecolor='white'))
        fig.suptitle(f'Готово {done.sum()} из {len(done)}; серый — не рассчитано, красный пунктир — смена сегмента')
        filename = 'bscan.png' if page == 0 else f'bscan_{page+1:03d}.png'
        fig.savefig(directory/filename, dpi=160); plt.close(fig)
    i = np.flatnonzero(done)[-1]; fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
    ax.plot(t, a[i], color='black')
    ax.set(xlabel='Время, нс', ylabel='Ey, В/м', title=f'A-скан №{i}: {route_name(positions, positions["route_id"][i])}, путь Tx {distance[i]:.2f} м')
    ax.grid(alpha=.2); fig.savefig(directory/'ascan.png', dpi=160); plt.close(fig)


def status(directory):
    try:
        with locked(directory):
            cfg = json.loads((directory/'config.json').read_text()); total = cfg['profile']['n']; done = 0
            if (directory/'bscan.h5').exists():
                with h5py.File(directory/'bscan.h5') as h:done = int(h['complete'][:].sum())
            return dict(running=False, completed=done, total=total, stopped_by_file=(directory/'STOP').exists())
    except RuntimeError:
        try:p = json.loads((directory/'progress.json').read_text())
        except (OSError, ValueError):p = dict(note='First trace running or progress is being updated')
        return dict(p, running=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'status', 'plot'])
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--config', type=Path, help='prepare only: configuration JSON')
    parser.add_argument('--trajectory', type=Path, help='prepare only: route JSON or CSV override')
    parser.add_argument('--mode', choices=GENERATORS, help='prepare only: soil generator override')
    parser.add_argument('--limit', type=int, help='run only: maximum number of NEW traces')
    parser.add_argument('--backend', choices=('cpu', 'metal', 'cuda', 'auto'), help='run only: CPU/GPU override')
    args = parser.parse_args(); directory = destination(args.data)
    if args.command == 'prepare' and args.config is None:parser.error('prepare requires --config')
    if args.command != 'prepare' and any(x is not None for x in (args.config, args.mode, args.trajectory)):
        parser.error('run/status/plot use only frozen config and positions')
    if args.limit is not None and (args.command != 'run' or args.limit < 1):parser.error('--limit must be positive and is only for run')
    if args.backend is not None and args.command != 'run':parser.error('--backend is only for run')
    if args.command != 'prepare' and not (directory/'config.json').exists():parser.error('Prepare this experiment first')
    if args.command == 'status':print(json.dumps(status(directory), indent=2)); return
    with locked(directory):
        if args.command == 'plot':plot(directory); print(directory/'bscan.png'); return
        if args.command == 'prepare':
            from src.experiment_config import load_settings
            original = args.config.read_text()
            cfg, positions = load_settings(args.config, mode=args.mode, trajectory_path=args.trajectory)
            prepare(directory, cfg, positions)
            if not (directory/'source.json').exists():(directory/'source.json').write_text(original)
        else:
            verify_frozen(directory)
            cfg = json.loads((directory/'config.json').read_text()); positions = solver.load_positions(directory)
            out = solver.run_survey(cfg, positions, directory, limit=args.limit, backend=args.backend)
            s = cfg['sfcw']; solver.effective_sfcw(out, fmin=s['fmin_hz'], fmax=s['fmax_hz'], df=s['df_hz'])
        print('READY', directory, flush=True)


if __name__ == '__main__':
    try:main()
    except (ValueError, RuntimeError, FileNotFoundError) as exc:raise SystemExit(str(exc))
