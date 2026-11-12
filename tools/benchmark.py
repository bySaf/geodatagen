#!/usr/bin/env python3
"""Compare explicit solver backends on identical, already prepared geometry.

Writes only to a new output directory. No measurement or result in the source
experiment is changed. Timing includes input loading/kernel compilation: it is
an end-to-end workload measurement, not a kernel-only accelerator claim.
"""
from pathlib import Path
import argparse
import json
import os
import platform
from importlib.metadata import version
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
import h5py
import numpy as np
from src.backends import probe_backends
from src import experiment_solver as solver
from src.soil import crop_scene


def crop_comparison(cfg, positions, source, dest, repetitions=3):
    """Compare previous coarse-file+gzip path with fused crop+LZF, exactly."""
    if cfg.get('simulation', {}).get('mode') != 'moving_window':return {'applicable': False}
    meta = solver.validate_scene(source); origin = solver.crop_origins(cfg, positions)[0]
    g = cfg['grid']; factor = round(g['dx']/g['solver_dx'])
    start = np.rint(origin/g['dx']).astype(int)
    count = np.rint(np.asarray(cfg['simulation']['window_xy'])/g['dx']).astype(int)
    slices = np.s_[start[0]:start[0]+count[0], start[1]:start[1]+count[1], :]
    directory = dest/'crop-comparison'; directory.mkdir()
    old_times, new_times = [], []
    for _ in range(repetitions):
        before = time.perf_counter()
        coarse, old = directory/'coarse.h5', directory/'previous.h5'
        crop_scene(source/'scene', coarse, origin, cfg['simulation']['window_xy'])
        with h5py.File(coarse) as src, h5py.File(old, 'w') as out:
            a = src['data'][:]
            for axis in range(3):a = np.repeat(a, factor, axis=axis)
            out.create_dataset('data', data=a, compression='gzip')
            out['material_keys'] = src['material_keys'][:]
            for name in ('MaterialDatabase', 'MaterialDatabaseSchemaVersion'):out.attrs[name] = src.attrs[name]
            out.attrs['dx_dy_dz'] = [g['solver_dx']]*3
        old_times.append(time.perf_counter()-before)
        before = time.perf_counter(); new = directory/'fused.h5'
        solver._refine(source/'scene'/meta['geometry'], new, factor, g['solver_dx'], slices=slices)
        new_times.append(time.perf_counter()-before)
        with h5py.File(old) as a, h5py.File(new) as b:
            for key in ('data', 'material_keys'):
                if not np.array_equal(a[key][:], b[key][:]):raise ValueError('Crop optimization changed geometry')
            if not np.array_equal(a.attrs['dx_dy_dz'], b.attrs['dx_dy_dz']):raise ValueError('Crop spacing changed')
    return dict(applicable=True, repetitions=repetitions, identical_material_ids=True,
                previous_seconds=old_times, fused_seconds=new_times,
                median_speedup=float(np.median(old_times)/np.median(new_times)),
                scope='one cache miss, bounded moving crop; not whole FDTD simulation speedup')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backends', nargs='+', choices=('cpu','cuda','metal'), default=['cpu','metal','cuda'])
    p.add_argument('--traces', type=int, default=1)
    p.add_argument('--relative-tolerance', type=float, default=1e-4,
                   help='Engineering acceptance threshold for relative L2 error; not a physics accuracy bound')
    args = p.parse_args(argv)
    if args.traces < 1 or not np.isfinite(args.relative_tolerance) or args.relative_tolerance <= 0:
        p.error('traces and relative tolerance must be positive')
    src, dest = args.data.resolve(), args.output.resolve()
    if dest.exists():raise ValueError('Benchmark output already exists; choose a new directory')
    from experiment import verify_frozen
    verify_frozen(src)
    solver.validate_scene(src)
    cfg = json.loads((src/'config.json').read_text())
    positions = solver.load_positions(src)
    count = min(args.traces, len(positions['tx_m']))
    positions = {key: value if key == 'route_names' else value[:count] for key, value in positions.items()}
    cfg['profile']['n'] = count
    availability = probe_backends(); dest.mkdir(parents=True)
    import gprMax
    report = dict(platform=platform.platform(), machine=platform.machine(), python=platform.python_version(),
                  gprMax=version('gprMax'), traces=count,
                  source_config_sha256=solver.signature(json.loads((src/'config.json').read_text())),
                  availability=availability, results={},
                  relative_tolerance=args.relative_tolerance,
                  timing_scope='fresh per-backend cache; includes geometry, input parsing and kernel compilation')
    report['crop_preparation'] = crop_comparison(cfg, positions, src, dest)
    signals = {}; axis = {}
    for name in dict.fromkeys(['cpu', *args.backends]):
        if not availability[name]['available']:
            report['results'][name] = dict(status='unavailable', reason=availability[name]['reason']); continue
        folder = dest/name; folder.mkdir()
        shutil.copytree(src/'scene', folder/'scene')
        (folder/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
        np.savez_compressed(folder/'positions.npz', **positions)
        before = time.perf_counter()
        try:
            path = solver.run_survey(cfg, positions, folder, backend=name)
            elapsed = time.perf_counter()-before
            with h5py.File(path) as h:
                if not h['complete'][:].all():raise ValueError('Benchmark was interrupted')
                signals[name] = h['Ey'][:].astype(float); axis[name] = h['time_s'][:]
                report['results'][name] = dict(status='completed', elapsed_seconds=elapsed,
                    geometry_seconds=float(h['geometry_seconds'][:].sum()),
                    solver_seconds=float(h['solve_seconds'][:].sum()))
        except Exception as exc:
            report['results'][name] = dict(status='failed', reason=f'{type(exc).__name__}: {exc}',
                                           elapsed_seconds=time.perf_counter()-before)
        (dest/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
    if 'cpu' in signals:
        norm = np.linalg.norm(signals['cpu'])
        for name, actual in signals.items():
            result = report['results'][name]
            if actual.shape != signals['cpu'].shape or not np.array_equal(axis[name], axis['cpu']):
                result.update(numerical_match=False, reason='Different time axes or signal shapes'); continue
            difference = actual-signals['cpu']
            relative = float(np.linalg.norm(difference)/max(norm, np.finfo(float).tiny))
            result.update(relative_l2_error=relative, max_abs_error_V_m=float(np.max(np.abs(difference))),
                          numerical_match=bool(np.isfinite(actual).all() and relative <= args.relative_tolerance),
                          end_to_end_speedup_vs_cpu=report['results']['cpu']['elapsed_seconds']/result['elapsed_seconds'])
    matched = [(v['elapsed_seconds'], name) for name, v in report['results'].items()
               if v.get('status') == 'completed' and v.get('numerical_match')]
    report['recommended_backend'] = min(matched)[1] if matched else None
    report['recommendation_scope'] = 'Only this geometry, trace count, hardware and cold-run workload; remeasure for larger models.'
    (dest/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if all(v.get('status') != 'failed' and v.get('numerical_match', True) for v in report['results'].values()) else 1


if __name__ == '__main__':
    try:raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:raise SystemExit(str(exc))
