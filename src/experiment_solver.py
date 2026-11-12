"""Frozen experiment runtime: fixed domains or bounded, cached moving windows."""
from pathlib import Path
import contextlib
import hashlib
import json
import os
import shutil
import time

import h5py
import numpy as np

from .soil import crop_scene, sha256


def signature(value):
    # Keep this byte-for-byte compatible with saved pre-schema experiments.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def normalized_positions(positions):
    """Add reporting coordinates to old two-array position archives in memory."""
    keys = ('tx_m', 'rx_m', 'requested_tx_m', 'requested_rx_m', 'distance_m', 'route_id', 'segment_id', 'route_names')
    result = {key: np.asarray(positions[key]) for key in keys if key in positions}
    tx = np.asarray(result['tx_m'], dtype=float)
    n = len(tx)
    result.setdefault('requested_tx_m', tx.copy())
    result.setdefault('requested_rx_m', np.asarray(result['rx_m'], dtype=float).copy())
    result.setdefault('route_id', np.zeros(n, dtype=np.int64))
    result.setdefault('segment_id', np.zeros(n, dtype=np.int64))
    result.setdefault('route_names', np.asarray(['route_0']))
    if 'distance_m' not in result:
        distance = np.zeros(n)
        for route in dict.fromkeys(result['route_id'].tolist()):
            ix = np.flatnonzero(result['route_id'] == route)
            distance[ix] = np.r_[0., np.cumsum(np.linalg.norm(np.diff(tx[ix], axis=0), axis=1))]
        result['distance_m'] = distance
    return result


def load_positions(directory):
    with np.load(Path(directory) / 'positions.npz', allow_pickle=False) as data:
        return normalized_positions(dict(data))


def crop_origins(cfg, positions):
    win = np.asarray(cfg.get('simulation', {}).get('window_xy', [1.2, 1.2]), dtype=float)
    midpoint = (positions['tx_m'][:, :2] + positions['rx_m'][:, :2]) / 2
    dx = float(cfg['grid']['dx'])
    return np.rint((midpoint - win / 2) / dx) * dx


def validate_positions(cfg, positions):
    """Validate every crop and receiver before generating the global geometry."""
    positions = normalized_positions(positions)
    g = cfg['grid']; domain = np.asarray(g['domain'], dtype=float)
    dx, coarse, pad = float(g['solver_dx']), float(g['dx']), float(g['pml_thickness'])
    factor = round(coarse / dx)
    if factor < 1 or not np.isclose(factor * dx, coarse):
        raise ValueError('solver_dx must divide geometry dx')
    tx, rx = positions['tx_m'], positions['rx_m']
    if tx.ndim != 2 or tx.shape[1:] != (3,) or not len(tx) or rx.shape != tx.shape:
        raise ValueError('Tx/Rx must have equal nonempty N × 3 coordinate arrays')
    for name in ('distance_m', 'route_id', 'segment_id'):
        if positions[name].shape != (len(tx),):
            raise ValueError(f'{name} must have one value per trace')
    if not np.isfinite(positions['distance_m']).all():
        raise ValueError('Route distances must be finite')
    route_ids = positions['route_id']
    if route_ids.dtype.kind not in 'iu' or np.any(route_ids < 0) or np.any(route_ids >= len(positions['route_names'])):
        raise ValueError('Route IDs must index route_names')
    for route in dict.fromkeys(route_ids.tolist()):
        distances = positions['distance_m'][route_ids == route]
        if not np.isclose(distances[0], 0) or np.any(np.diff(distances) <= 0):
            raise ValueError('Each route distance must start at zero and increase')
    for name in ('requested_tx_m', 'requested_rx_m'):
        if positions[name].shape != tx.shape or not np.isfinite(positions[name]).all():
            raise ValueError(f'{name} must have finite N × 3 coordinates')
    for coords in (tx, rx):
        if not np.isfinite(coords).all() or not np.allclose(coords/dx, np.rint(coords/dx), atol=1e-7, rtol=0):
            raise ValueError('Tx/Rx must lie on the solver grid')
        if np.any(coords < pad-1e-9) or np.any(coords > domain-pad+1e-9):
            raise ValueError('Tx/Rx intersects global PML')
    mode = cfg.get('simulation', {}).get('mode', 'fixed')
    if mode not in ('fixed', 'moving_window'):
        raise ValueError('simulation.mode must be fixed or moving_window')
    if mode == 'moving_window':
        win = np.asarray(cfg['simulation']['window_xy'], dtype=float)
        if win.shape != (2,) or not np.isfinite(win).all() or np.any(win <= 2*pad):
            raise ValueError('Moving window must have two dimensions larger than twice PML')
        if not np.allclose(win/coarse, np.rint(win/coarse), atol=1e-7, rtol=0):
            raise ValueError('Moving window must align with the geometry grid')
        origins = crop_origins(cfg, positions)
        if np.any(origins < -1e-9) or np.any(origins + win > domain[:2]+1e-9):
            raise ValueError('Moving-window crop outside scene; move the route inward or reduce window_xy')
        for coords in (tx, rx):
            local = coords[:, :2] - origins
            if np.any(local < pad-1e-9) or np.any(local > win-pad+1e-9):
                raise ValueError('Tx/Rx intersects moving-window PML; enlarge window_xy or reduce offset')
    return positions


def validate_scene(directory):
    scene = Path(directory) / 'scene'
    meta = json.loads((scene / 'soil.json').read_text())
    for name, key in [('geometry', 'geometry_sha256'), ('materials', 'materials_sha256')]:
        if sha256(scene / meta[name]) != meta[key]:
            raise ValueError('Frozen experiment geometry/materials changed')
    return meta


def _refine(source, dest, factor, dx):
    """Refine only the selected crop; fixed mode explicitly refines its full scene."""
    with h5py.File(source) as src, h5py.File(dest, 'w') as dst:
        a = src['data'][:]
        for axis in range(3):
            if factor > 1:a = np.repeat(a, factor, axis=axis)
        dst.create_dataset('data', data=a, compression='gzip')
        dst['material_keys'] = src['material_keys'][:]
        for name in ('MaterialDatabase', 'MaterialDatabaseSchemaVersion'):
            dst.attrs[name] = src.attrs[name]
        dst.attrs['dx_dy_dz'] = [dx]*3


def prepare_geometry(cfg, directory, meta, origin=None):
    """Return a verified cache entry. Cached files never define experiment identity."""
    directory = Path(directory); scene = directory/'scene'; run = directory/'run'
    run.mkdir(exist_ok=True)
    g = cfg['grid']; moving = origin is not None
    description = dict(geometry=meta['geometry_sha256'], materials=meta['materials_sha256'],
                       solver_dx=g['solver_dx'], origin=None if origin is None else np.asarray(origin).tolist(),
                       window=cfg.get('simulation', {}).get('window_xy') if moving else g['domain'][:2])
    cache = run/'cache'/signature(description)[:24] if moving else run
    cache.mkdir(parents=True, exist_ok=True)
    geometry, stamp = cache/'geometry.h5', cache/'geometry-cache.json'
    valid = False
    if geometry.exists() and stamp.exists():
        saved = json.loads(stamp.read_text())
        valid = saved.get('description') == description and saved.get('sha256') == sha256(geometry)
    if not valid:
        source = scene/meta['geometry']
        if moving:
            source = cache/'coarse.h5'
            crop_scene(scene, source, origin, window_xy=cfg['simulation']['window_xy'])
        temp = cache/'geometry.tmp.h5'
        _refine(source, temp, round(g['dx']/g['solver_dx']), g['solver_dx'])
        temp.replace(geometry)
        stamp.write_text(json.dumps(dict(description=description, sha256=sha256(geometry)), indent=2)+'\n')
        if moving:source.unlink(missing_ok=True)
    database = cache/meta['materials']
    if not database.exists() or sha256(database) != meta['materials_sha256']:
        shutil.copyfile(scene/meta['materials'], database)
    domain = [*cfg['simulation']['window_xy'], g['domain'][2]] if moving else g['domain']
    if moving:
        # Derived crop caches are bounded even for long surveys.
        stamp.touch()
        entries = sorted((p for p in (run/'cache').iterdir() if p.is_dir()),
                         key=lambda p: (p/'geometry-cache.json').stat().st_mtime if (p/'geometry-cache.json').exists() else 0, reverse=True)
        for old in entries[8:]:
            if old != cache:shutil.rmtree(old)
    return geometry, database.stem, domain


def input_text(cfg, geometry, database, tx, rx, domain=None):
    g, s = cfg['grid'], cfg['source']; dx = g['solver_dx']
    xyz = lambda values: ' '.join(f'{float(v):.12g}' for v in values)
    return '\n'.join([
        '#title: frozen experiment 3D FDTD',
        f'#domain: {xyz(domain or g["domain"])}', f'#dx_dy_dz: {dx} {dx} {dx}',
        f'#time_window: {g["time_window"]}', f'#num_threads: {g.get("threads", 2)}',
        f'#pml_cells: {round(g["pml_thickness"]/dx)}',
        f'#geometry_objects_read: 0 0 0 {geometry.name} {database} n',
        f'#waveform: {s["waveform"]} {s["moment_Am"]/dx} {s["frequency_hz"]} pulse',
        f'#hertzian_dipole: y {xyz(tx)} pulse', f'#rx: {xyz(rx)}', ''])


def source_values(cfg, dt, n):
    from gprMax.waveforms import Waveform
    w = Waveform(); w.type = cfg['source']['waveform']; w.freq = cfg['source']['frequency_hz']
    w.amp = cfg['source']['moment_Am']/cfg['grid']['solver_dx']
    times = (np.arange(n)+.5)*dt
    return times, np.asarray([w.calculate_value(float(t), dt) for t in times], dtype=np.float64)


def _store_positions(out, positions):
    for name, values in positions.items():
        values = np.asarray(values)
        if name in out:
            actual = out[name].asstr()[:] if out[name].dtype.kind in 'SO' else out[name][:]
            if not np.array_equal(actual, values):raise ValueError(f'Existing output {name} differs from frozen positions')
        else:
            if values.dtype.kind in 'UO':
                out.create_dataset(name, data=values.astype(object), dtype=h5py.string_dtype('utf-8'))
            else:out.create_dataset(name, data=values)


def run_survey(cfg, positions, directory, limit=None):
    """Resume only the saved configuration and saved coordinates, without reloading inputs."""
    directory = Path(directory); positions = validate_positions(cfg, positions)
    meta = validate_scene(directory); key = signature(cfg)
    moving = cfg.get('simulation', {}).get('mode', 'fixed') == 'moving_window'
    origins = crop_origins(cfg, positions) if moving else None
    n = len(positions['tx_m']); outpath = directory/'bscan.h5'
    with h5py.File(outpath, 'a') as out:
        if 'complete' not in out:
            out.attrs.update(config_sha256=key, source='gprMax 3D FDTD; raw point Ey', units='V/m',
                             axis_order='scan,time', fixed_domain=not moving,
                             simulation_mode='moving_window' if moving else 'fixed',
                             with_object=bool(cfg.get('experiment_objects', [])))
            out['complete'] = np.zeros(n, dtype=bool); out['seconds'] = np.zeros(n)
        if out.attrs.get('config_sha256') != key:
            raise ValueError('Existing output configuration differs from saved configuration')
        if out['complete'].shape != (n,):raise ValueError('Existing output trace count differs')
        if out['complete'][:].any():
            if 'Ey' not in out or out['Ey'].shape[0] != n or any(name not in out for name in ('time_s', 'source_time_s', 'source_current_A')):
                raise ValueError('Completed traces lack their saved signal/source arrays')
            if any(out[name].shape != (out['Ey'].shape[1],) for name in ('time_s', 'source_time_s', 'source_current_A')):
                raise ValueError('Saved time/source axes differ from signals')
        for name in ('geometry_sha256', 'materials_sha256'):
            if name in out.attrs and out.attrs[name] != meta[name]:raise ValueError('Output geometry identity differs')
            out.attrs[name] = meta[name]
        _store_positions(out, positions)
        if moving:
            if 'crop_origin_xy_m' in out and not np.array_equal(out['crop_origin_xy_m'][:], origins):
                raise ValueError('Saved moving-window origins differ')
            if 'crop_origin_xy_m' not in out:out['crop_origin_xy_m'] = origins
        todo = np.flatnonzero(~out['complete'][:]); todo = todo[:limit] if limit is not None else todo
        for i in todo:
            if (directory/'STOP').exists():break
            import gprMax
            start = time.monotonic(); origin = origins[i] if moving else None
            geometry, database, domain = prepare_geometry(cfg, directory, meta, origin)
            shift = np.r_[origin, 0.] if moving else np.zeros(3)
            fn = geometry.parent/'scan.in'
            fn.write_text(input_text(cfg, geometry, database, positions['tx_m'][i]-shift, positions['rx_m'][i]-shift, domain))
            with (directory/'run'/'last.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                gprMax.run(inputfile=str(fn), metal=(True if cfg['grid'].get('backend', 'cpu') == 'metal' else None), hide_progress_bars=True, log_level=40)
            with h5py.File(fn.with_suffix('.h5')) as raw:
                a = raw['rxs/rx1/Ey'][:]; dt = float(raw.attrs['dt'])
                if not np.isfinite(a).all() or not np.any(a):raise ValueError(f'Invalid FDTD field at trace {i}')
                if 'Ey' not in out:
                    out.create_dataset('Ey', shape=(n, len(a)), dtype='f4', chunks=(1, len(a)), compression='gzip')
                    out['time_s'] = np.arange(len(a))*dt
                    st, sv = source_values(cfg, dt, len(a)); out['source_time_s'] = st; out['source_current_A'] = sv
                    out.attrs.update(dt=dt, gprMax=raw.attrs['gprMax'], source_dl_m=cfg['grid']['solver_dx'])
                if len(a) != out['Ey'].shape[1] or dt != out.attrs['dt']:raise ValueError('Time grid changed')
                out['Ey'][i] = a; out['seconds'][i] = time.monotonic()-start
                out.flush(); out['complete'][i] = True; out.flush()
            progress = dict(mode=cfg['generator'], completed=int(out['complete'][:].sum()), total=n,
                            last_index=int(i), last_route=str(positions['route_id'][i]),
                            last_distance_m=float(positions['distance_m'][i]), last_seconds=float(out['seconds'][i]), pid=os.getpid())
            (directory/'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
            print(json.dumps(progress), flush=True)
    return outpath


def effective_sfcw(outpath, **sweep):
    # Retain the established physical units and exact source deconvolution.
    from tools.run_realism_pilot import effective_sfcw as process
    process(Path(outpath), **sweep)
    path = Path(outpath).parent/'effective_sfcw.npz'
    with h5py.File(outpath) as h:
        if not h['complete'][:].all():return
        route_data = {k: np.asarray(h[k].asstr()[:], dtype=str) if h[k].dtype.kind in 'SO' else h[k][:]
                      for k in ('distance_m', 'route_id', 'segment_id', 'route_names', 'tx_m', 'rx_m') if k in h}
    with np.load(path, allow_pickle=False) as data:values = dict(data)
    values.update(route_data)
    np.savez_compressed(path, **values)
