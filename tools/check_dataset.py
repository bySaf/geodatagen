#!/usr/bin/env python3
"""Independent structural audit. Does not disable HDF5 locks or read live files."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone

import h5py
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()


def check_signal(path, tx, rx, config_hash):
    """Validate every finished row and its position; locked files raise OSError."""
    with h5py.File(path, 'r') as f:
        n = len(tx)
        if f.attrs.get('config_sha256') != config_hash: raise ValueError('configuration hash mismatch')
        for key, expected in [('tx_m', tx), ('rx_m', rx)]:
            if key not in f or not np.array_equal(f[key][:], expected): raise ValueError(f'{key} does not match path coordinates')
        mask = f['complete'][:]
        if mask.shape != (n,) or mask.dtype.kind != 'b': raise ValueError('invalid complete mask')
        done = int(mask.sum())
        if 'Ey' not in f:
            if done: raise ValueError('completed rows without signal')
            return dict(completed=0, total=n, valid=False)
        a = f['Ey']
        t = f['time_s'][:]
        dt = float(f.attrs['dt'])
        if a.ndim != 2 or a.shape != (n, len(t)) or len(t) < 2: raise ValueError('signal/time shape mismatch')
        if not np.isfinite(t).all() or not np.isfinite(dt) or dt <= 0: raise ValueError('invalid time axis')
        if not np.allclose(t, np.arange(len(t))*dt, rtol=1e-10, atol=1e-18): raise ValueError('time axis disagrees with dt')
        seconds = f['seconds'][:]
        if seconds.shape != (n,) or not np.isfinite(seconds[mask]).all() or np.any(seconds[mask] <= 0): raise ValueError('invalid completed timing metadata')
        for start in range(0, n, 32):
            rows = a[start:start+32][mask[start:start+32]]
            if not np.isfinite(rows).all(): raise ValueError(f'nonfinite signal after row {start}')
            if len(rows) and np.any(np.max(np.abs(rows), axis=1) == 0): raise ValueError(f'zero completed signal after row {start}')
        if f.attrs.get('units') != 'V/m' or f.attrs.get('axis_order') != 'scan,time': raise ValueError('missing signal units/axis order')
        return dict(completed=done, total=n, samples=len(t), dt_s=dt, valid=done == n)


def audit(data, geometry_only=False):
    data = Path(data)
    result = dict(mode='geometry_only' if geometry_only else 'full',
                  timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  complete=False, errors=[], scenes=[], completed_traces=0, expected_traces=0)
    errors = result['errors']
    try:
        cfg = json.loads((data/'config.json').read_text())
        key = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        result['config_sha256'] = key
        domain = np.array(cfg['grid']['domain']); dx = cfg['grid']['dx']; surface = cfg['grid']['surface_z']
        n = cfg['trajectory']['n']; presets = cfg['presets']
        if len(presets) < 3 or n < 500: errors.append('Requirement: at least 3 scenes and 500 scans/path')
        if (ROOT/'objects.txt').exists() and checksum(ROOT/'objects.txt') != cfg.get('object_spec_sha256'):
            errors.append('Current objects.txt differs from saved configuration hash')
    except Exception as exc:
        result['errors'].append(f'configuration: {exc}'); return result
    seeds=[]; types=set(); found_control=False; found_curved=False
    for scene_index,preset in enumerate(presets):
        directory=data/preset['name']; info=dict(name=preset['name'],paths=[]);result['scenes'].append(info)
        try:
            meta=json.loads((directory/'soil.json').read_text()); seeds.append(meta['seed'])
            if meta['seed'] != cfg['seed']+scene_index*100: raise ValueError('scene seed differs from configuration')
            if meta['domain'] != domain.tolist() or meta['dx'] != dx or meta['surface_z'] != surface: raise ValueError('soil grid differs from configuration')
            for filename, digest in [('geometry','geometry_sha256'),('materials','materials_sha256')]:
                if checksum(directory/meta[filename]) != meta[digest]: raise ValueError(f'{filename} SHA256 mismatch')
            with h5py.File(directory/meta['geometry'],'r') as f:
                shape=np.rint(domain/dx).astype(int)
                if tuple(f['data'].shape) != tuple(shape) or meta['shape'] != shape.tolist(): raise ValueError('voxel grid shape mismatch')
                if not np.allclose(f.attrs['dx_dy_dz'],dx): raise ValueError('voxel resolution mismatch')
                keys=f['material_keys'][:]
                for i in range(0,len(f['data']),32):
                    values=f['data'][i:i+32]
                    if values.min() < 0 or values.max() >= len(keys): raise ValueError('voxel material index outside material table')
            if not (directory/'scene.vtkhdf').is_file(): raise ValueError('ParaView geometry missing')
            objects=json.loads((directory/'objects.json').read_text()); info['object_count']=len(objects)
            if preset.get('control') or preset['name']=='field':
                found_control=True
                if objects: raise ValueError('control scene contains objects')
            for i,o in enumerate(objects):
                centre=np.asarray(o['center_m']); radius=o['radius_m']; types.add(o['material'])
                if radius <= 0 or not np.isfinite(centre).all() or np.any(centre-radius < 0) or np.any(centre+radius > domain): raise ValueError('object outside domain')
                if centre[2]+radius > surface or centre[2]-radius < .2-1e-9: raise ValueError('object outside usable soil')
                if not np.isclose(o['depth_center_m'],surface-centre[2]): raise ValueError('object depth metadata mismatch')
                for previous in objects[:i]:
                    if np.linalg.norm(centre-previous['center_m']) < radius+previous['radius_m']+.04-1e-9: raise ValueError('objects intersect or violate clearance')
            paths=[]
            for path_id in range(3):
                with np.load(directory/f'path_{path_id}.npz') as p:
                    tx,rx=p['tx'],p['rx'];rtx,rrx=p['requested_tx'],p['requested_rx']
                if any(a.shape != (n,3) or not np.isfinite(a).all() for a in [tx,rx,rtx,rrx]): raise ValueError('path size/finite coordinate failure')
                if not np.allclose(tx,np.round(rtx/dx)*dx) or not np.allclose(rx,np.round(rrx/dx)*dx): raise ValueError('snapped/requested coordinates disagree')
                if not np.allclose(rx-tx,[cfg['trajectory']['tx_rx_offset'],0,0]): raise ValueError('Tx/Rx separation mismatch')
                origin=np.rint((tx[:,:2]-np.asarray(cfg['window_xy'])/2)/dx)*dx
                if np.any(origin<0) or np.any(origin+cfg['window_xy']>domain[:2]+1e-9): raise ValueError('moving window outside global domain')
                if np.any(tx[:,2] <= surface) or np.any(rx[:,2] <= surface) or np.any(tx[:,2] >= domain[2]-.1): raise ValueError('antenna height invalid')
                step=np.linalg.norm(np.diff(tx,axis=0),axis=1)
                nominal=cfg['trajectory']['step']
                if np.any(step<.5*nominal-1e-8) or np.any(step>1.3*nominal+1e-8): raise ValueError('actual path step outside .5–1.3 nominal tolerance')
                directions=np.diff(rtx[:,:2],axis=0)
                found_curved |= len(np.unique(np.round(directions,6),axis=0))>4
                paths.append(tx[:,:2]); item=dict(id=path_id,positions=n,step_min_m=float(step.min()),step_max_m=float(step.max()))
                result['expected_traces']+=n
                if not geometry_only:
                    try:
                        item['signal']=check_signal(directory/f'bscan_{path_id}.h5',tx,rx,key)
                        if 'samples' in item['signal']:
                            duration=item['signal']['samples']*item['signal']['dt_s']
                            if abs(duration-cfg['grid']['time_window']) > 2*item['signal']['dt_s']:
                                raise ValueError('signal duration differs from configured time window')
                        result['completed_traces']+=item['signal']['completed']
                        if not item['signal']['valid']: errors.append(f'{preset["name"]}/path_{path_id}: incomplete')
                    except Exception as exc: errors.append(f'{preset["name"]}/path_{path_id}: {exc}')
                info['paths'].append(item)
            radius=cfg['overlap_radius']
            overlaps=np.array([[np.mean(cKDTree(b).query(a)[0]<=radius) for b in paths] for a in paths])
            stored=json.loads((directory/'overlap.json').read_text())
            if stored['radius_m']!=radius or not np.allclose(overlaps,stored['matrix']): raise ValueError('stored overlap disagrees with actual coordinates')
            pairs=[min(overlaps[i,j],overlaps[j,i]) for i in range(3) for j in range(i)]
            if max(pairs)<.5 or min(pairs)>0: raise ValueError('need substantial overlap and a disjoint pair')
            info['overlap_matrix']=overlaps.tolist()
        except Exception as exc: errors.append(f'{preset["name"]}: {exc}')
    if len(seeds)!=len(set(seeds)): errors.append('scene seeds are not distinct')
    if not found_control: errors.append('no control scene')
    if not {'pec','free_space','inclusion'} <= types: errors.append('three object materials absent across dataset')
    if not found_curved: errors.append('no curved trajectory')
    if not geometry_only and (result['completed_traces']<4500 or result['completed_traces']!=result['expected_traces']): errors.append('Full minimum 4500 traces not proven complete')
    result['geometry_verified']=not errors if geometry_only else None
    result['complete']=not geometry_only and not errors
    result['status']='geometry_verified_signals_unchecked' if geometry_only and not errors else ('complete' if result['complete'] else 'incomplete_or_invalid')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=ROOT/'data'/'final');p.add_argument('--geometry-only',action='store_true');args=p.parse_args()
    result=audit(args.data,args.geometry_only)
    args.data.mkdir(parents=True,exist_ok=True)
    target=args.data/('audit-geometry.json' if args.geometry_only else 'audit.json')
    target.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    sys.exit(1 if result['errors'] else 0)

if __name__=='__main__': main()
