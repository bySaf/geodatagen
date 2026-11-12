"""Moving-window FDTD. One HDF5 per path; interrupted jobs resume by completion mask."""
from pathlib import Path
import contextlib, hashlib, json, os, time, copy
import h5py
import numpy as np
from .soil import crop_scene, sha256


def signature(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()


def input_text(grid, crop, tx, rx, window, material_database):
    xyz=lambda v:' '.join(f'{x:.9g}' for x in v)
    return '\n'.join([
        '#title: frozen scene moving window',
        f'#domain: {xyz([*window,grid["domain"][2]])}',
        f'#dx_dy_dz: {xyz([grid["dx"]]*3)}',
        f'#time_window: {grid["time_window"]}',
        f'#num_threads: {grid["threads"]}',
        f'#pml_cells: {round(grid.get("pml_thickness",.2)/grid["dx"])}',
        f'#geometry_objects_read: 0 0 0 {crop} {material_database} n',
        f'#waveform: ricker {grid.get("source_moment",0.02)/grid["dx"]} {grid["frequency"]} pulse',
        f'#hertzian_dipole: y {xyz(tx)} pulse',
        f'#rx: {xyz(rx)}', ''])


def simulate_path(scene_dir, path_id, config, limit=None):
    import gprMax
    scene_dir=Path(scene_dir).resolve(); g=copy.deepcopy(config['grid']); win=config['window_xy']
    geometry_dx=g['dx'];g['dx']=g.get('solver_dx',geometry_dx)
    factor=round(geometry_dx/g['dx'])
    if factor<1 or not np.isclose(factor*g['dx'],geometry_dx): raise ValueError('solver_dx must divide geometry dx')
    # Files are unique per path, allowing independent worker processes.
    meta=json.loads((scene_dir/'soil.json').read_text())
    for name,key in [('geometry','geometry_sha256'),('materials','materials_sha256')]:
        if sha256(scene_dir/meta[name])!=meta[key]: raise ValueError('Frozen scene was changed')
    coords=np.load(scene_dir/f'path_{path_id}.npz')
    txs,rxs=coords['tx'],coords['rx']; n=len(txs)
    run_dir=scene_dir/'runs'/f'path_{path_id}'; run_dir.mkdir(parents=True,exist_ok=True)
    fn=run_dir/'scan.in'; crop=run_dir/'crop.h5'; result=scene_dir/f'bscan_{path_id}.h5'
    key=signature(config)
    with h5py.File(result,'a') as out:
        if 'complete' not in out:
            out.attrs.update(config_sha256=key,source='gprMax FDTD; raw Ey',scene=scene_dir.name,units='V/m',axis_order='scan,time',window_xy=win)
            out.create_dataset('complete',data=np.zeros(n,dtype=bool))
            out.create_dataset('tx_m',data=txs);out.create_dataset('rx_m',data=rxs)
            out.create_dataset('seconds',data=np.zeros(n))
        if out.attrs['config_sha256']!=key: raise ValueError('Configuration changed: choose a new dataset directory.')
        if not np.array_equal(out['tx_m'][:],txs) or not np.array_equal(out['rx_m'][:],rxs):
            raise ValueError('Saved receiver/source coordinates differ from the path')
        todo=np.flatnonzero(~out['complete'][:]); todo=todo[:limit] if limit is not None else todo
        for i in todo:
            if (scene_dir.parent/'STOP').exists(): break
            start=time.monotonic()
            origin=np.rint((txs[i,:2]-np.array(win)/2)/geometry_dx)*geometry_dx
            crop_info=crop_scene(scene_dir,crop,origin,window_xy=win)
            if factor>1:
                with h5py.File(crop,'r+') as h:
                    vox=h['data'][:]
                    for axis in range(3): vox=np.repeat(vox,factor,axis=axis)
                    del h['data'];h.create_dataset('data',data=vox)
                    h.attrs['dx_dy_dz']=[g['dx']]*3
            tx=txs[i]-[*origin,0];rx=rxs[i]-[*origin,0]
            fn.write_text(input_text(g,crop,tx,rx,win,crop_info["material_database"]))
            with (run_dir/'last.log').open('w') as log, contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                gprMax.run(inputfile=str(fn),metal=(True if g.get("backend","cpu")=="metal" else None),hide_progress_bars=True,log_level=40)
            with h5py.File(fn.with_suffix('.h5')) as raw:
                a=raw['rxs/rx1/Ey'][:];dt=float(raw.attrs['dt'])
                if not np.isfinite(a).all() or not np.any(a): raise ValueError(f'Invalid signal at {i}')
                if 'Ey' not in out:
                    out.create_dataset('Ey',shape=(n,len(a)),dtype='f4',chunks=(1,len(a)),compression='gzip')
                    out.create_dataset('time_s',data=np.arange(len(a))*dt)
                    out.attrs.update(dt=dt,gprMax=raw.attrs['gprMax'])
                if len(a)!=out['Ey'].shape[1] or dt!=out.attrs['dt']: raise ValueError('Timing changed')
                out['Ey'][i]=a;out['seconds'][i]=time.monotonic()-start;out.flush()
                out['complete'][i]=True;out.flush()
            (run_dir/'progress.json').write_text(json.dumps({'completed':int(out['complete'][:].sum()),'total':n,'last_scan':int(i),'seconds':float(out['seconds'][i])}))
        return {'scene':scene_dir.name,'path':path_id,'done':int(out['complete'][:].sum()),'n':n}
