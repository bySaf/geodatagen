#!/usr/bin/env python3
"""Small fixed-domain 3D FDTD survey; resumable and separate from data/final."""
from pathlib import Path
import argparse
import contextlib
import copy
import hashlib
import json
import os
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
import h5py
import numpy as np
from src.soil import prepare_scene, sha256, custom_generator_fingerprint

OUT = ROOT/'user_review/realism/pilot'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def configuration(path, mode):
    cfg = json.loads(Path(path).read_text())
    cfg = copy.deepcopy(cfg)
    cfg['generator'] = mode
    if mode == 'custom_crim_archie':
        cfg['generator_fingerprint'] = custom_generator_fingerprint()
    cfg['script_sha256'] = sha256(__file__)
    g, p, s = cfg['grid'], cfg['profile'], cfg['source']
    dx = g['solver_dx']
    factor = round(g['dx']/dx)
    if factor < 1 or not np.isclose(factor*dx, g['dx']):
        raise ValueError('solver_dx must divide geometry dx')
    n = round((p['x_end_m']-p['x_start_m'])/p['step_m'])+1
    x = p['x_start_m']+np.arange(n)*p['step_m']
    if not np.isclose(x[-1], p['x_end_m']):
        raise ValueError('Profile endpoint must fit step')
    tx = np.c_[x, np.full(n,p['y_m']), np.full(n,g['surface_z']+s['height_m'])]
    rx = tx + [s['tx_rx_offset_m'], 0, 0]
    for coords in (tx,rx):
        if not np.allclose(coords/dx,np.rint(coords/dx)):
            raise ValueError('Tx/Rx must lie exactly on the solver grid')
        if np.any(coords < g['pml_thickness']) or np.any(coords > np.asarray(g['domain'])-g['pml_thickness']):
            raise ValueError('Tx/Rx intersects PML')
    cfg['profile']['n'] = n
    return cfg, tx, rx


def prepare(cfg, directory, with_object=True):
    directory.mkdir(parents=True,exist_ok=True)
    key = digest(cfg)
    stamp = directory/'config.json'
    if stamp.exists() and json.loads(stamp.read_text()) != cfg:
        raise ValueError('Pilot configuration changed; choose a new output folder')
    stamp.write_text(json.dumps(cfg,indent=2)+'\n')
    scene = directory/'scene'
    if not (scene/'soil.json').exists():
        target = cfg['target']
        xyz = ' '.join(str(v) for v in target['center_m'])
        objects = [f'#sphere: {xyz} {target["radius_m"]} pec n'] if with_object else []
        prepare_scene(scene,cfg['preset'],cfg['grid'],cfg['seed'],object_lines=objects,generator=cfg['generator'])
    meta = json.loads((scene/'soil.json').read_text())
    for item, checksum in [('geometry','geometry_sha256'),('materials','materials_sha256')]:
        if sha256(scene/meta[item]) != meta[checksum]:
            raise ValueError('Frozen pilot geometry/materials changed')
    run = directory/'run';run.mkdir(exist_ok=True)
    geometry = run/'geometry.h5'
    factor = round(cfg['grid']['dx']/cfg['grid']['solver_dx'])
    if not geometry.exists():
        with h5py.File(scene/meta['geometry']) as src,h5py.File(geometry,'w') as dst:
            a = src['data'][:]
            for axis in range(3):a=np.repeat(a,factor,axis=axis)
            dst['data']=a;dst['material_keys']=src['material_keys'][:]
            for name in ('MaterialDatabase','MaterialDatabaseSchemaVersion'):dst.attrs[name]=src.attrs[name]
            dst.attrs['dx_dy_dz']=[cfg['grid']['solver_dx']]*3
        shutil.copyfile(scene/meta['materials'],run/meta['materials'])
    return run, geometry, Path(meta['materials']).stem, key


def input_text(cfg, geometry, database, tx, rx):
    g,s=cfg['grid'],cfg['source'];dx=g['solver_dx']
    xyz=lambda values:' '.join(f'{float(v):.12g}' for v in values)
    return '\n'.join([
        '#title: fixed 3D park pilot - no moving crop',
        f'#domain: {xyz(g["domain"])}',f'#dx_dy_dz: {dx} {dx} {dx}',
        f'#time_window: {g["time_window"]}',f'#num_threads: {g["threads"]}',
        f'#pml_cells: {round(g["pml_thickness"]/dx)}',
        f'#geometry_objects_read: 0 0 0 {geometry.name} {database} n',
        f'#waveform: {s["waveform"]} {s["moment_Am"]/dx} {s["frequency_hz"]} pulse',
        f'#hertzian_dipole: y {xyz(tx)} pulse',f'#rx: {xyz(rx)}',''])


def source_values(cfg, dt, n):
    from gprMax.waveforms import Waveform
    w=Waveform();w.type=cfg['source']['waveform'];w.freq=cfg['source']['frequency_hz']
    # HertzianDipole evaluates current on the half-time lattice; retain it.
    w.amp=cfg['source']['moment_Am']/cfg['grid']['solver_dx']
    times=(np.arange(n)+.5)*dt
    values=np.asarray([w.calculate_value(float(t),dt) for t in times],dtype=np.float64)
    return times,values


def run_survey(cfg, txs, rxs, directory, limit=None, with_object=True):
    import gprMax
    run,geometry,database,key=prepare(cfg,directory,with_object)
    fn=run/'scan.in';n=len(txs);outpath=directory/'bscan.h5'
    with h5py.File(outpath,'a') as out:
        if 'complete' not in out:
            out.attrs.update(config_sha256=key,source='gprMax 3D FDTD; raw point Ey',units='V/m',
                             axis_order='scan,time',fixed_domain=True,with_object=with_object)
            out['tx_m']=txs;out['rx_m']=rxs;out['complete']=np.zeros(n,dtype=bool);out['seconds']=np.zeros(n)
        if out.attrs['config_sha256'] != key or not np.array_equal(out['tx_m'][:],txs):
            raise ValueError('Existing output configuration/coordinates differ')
        todo=np.flatnonzero(~out['complete'][:])
        if limit is not None:todo=todo[:limit]
        for i in todo:
            if (OUT/'STOP').exists():break
            start=time.monotonic();fn.write_text(input_text(cfg,geometry,database,txs[i],rxs[i]))
            with (run/'last.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                gprMax.run(inputfile=str(fn),hide_progress_bars=True,log_level=40)
            with h5py.File(fn.with_suffix('.h5')) as raw:
                a=raw['rxs/rx1/Ey'][:];dt=float(raw.attrs['dt'])
                if not np.isfinite(a).all() or not np.any(a):raise ValueError('Invalid FDTD field')
                if 'Ey' not in out:
                    out.create_dataset('Ey',shape=(n,len(a)),dtype='f4',chunks=(1,len(a)),compression='gzip')
                    out['time_s']=np.arange(len(a))*dt
                    st,sv=source_values(cfg,dt,len(a));out['source_time_s']=st;out['source_current_A']=sv
                    out.attrs.update(dt=dt,gprMax=raw.attrs['gprMax'],source_dl_m=cfg['grid']['solver_dx'])
                if len(a)!=out['Ey'].shape[1] or dt!=out.attrs['dt']:raise ValueError('Time grid changed')
                out['Ey'][i]=a;out['seconds'][i]=time.monotonic()-start;out.flush();out['complete'][i]=True;out.flush()
                progress=dict(mode=cfg['generator'],with_object=with_object,completed=int(out['complete'][:].sum()),
                              total=n,last_index=int(i),last_seconds=float(out['seconds'][i]),pid=os.getpid())
            (directory/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
            print(json.dumps(progress),flush=True)
    return outpath


def effective_sfcw(outpath, fmin=400e6, fmax=900e6, df=5e6):
    """Synthetic coherent sweep of the linear point-source transfer response.

    This is NOT measured S21 or a calibrated antenna. Fractional frequency steps
    interpolate the finite time record and add no independent measured data.
    """
    with h5py.File(outpath) as h:
        mask=h['complete'][:]
        if not mask.all():return
        y=h['Ey'][:].astype(float);t=h['time_s'][:];dt=float(h.attrs['dt'])
        source=h['source_current_A'][:]*float(h.attrs['source_dl_m']);st=h['source_time_s'][:]
        x=h['tx_m'][:,0]
    freq=np.arange(round(fmin/df),round(fmax/df)+1)*df
    # Mild tail taper reduces the artificial finite-record discontinuity.
    taper=np.ones(len(t));m=max(2,round(.05*len(t)));taper[-m:]=.5*(1+np.cos(np.linspace(0,np.pi,m)))
    phase=np.exp(-2j*np.pi*t[:,None]*freq[None,:])
    Y=(y*taper)@phase*dt
    S=source@np.exp(-2j*np.pi*st[:,None]*freq[None,:])*dt
    reg=(1e-3*np.max(np.abs(S)))**2
    H=Y*np.conj(S)[None,:]/(np.abs(S)[None,:]**2+reg)
    window=np.hanning(len(freq));nfft=4096
    spectrum=np.zeros((len(y),nfft//2+1),dtype=np.complex128)
    bins=np.rint(freq/df).astype(int);spectrum[:,bins]=H*window
    impulse=np.fft.irfft(spectrum,n=nfft,axis=1)*nfft*df
    it=np.arange(nfft)/(nfft*df)
    dest=outpath.parent/'effective_sfcw.npz'
    np.savez_compressed(dest,frequency_hz=freq,transfer=H,source_spectrum=S,band_window=window,
                        time_s=it,response=impulse,x_m=x,regularizer=reg)
    meta=dict(description='Synthetic effective coherent sweep of point-source transfer; not measured antenna S21',
              transfer_units='(V/m)/(A*m)',response_units='(V/m)/(A*m*s)',
              frequency_min_hz=fmin,frequency_max_hz=fmax,frequency_step_hz=df,n_frequencies=len(freq),
              finite_record_frequency_resolution_hz=1/(len(t)*dt),time_spacing_s=float(it[1]-it[0]),
              unambiguous_period_s=1/df,bandwidth_time_scale_s=1/(fmax-fmin),
              regularization_relative_amplitude=1e-3,min_source_band_amplitude_relative=float(np.min(abs(S))/np.max(abs(S))),
              source_half_step_time_axis_used=True,tail_taper_fraction=.05,
              processing='Direct DFT on specified frequencies, regularized deconvolution by actual injected current moment, Hann frequency window, Hermitian inverse FFT',
              limitations=['No antenna calibration, shielding, port response, electronics, gain or noise model.',
                           '5 MHz sampling oversamples a 32 ns finite record; it does not create independently resolved frequencies.',
                           'Zero padding changes plot sample density, not bandwidth-limited range resolution.'])
    (outpath.parent/'effective_sfcw.json').write_text(json.dumps(meta,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'user_review/realism/park-pilot.json')
    p.add_argument('--mode',choices=['native_peplinski','custom_crim_archie'],required=True)
    p.add_argument('--limit',type=int);p.add_argument('--background-center',action='store_true')
    p.add_argument('--postprocess-only',action='store_true')
    a=p.parse_args();cfg,tx,rx=configuration(a.config,a.mode)
    directory=OUT/a.mode
    if a.background_center:
        index=len(tx)//2;tx=tx[index:index+1];rx=rx[index:index+1];directory=directory/'background_center'
    if a.postprocess_only:out=directory/'bscan.h5'
    else:out=run_survey(cfg,tx,rx,directory,a.limit,not a.background_center)
    effective_sfcw(out)
    print('READY',out,flush=True)

if __name__=='__main__':main()
