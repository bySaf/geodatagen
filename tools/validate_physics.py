#!/usr/bin/env python3
"""Independent physics comparisons. Metrics are evidence, not accuracy certification."""
from pathlib import Path
import argparse, contextlib, copy, json, os, sys, time, re
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache'/'matplotlib'))
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.soil import prepare_scene,crop_scene
from src.runner import input_text


def calculate(scene_dir, name, output, grid, position, window=(1.2,1.2), refine=False):
    import gprMax
    folder=output/name;folder.mkdir(parents=True,exist_ok=True)
    origin=np.rint((position[:2]-np.array(window)/2)/grid['dx'])*grid['dx']
    geom=folder/'geometry.h5'
    info=crop_scene(scene_dir,geom,origin,window)
    g=copy.deepcopy(grid)
    factor=(2 if refine else 1) if isinstance(refine,bool) else int(refine)
    if factor>1:
        g['dx']/=factor
        with h5py.File(geom,'r+') as f:
            a=f['data'][:]
            for axis in range(3):a=np.repeat(a,factor,axis=axis)
            del f['data'];f.create_dataset('data',data=a)
            f.attrs['dx_dy_dz']=[g['dx']]*3
    tx=position-[*origin,0];rx=tx+[.04,0,0]
    fn=folder/'scan.in'
    text=input_text(g,geom,tx,rx,window,info['material_database'])
    moment=grid.get('source_moment',grid['dx'])
    text=re.sub(r'#waveform: ricker \S+ ',f'#waveform: ricker {moment/g["dx"]} ',text)
    text=re.sub(r'^#pml_cells:.*\n','',text,flags=re.M)
    text+=f'#pml_cells: {10*factor}\n'
    fn.write_text(text)
    start=time.monotonic()
    with (folder/'solver.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        gprMax.run(inputfile=str(fn),hide_progress_bars=True,log_level=40)
    with h5py.File(fn.with_suffix('.h5')) as f:
        a=f['rxs/rx1/Ey'][:]; t=np.arange(len(a))*f.attrs['dt']
    if not np.isfinite(a).all() or not np.any(a):raise ValueError('Invalid FDTD output')
    return dict(time=t,signal=a,seconds=time.monotonic()-start,dx=g['dx'],window=list(window),tx=position.tolist())


def compare(reference, alternative, late_start=6e-9):
    t=reference['time'];a=reference['signal'];b=np.interp(t,alternative['time'],alternative['signal'])
    metrics={}
    for name,mask in [('full',np.ones(len(t),bool)),('late',t>=late_start)]:
        x,y=a[mask],b[mask]
        metrics[name]=dict(relative_l2=float(np.linalg.norm(y-x)/np.linalg.norm(x)),correlation=float(np.corrcoef(x,y)[0,1]),reference_peak=float(np.max(abs(x))),alternative_peak=float(np.max(abs(y))))
    return metrics


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=ROOT/'data'/'demo');ap.add_argument('--extra-fine',action='store_true',help='also compare 1cm to 5mm; can take minutes');args=ap.parse_args()
    data=args.data.resolve();cfg=json.loads((data/'config.json').read_text());g=cfg['grid'];out=data/'validation';out.mkdir(exist_ok=True)
    scene=data/'park';metal=next(o for o in json.loads((scene/'objects.json').read_text()) if o['material']=='pec')
    pos=np.array([*metal['center_m'][:2],g['surface_z']+cfg['trajectory']['antenna_height']]);pos=np.rint(pos/g['dx'])*g['dx']
    preset=cfg['presets'][0];control=out/'park_without_objects';dry=out/'park_drier'
    if not (control/'soil.json').exists(): prepare_scene(control,preset,g,cfg['seed'])
    drypreset=copy.deepcopy(preset)
    for layer in drypreset['layers']:
        if 'water_min' in layer:
            layer['water_min']=max(.01,layer['water_min']-.07);layer['water_max']-=.07
    if not (dry/'soil.json').exists():prepare_scene(dry,drypreset,g,cfg['seed'])
    runs={}
    cases=[('target',scene,False,(1.2,1.2)),('repeat_target',scene,False,(1.2,1.2)),('control',control,False,(1.2,1.2)),('drier_control',dry,False,(1.2,1.2)),('larger_window',scene,False,(1.6,1.6)),('finer_grid',scene,True,(1.2,1.2))]
    if args.extra_fine:cases.append(('finest_grid',scene,4,(1.2,1.2)))
    for name,where,refine,win in cases:
        print('Calculating',name,flush=True)
        runs[name]=calculate(where,name,out,g,pos,win,refine)
    comparisons={
        'repeat_identical_setup':compare(runs['target'],runs['repeat_target']),
        'objects_vs_control':compare(runs['control'],runs['target']),
        'wet_vs_drier_control':compare(runs['drier_control'],runs['control']),
        'window_1p2_vs_1p6':compare(runs['larger_window'],runs['target']),
        'mesh_0p02_vs_0p01':compare(runs['finer_grid'],runs['target'])}
    if args.extra_fine:comparisons['mesh_0p01_vs_0p005']=compare(runs['finest_grid'],runs['finer_grid'])
    assert np.array_equal(runs['target']['signal'],runs['repeat_target']['signal'])
    assert comparisons['objects_vs_control']['full']['relative_l2']>1e-6
    assert comparisons['wet_vs_drier_control']['full']['relative_l2']>1e-6
    # This mesh comparison refines the SAME voxel geometry, not a new random scene.
    result=dict(status='comparisons_completed_not_a_convergence_certificate',position_m=pos.tolist(),late_start_ns=6,
      comparisons=comparisons,runs={k:{p:v for p,v in r.items() if p not in ('time','signal')} for k,r in runs.items()},
      interpretation='Relative L2 uses the named reference norm. No per-trace alignment or amplitude normalisation. Larger-window and finer-grid results are references. Both meshes preserve the same current moment I*dy by scaling source amplitude inversely with dx, and the same physical PML thickness. The finer-grid run upsamples the same frozen coarse voxels, isolating field-update resolution; it does not improve geometric detail. Window sensitivity includes additional surrounding scene. No arbitrary convergence threshold is asserted.')
    (out/'physics.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    pairs=[('Object effect','control','target'),('Moisture effect','drier_control','control'),('Window sensitivity','target','larger_window'),('Mesh sensitivity','target','finer_grid')]
    for ax,(title,a,b) in zip(axes.flat,pairs):
        for k in (a,b):
            mask=runs[k]['time']>=6e-9
            ax.plot(runs[k]['time'][mask]*1e9,runs[k]['signal'][mask],label=k,lw=1)
        ax.axvline(6,color='grey',ls=':',lw=.8);ax.set(title=title,xlabel='Time, ns',ylabel='Ey, V/m');ax.legend()
    fig.tight_layout();fig.savefig(out/'physics.png',dpi=160)
    print(json.dumps(comparisons,indent=2))

if __name__=='__main__':main()
