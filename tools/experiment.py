#!/usr/bin/env python3
"""Create a separate, editable park experiment without changing the saved pilot."""
from pathlib import Path
import argparse
import contextlib
import fcntl
import json
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
import h5py
import numpy as np
import run_realism_pilot as pilot
from src.layout import load_object_coordinates, object_commands
from src.soil import GENERATORS, prepare_scene, sha256


def destination(path):
    path=Path(path).resolve()
    protected=[ROOT/'data/final', ROOT/'user_review/realism/pilot']
    if any(path == p or path.is_relative_to(p) or p.is_relative_to(path) for p in protected):
        raise ValueError('Choose a NEW experiment directory, outside the completed datasets')
    return path


def settings(path, mode):
    raw=json.loads(Path(path).read_text())
    g,p,s=raw['grid'],raw['profile'],raw['source']
    numbers=[*g['domain'],g['dx'],g['solver_dx'],g['time_window'],g['pml_thickness'],
             p['step_m'],s['frequency_hz'],s['moment_Am']]
    if not all(np.isfinite(v) and v>0 for v in numbers):
        raise ValueError('Dimensions, spacing, duration, frequency and source moment must be positive')
    if not 0<g['surface_z']<g['domain'][2] or s['height_m']<0:
        raise ValueError('Surface must be inside the domain; source height must be nonnegative')
    if p['x_end_m']<p['x_start_m']:
        raise ValueError('Profile end must be at or after its start')
    for value in [*g['domain'],g['surface_z']]:
        if not np.isclose(value/g['dx'],round(value/g['dx'])):
            raise ValueError('Domain and soil surface must align with the geometry grid')
    if not np.isclose(g['pml_thickness']/g['solver_dx'],round(g['pml_thickness']/g['solver_dx'])):
        raise ValueError('PML thickness must align with the solver grid')
    if not np.isclose(sum(v['thickness'] for v in raw['preset']['layers']),g['surface_z']):
        raise ValueError('Layer thicknesses must add up to surface_z')
    sweep=raw['sfcw'];lo,hi,step=[float(sweep[k]) for k in ('fmin_hz','fmax_hz','df_hz')]
    if not all(np.isfinite(v) for v in (lo,hi,step)) or not 0<lo<hi or step<=0:
        raise ValueError('Require 0 < sweep fmin < fmax and df > 0')
    if not all(np.isclose(v/step,round(v/step)) for v in (lo,hi)) or hi/step>=2048:
        raise ValueError('Sweep edges must be multiples of df and fit the 4096-point spectrum')
    if 1/step<g['time_window']:
        raise ValueError('Sweep period must cover the FDTD record')
    cfg,tx,rx=pilot.configuration(path,mode)
    objects=raw.get('objects')
    if objects is None:
        target=raw['target']
        objects=[dict(type='metal',center_m=target['center_m'],radius_m=target['radius_m'])]
    with tempfile.TemporaryDirectory(prefix='gpr-objects-') as tmp:
        f=Path(tmp)/'objects.json';f.write_text(json.dumps({'park':objects}))
        checked=load_object_coordinates(f,['park'],g['domain'],g['surface_z'])['park']
    for item in checked:
        c=np.array(item['center_m']);r=item['radius_m'];pad=g['pml_thickness']
        if r<g['dx'] or np.any(c-r<pad) or np.any(c+r>np.array(g['domain'])-pad):
            raise ValueError('Each sphere must be at least one geometry cell in radius and outside PML')
    cfg['experiment_objects']=checked
    cfg['experiment_entrypoint_sha256']=sha256(__file__)
    return cfg,tx,rx


@contextlib.contextmanager
def locked(directory):
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.experiment.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This experiment is running; use status or wait')
        yield


def prepare(directory,cfg,tx,rx):
    stamp=directory/'config.json'
    if stamp.exists():
        if json.loads(stamp.read_text())!=cfg:
            raise ValueError('Saved settings or code differ. Choose a NEW --data directory')
    else:
        if any(p.name!='.experiment.lock' for p in directory.iterdir()):
            raise ValueError('Output directory is not empty and has no experiment config')
        stamp.write_text(json.dumps(cfg,indent=2)+'\n')
    scene=directory/'scene'
    if not (scene/'soil.json').exists():
        lines=['#material: 4 0.001 1 0 inclusion']
        lines += [command+' n' for command in object_commands(cfg['experiment_objects'])]
        prepare_scene(scene,cfg['preset'],cfg['grid'],cfg['seed'],object_lines=lines,generator=cfg['generator'])
    # The protected pilot sees an already prepared scene and only validates it /
    # builds the solver-resolution copy. Its implementation is not modified.
    pilot.prepare(cfg,directory,with_object=bool(cfg['experiment_objects']))
    (directory/'objects.json').write_text(json.dumps(cfg['experiment_objects'],indent=2)+'\n')
    np.savez_compressed(directory/'positions.npz',tx_m=tx,rx_m=rx)
    layout_plot(directory,cfg,tx,rx)


def layout_plot(directory,cfg,tx,rx):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    g=cfg['grid'];fig,axs=plt.subplots(2,1,figsize=(10,7),layout='constrained')
    axs[0].plot(tx[:,0],tx[:,1],'.-',label='Tx');axs[0].plot(rx[:,0],rx[:,1],'.--',label='Rx')
    axs[1].plot(tx[:,0],g['surface_z']-tx[:,2],'.',label='Tx')
    d=0
    for layer in cfg['preset']['layers']:
        d+=layer['thickness'];axs[1].axhline(d,color='.6',ls='--',lw=.8)
    colors={'pec':'black','free_space':'white','inclusion':'#a6a6a6'}
    for obj in cfg['experiment_objects']:
        x,y,z=obj['center_m'];r=obj['radius_m'];color=colors[obj['material']]
        for ax,center in zip(axs,[(x,y),(x,g['surface_z']-z)]):
            ax.add_patch(Circle(center,r,facecolor=color,edgecolor='black'))
    axs[0].set(xlim=(0,g['domain'][0]),ylim=(0,g['domain'][1]),xlabel='X, м',ylabel='Y, м',title='План: маршрут и проекции объектов',aspect='equal')
    axs[0].legend()
    axs[1].set(xlim=(0,g['domain'][0]),ylim=(g['surface_z'],-max(.1,cfg['source']['height_m']*2)),xlabel='X, м',ylabel='Глубина, м',title='Проекции объектов по глубине; пунктир — номинальные границы',aspect='equal')
    fig.savefig(directory/'scene_layout.png',dpi=150);plt.close(fig)


def plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with h5py.File(directory/'bscan.h5') as h:
        done=h['complete'][:]
        if not done.any():raise ValueError('No completed traces to plot')
        a=h['Ey'][:].astype(float);t=h['time_s'][:]*1e9;x=h['tx_m'][:,0]
    a[~done]=np.nan
    if not np.isfinite(a[done]).all():raise ValueError('Nonfinite completed signal')
    cfg=json.loads((directory/'config.json').read_text())
    dx=cfg['profile']['step_m'];dt=t[1]-t[0];x=x-x[0]
    bg=a-np.nanmean(a,axis=0,keepdims=True)
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    cmap=plt.get_cmap('gray').copy();cmap.set_bad('#b7bdc4')
    for ax,values,title in zip(axs,[a,bg],['Исходное поле Ey','Минус средняя готовая трасса']):
        limit=float(np.nanpercentile(abs(values),99)) or 1
        im=ax.imshow(values.T/limit,aspect='auto',cmap=cmap,vmin=-1,vmax=1,interpolation='nearest',extent=[x[0]-dx/2,x[-1]+dx/2,t[-1]+dt/2,t[0]-dt/2])
        ax.set(title=title,xlabel='Путь Tx от начала, м',ylabel='Время, нс')
        fig.colorbar(im,ax=ax,label='Амплитуда / P99 своей панели')
    if done.sum()==1:axs[1].text(.5,.5,'Для сравнения нужны хотя бы 2 трассы',transform=axs[1].transAxes,ha='center',bbox=dict(facecolor='white'))
    fig.suptitle(f'Готово {done.sum()} из {len(done)} трасс; серые области ещё не рассчитаны')
    fig.savefig(directory/'bscan.png',dpi=160);plt.close(fig)
    i=np.flatnonzero(done)[-1];fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    ax.plot(t,a[i],color='black');ax.set(xlabel='Время, нс',ylabel='Ey, В/м',title=f'A-скан №{i}: путь Tx {x[i]:.2f} м')
    ax.grid(alpha=.2);fig.savefig(directory/'ascan.png',dpi=160);plt.close(fig)


def status(directory):
    try:
        with locked(directory):
            cfg=json.loads((directory/'config.json').read_text());total=cfg['profile']['n'];done=0
            if (directory/'bscan.h5').exists():
                with h5py.File(directory/'bscan.h5') as h:done=int(h['complete'][:].sum())
            return dict(running=False,completed=done,total=total,stopped_by_file=(directory/'STOP').exists())
    except RuntimeError:
        try:p=json.loads((directory/'progress.json').read_text())
        except (OSError,ValueError):p=dict(note='First trace running or progress is being updated')
        return dict(p,running=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','status','plot'])
    parser.add_argument('--data',required=True,type=Path,help='new experiment directory; one soil mode per directory')
    parser.add_argument('--config',type=Path,help='prepare only: configuration JSON')
    parser.add_argument('--mode',choices=GENERATORS,help='prepare only: soil generator')
    parser.add_argument('--limit',type=int,help='run only: maximum number of NEW traces')
    args=parser.parse_args();directory=destination(args.data)
    if args.command=='prepare' and (args.config is None or args.mode is None):parser.error('prepare requires --config and --mode')
    if args.command!='prepare' and (args.config is not None or args.mode is not None):parser.error('run/status/plot use the saved config; do not pass --config/--mode')
    if args.limit is not None and (args.command!='run' or args.limit<1):parser.error('--limit must be positive and is only for run')
    if args.command!='prepare' and not (directory/'config.json').exists():
        parser.error('Prepare this experiment first; config.json was not found')
    if args.command=='status':
        print(json.dumps(status(directory),indent=2));return
    with locked(directory):
        if args.command=='plot':
            plot(directory);print(directory/'bscan.png');return
        path=args.config if args.command=='prepare' else directory/'config.json'
        mode=args.mode if args.command=='prepare' else json.loads(path.read_text())['generator']
        cfg,tx,rx=settings(path,mode)
        prepare(directory,cfg,tx,rx)
        if args.command=='run':
            pilot.OUT=directory  # STOP belongs to this new experiment only.
            out=pilot.run_survey(cfg,tx,rx,directory,limit=args.limit,with_object=bool(cfg['experiment_objects']))
            s=cfg['sfcw'];pilot.effective_sfcw(out,fmin=s['fmin_hz'],fmax=s['fmax_hz'],df=s['df_hz'])
        print('READY',directory,flush=True)


if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,FileNotFoundError) as exc:raise SystemExit(str(exc))
