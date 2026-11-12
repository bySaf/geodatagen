#!/usr/bin/env python3
"""prepare -> run -> report; all paths relative to this project by default."""
from pathlib import Path
import argparse, json, os
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache'/'matplotlib'))
import numpy as np
import h5py
from src.layout import generate_objects,load_object_coordinates,object_commands,make_trajectories,write_layout,overlap_matrix
from src.soil import prepare_scene,sha256,GENERATORS,custom_generator_fingerprint
from src.runner import signature,simulate_path


def prepare(data,config,objects_coordinates=None):
    data=Path(data)
    spec=ROOT/'objects.txt'
    cfg=dict(config,object_spec_sha256=sha256(spec))
    modes=[p.get('generator',cfg.get('soil_generator','native_peplinski')) for p in cfg['presets']]
    if any(mode not in GENERATORS for mode in modes):
        raise ValueError(f'Soil generator must be one of {GENERATORS}')
    if 'custom_crim_archie' in modes:
        cfg['soil_generator_fingerprint']=custom_generator_fingerprint()
    explicit=None
    if objects_coordinates is not None:
        objects_coordinates=Path(objects_coordinates).resolve()
        explicit=load_object_coordinates(objects_coordinates,[p['name'] for p in cfg['presets']],
            cfg['grid']['domain'],cfg['grid']['surface_z'],
            control_scenes=[p['name'] for p in cfg['presets'] if p.get('control',False)])
        cfg.update(object_coordinates_sha256=sha256(objects_coordinates),object_placement='explicit')
    if (data/'config.json').exists() and json.loads((data/'config.json').read_text())!=cfg:
        raise ValueError('Existing dataset configuration differs. Select a new --data directory.')
    data.mkdir(parents=True,exist_ok=True)
    (data/'config.json').write_text(json.dumps(cfg,indent=2))
    (data/'objects.txt').write_text(spec.read_text())
    if objects_coordinates is not None:
        (data/'objects-coordinates.json').write_bytes(objects_coordinates.read_bytes())
    for k,preset in enumerate(cfg['presets']):
        out=data/preset['name']; out.mkdir(exist_ok=True);seed=cfg['seed']+k*100
        objects=explicit[preset['name']] if explicit is not None else generate_objects(spec,seed,cfg['grid']['domain'],cfg['grid']['surface_z'],control=preset.get('control',False))
        paths=make_trajectories(cfg['grid']['domain'],cfg['grid']['surface_z'],cfg['grid']['dx'],**cfg['trajectory'])
        if (out/'soil.json').exists():
            if json.loads((out/'objects.json').read_text()) != objects:
                raise ValueError('Object generator changed: select a new dataset directory.')
            for i,p in enumerate(paths):
                with np.load(out/f'path_{i}.npz') as saved:
                    for key,source in [('tx','snapped_tx'),('rx','snapped_rx'),('requested_tx','requested_tx'),('requested_rx','requested_rx')]:
                        if not np.array_equal(saved[key],p[source]):
                            raise ValueError('Trajectory generator changed: select a new dataset directory.')
            print('Already prepared',preset['name'],flush=True)
            continue
        write_layout(out,objects,paths,overlap_radius=cfg['overlap_radius'])
        for i,p in enumerate(paths):
            np.savez_compressed(out/f'path_{i}.npz',tx=p['snapped_tx'],rx=p['snapped_rx'],requested_tx=p['requested_tx'],requested_rx=p['requested_rx'])
        (out/'overlap.json').write_text(json.dumps({'radius_m':cfg['overlap_radius'],'matrix':np.asarray(overlap_matrix(paths,cfg['overlap_radius'])).tolist()},indent=2))
        if not (out/'soil.json').exists():
            prepare_scene(out,preset,cfg['grid'],seed,object_lines=['#material: 4 0.001 1 0 inclusion']+object_commands(objects),
                          generator=preset.get('generator',cfg.get('soil_generator')))
        print('Prepared',preset['name'],flush=True)
    return cfg


def status(data,cfg):
    report=[]
    for p in cfg['presets']:
        for i in range(3):
            f=data/p['name']/f'bscan_{i}.h5';done=0;seconds=[]
            if f.exists():
                try:
                    with h5py.File(f) as h:
                        mask=h['complete'][:];done=int(mask.sum());seconds=h['seconds'][:][mask].tolist()
                except OSError:
                    progress=data/p['name']/'runs'/f'path_{i}'/'progress.json'
                    if progress.exists():done=json.loads(progress.read_text())['completed']
            report.append(dict(scene=p['name'],path=i,done=done,total=cfg['trajectory']['n'],median_seconds=float(np.median(seconds)) if seconds else None))
    print(json.dumps(report,indent=2));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','report','status'])
    p.add_argument('--config',type=Path,default=ROOT/'project.json');p.add_argument('--data',type=Path,default=ROOT/'data'/'final')
    p.add_argument('--workers',type=int,default=3);p.add_argument('--limit',type=int,help='maximum NEW scans per trajectory this invocation')
    p.add_argument('--objects-coordinates',type=Path,help='prepare only: explicit object coordinates by scene, in JSON')
    p.add_argument('--soil-generator',choices=GENERATORS,help='prepare only: default soil generator (per-preset generator takes precedence)')
    a=p.parse_args();data=a.data.resolve()
    if a.command!='prepare' and (a.objects_coordinates is not None or a.soil_generator is not None):
        p.error('--objects-coordinates and --soil-generator are only valid with prepare; run uses the frozen dataset configuration')
    if a.command in ['prepare','run']:
        import fcntl
        data.mkdir(parents=True,exist_ok=True)
        lock=(data/'.run.lock').open('w')
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('A calculation or preparation is already running in this dataset.')
    if a.command=='prepare':
        config=json.loads(a.config.read_text())
        if a.soil_generator is not None:config['soil_generator']=a.soil_generator
        prepare(data,config,a.objects_coordinates);return
    cfg=json.loads((data/'config.json').read_text())
    if a.command=='status':status(data,cfg);return
    if a.command=='report':
        from src.report import report
        report(data,cfg);return
    from concurrent.futures import ProcessPoolExecutor,as_completed
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(simulate_path,data/p['name'],i,cfg,a.limit) for p in cfg['presets'] for i in range(3)]
        for j in as_completed(jobs): print(j.result(),flush=True)
    status(data,cfg)

if __name__=='__main__':main()
