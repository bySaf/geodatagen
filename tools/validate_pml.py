#!/usr/bin/env python3
"""Compare half PML thickness to the stored .2m validation runs, without tuning."""
from pathlib import Path
import contextlib,json,os,re,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import gprMax,h5py,numpy as np
from validate_physics import compare
out=ROOT/'data/final/validation';result={}
cases={'park':ROOT/'data/demo/validation/finer_grid',
       'field':out/'field_window_1.2','road':out/'road_window_1.2'}
for name,ref in cases.items():
 folder=out/f'{name}_pml_0.1';folder.mkdir(exist_ok=True)
 text=(ref/'scan.in').read_text();text=re.sub(r'^#pml_cells:.*$', '#pml_cells: 10',text,flags=re.M)
 fn=folder/'scan.in';fn.write_text(text)
 print(name,flush=True);start=time.monotonic()
 with (folder/'solver.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
  gprMax.run(inputfile=str(fn),hide_progress_bars=True,log_level=40)
 seconds=time.monotonic()-start
 runs=[]
 for p in (ref,folder):
  with h5py.File(p/'scan.h5') as f:
   a=f['rxs/rx1/Ey'][:];t=np.arange(len(a))*f.attrs['dt']
   assert np.isfinite(a).all() and np.any(a)
   runs.append(dict(signal=a,time=t))
 assert np.array_equal(runs[0]['time'],runs[1]['time'])
 metrics=compare(*runs)
 for part in ('full','late'):
  m=metrics[part];m['relative_absolute_peak_difference']=abs(m['alternative_peak']-m['reference_peak'])/m['reference_peak']
 result[name]=dict(comparison=metrics,seconds=seconds,reference=str(ref.relative_to(ROOT)),input=str(fn.relative_to(ROOT)))
 (out/'pml.json').write_text(json.dumps(dict(status='partial' if len(result)<3 else 'completed',reference_pml_m=.2,alternative_pml_m=.1,solver_dx_m=.01,late_start_ns=6,results=result,
 scope='One representative point in each scene, exact static voxel inputs, same current moment, positions and time axis. Timings include competing dataset workers.'),indent=2))
 print(json.dumps(result[name]),flush=True)
