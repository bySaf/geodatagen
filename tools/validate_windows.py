#!/usr/bin/env python3
"""Representative field/road window checks; no global sufficiency claim."""
from pathlib import Path
import argparse,json
import numpy as np
from validate_physics import ROOT,calculate,compare
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data'/'final');a=p.parse_args()
 data=a.data.resolve();cfg=json.loads((data/'config.json').read_text());g=cfg['grid'];out=data/'validation';out.mkdir(exist_ok=True)
 fig,axes=plt.subplots(1,2,figsize=(12,4));result={}
 for ax,name in zip(axes,['field','road']):
  scene=data/name
  if name=='road':
   obj=next(o for o in json.loads((scene/'objects.json').read_text()) if o['material']=='pec')
   pos=np.r_[obj['center_m'][:2],g['surface_z']+cfg['trajectory']['antenna_height']]
   pos=np.rint(pos/g['dx'])*g['dx']
  else:
   with np.load(scene/'path_0.npz') as path:pos=path['tx'][len(path['tx'])//2]
  runs={}
  for width in (1.2,1.6):
   key=f'{name}_window_{width}'
   print(key,flush=True)
   runs[width]=calculate(scene,key,out,g,pos,(width,width),refine=True)
  assert np.array_equal(runs[1.2]['time'],runs[1.6]['time'])
  metrics=compare(runs[1.6],runs[1.2])
  result[name]=dict(position_m=pos.tolist(),receiver_m=(pos+[.04,0,0]).tolist(),dt_s=float(runs[1.2]['time'][1]),
   solver_dx_m=runs[1.2]['dx'],current_moment_A_m=g['source_moment'],PML_thickness_m=g['pml_thickness'],
   comparison=metrics,timings_seconds={str(w):r['seconds'] for w,r in runs.items()},
   sample_extrema={str(w):[float(r['signal'].min()),float(r['signal'].max())] for w,r in runs.items()})
  for w,r in runs.items():
   mask=r['time']>=6e-9;ax.plot(r['time'][mask]*1e9,r['signal'][mask],label=f'{w} m',lw=1)
  ax.set(title=name,xlabel='Time, ns',ylabel='Ey, V/m');ax.legend()
 fig.tight_layout();fig.savefig(out/'windows.png',dpi=160)
 (out/'windows.json').write_text(json.dumps(dict(reference='1.6m window',late_start_ns=6,results=result,
  scope='One selected location per scene; no claim of uniform error bounds for entire paths. CPU2 threads per solver, concurrent dataset generation; timings are not standalone benchmarks.'),indent=2))
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()
