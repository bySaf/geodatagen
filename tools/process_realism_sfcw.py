#!/usr/bin/env python3
"""Reprocess a completed pilot using its saved sweep settings, without FDTD."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from run_realism_pilot import effective_sfcw,sha256
import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True,help='completed pilot mode or center-check directory')
    a=p.parse_args();folder=a.data.resolve()
    config_path=folder/'config.json';cfg=json.loads(config_path.read_text());s=cfg['sfcw']
    fmin,fmax,df=(float(s[k]) for k in ('fmin_hz','fmax_hz','df_hz'))
    if not all(np.isfinite(v) for v in (fmin,fmax,df)) or not 0<fmin<fmax or df<=0:
        raise ValueError('Require finite 0 < fmin < fmax and df > 0')
    if not all(np.isclose(v/df,round(v/df)) for v in (fmin,fmax)):
        raise ValueError('Band edges must be multiples of frequency step')
    if fmax/df>=2048:raise ValueError('Frequency grid exceeds the 4096-point reconstruction')
    raw=folder/'bscan.h5'
    with h5py.File(raw) as h:
        if not h['complete'][:].all():raise ValueError('All traces must be complete before processing')
        duration=(len(h['time_s']))*float(h.attrs['dt'])
        if 1/df<duration:raise ValueError('Sweep period is shorter than the original time record')
    effective_sfcw(raw,fmin=fmin,fmax=fmax,df=df)
    path=folder/'effective_sfcw.json';meta=json.loads(path.read_text())
    meta.update(processing_parameter_source=str(config_path),saved_sfcw_settings=s,
                processing_entrypoint_sha256=sha256(__file__),
                original_compute_script_sha256=cfg['script_sha256'],
                raw_output_sha256=sha256(raw),original_record_duration_s=duration,
                recommended_plot_time_max_s=min(30e-9,duration))
    path.write_text(json.dumps(meta,indent=2)+'\n')
    print(path,flush=True)

if __name__=='__main__':main()
