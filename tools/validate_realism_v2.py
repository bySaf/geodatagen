#!/usr/bin/env python3
"""Make a shared display grid and audit contrast/layers without modifying raw data.

This is format harmonisation, not calibration or additional FDTD simulation.
"""
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import h5py
import numpy as np
from scipy.signal import resample_poly
import scipy

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'user_review/realism'
MODES = [('native_peplinski','native_raw'), ('custom_crim_archie','fields_raw')]
EPS0 = 8.8541878128e-12
C = 299792458.0


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def uniform(a):
    delta=float(np.median(np.diff(a)))
    if not np.allclose(np.diff(a),delta,rtol=1e-8,atol=1e-18):
        raise ValueError('Input axis is not uniform')
    return delta


def time_grid(a,t,new_t):
    dt=uniform(t);new_dt=uniform(new_t)
    q=Fraction(dt/new_dt).limit_denominator(2000)
    # Symmetric FIR with automatic group-delay compensation; no picked-event shift.
    b=resample_poly(a,q.numerator,q.denominator,axis=1,
                    window=('kaiser',8.6),padtype='line')
    bt=t[0]+np.arange(b.shape[1])*dt*q.denominator/q.numerator
    if new_t[0]<t[0] or new_t[-1]>t[-1] or new_t[-1]>bt[-1]:
        raise ValueError('Common grid would extrapolate data')
    result=np.asarray([np.interp(new_t,bt,row) for row in b])
    return result,dict(source_dt_s=dt,output_dt_s=new_dt,
        polyphase_up=q.numerator,polyphase_down=q.denominator,
        polyphase_dt_s=dt*q.denominator/q.numerator,
        final_alignment='Linear interpolation onto exact 50 ps sample centres; no time-zero shift',
        interpolation_max_clock_adjustment_s=float(abs(dt*q.denominator/q.numerator-new_dt)*(len(new_t)-1)))


def contrast(a):
    dc=a-a.mean(axis=1,keepdims=True)
    mean=dc.mean(axis=0,keepdims=True);r=dc-mean
    before=float(np.percentile(abs(dc),99));after=float(np.percentile(abs(r),99))
    e=float(np.sum(dc*dc));er=float(np.sum(r*r));eb=float(a.shape[0]*np.sum(mean*mean))
    return dict(p99_before_background_removal=before,p99_after_background_removal=after,
                p99_display_gain_when_renormalised=before/after,
                energy_retained_fraction=er/e,common_component_energy_fraction=eb/e,
                orthogonal_energy_identity_relative_error=abs(e-er-eb)/e,
                protocol='Per-trace DC removal, then profile-mean subtraction; all values on shared 29 x 320 display grid. No AGC, peak alignment, deconvolution or temporal band-pass.')


def bandwidth(a,dt):
    dc=a-a.mean(axis=1,keepdims=True);res=dc-dc.mean(axis=0,keepdims=True)
    f=np.fft.rfftfreq(a.shape[1],dt)
    p=np.sum(abs(np.fft.rfft(res*np.hanning(a.shape[1]),axis=1))**2,axis=0)
    selected=(f>=400e6)&(f<=900e6)
    return dict(residual_windowed_spectral_peak_hz=float(f[np.argmax(p[1:])+1]),
                residual_energy_fraction_400_900MHz=float(p[selected].sum()/p.sum()),
                protocol='Hann time window, sum spectrum energy over traces; rectangular frequency-bin accounting, not an applied band-pass.')


def layers(mode):
    folder=DEST/'pilot'/mode/'scene';meta=json.loads((folder/'soil.json').read_text())
    db=json.loads((folder/meta['materials']).read_text())['materials']
    with h5py.File(folder/meta['geometry']) as h:
        ids=h['data'][:];keys=[k.decode() for k in h['material_keys'][:]]
    ep=np.empty(len(keys));sig=np.empty(len(keys));omega=2*np.pi*600e6
    for i,key in enumerate(keys):
        m=db[key];base=m['base'];ep[i]=base['relative_permittivity'];sig[i]=base['electric_conductivity_s_per_m']
        for pole in m.get('poles',[]):
            delta=pole['relative_permittivity_difference'];wt=omega*pole['relaxation_time_s']
            ep[i]+=delta/(1+wt*wt);sig[i]+=EPS0*omega*delta*wt/(1+wt*wt)
    epsilon=ep[ids];sigma=sig[ids];soil=np.isfinite(sigma)&(epsilon>1)
    depth=meta['surface_z']-(np.arange(ids.shape[2])+.5)*meta['dx']
    if mode=='custom_crim_archie':
        with np.load(folder/'custom_fields.npz',allow_pickle=False) as f:lid=f['layer_id'][:]
    else:
        lid=np.zeros(ids.shape,dtype=int);top=0
        for i,layer in enumerate(meta['layers']):
            bottom=top+layer['thickness'];lid[:,:, (depth>=top-1e-8)&(depth<bottom-1e-8)]=i+1;top=bottom
    records=[];previous=1+0j;top=0;travel=0
    for i,layer in enumerate(meta['layers']):
        mask=soil&(lid==i+1);values=epsilon[mask];sv=sigma[mask]
        mean=float(values.mean());sigma_mean=float(sv.mean());bottom=top+layer['thickness']
        er=mean-1j*sigma_mean/(omega*EPS0)
        coefficient=(np.sqrt(previous)-np.sqrt(er))/(np.sqrt(previous)+np.sqrt(er))
        travel+=2*layer['thickness']*np.sqrt(mean)/C
        records.append(dict(layer=i+1,nominal_depth_top_m=top,nominal_depth_bottom_m=bottom,
            epsilon_mean=mean,epsilon_std=float(values.std()),epsilon_min=float(values.min()),epsilon_max=float(values.max()),
            sigma_effective_mean_s_m=sigma_mean,
            illustrative_interface_reflection_amplitude_abs=float(abs(coefficient)),
            illustrative_interface_reflected_power_fraction=float(abs(coefficient)**2),
            normal_ray_delay_to_bottom_from_ground_surface_ns=float(travel*1e9),soil_voxels=int(mask.sum())))
        previous=er;top=bottom
    return dict(layers=records,interface_model='Normal incidence between homogeneous half-spaces formed from layer-average complex permittivity at 600 MHz; illustration of contrast only. No roughness, lateral variation, propagation loss, multiple reflections or antenna response.',
                travel_model='Normal-ray phase-speed estimate using mean epsilon; not exact arrival time, no instrument/source delay, no dispersive group velocity. Bottom of model is not a measured reflector.')


def main():
    x=np.arange(29)*.05;t=np.arange(320)*50e-12
    output={'x_m':x,'time_s':t};records={};inputs={}
    for mode,key in MODES:
        path=DEST/'pilot'/mode/'bscan.h5'
        with h5py.File(path) as h:
            if not h['complete'][:].all():raise ValueError('Incomplete raw data')
            a=h['Ey'][:].astype(float);old_t=h['time_s'][:];old_x=h['tx_m'][:,0]
        if not np.allclose(old_x-old_x[0],x,rtol=0,atol=1e-12):raise ValueError('Synthetic X grid differs')
        result,meta=time_grid(a,old_t,t);output[key]=result
        records[key]=dict(shape_original=list(a.shape),shape_output=list(result.shape),units='V/m',
            spatial_operation='None; original 5 cm positions retained, only common origin set to start of profile',
            temporal_operation=meta,contrast=contrast(result),bandwidth=bandwidth(result,50e-12))
        inputs[str(path.relative_to(ROOT))]=sha(path)
    path=DEST/'reference_radar/concrete_specimen_II.npz'
    with np.load(path,allow_pickle=False) as h:a=h['amplitude'][:].astype(float);old_t=h['time_s'][:];old_x=h['x_m'][:]
    dx=uniform(old_x)
    if not np.isclose(dx,.005,rtol=0,atol=1e-12):raise ValueError('Unexpected real trace spacing')
    # Filter the complete measured profile before cropping to the shared 1.4 m view.
    spatial=resample_poly(a,1,10,axis=0,window=('kaiser',8.6),padtype='line')
    new_x=old_x[0]+np.arange(len(spatial))*dx*10
    if not np.allclose(new_x[:29]-new_x[0],x,rtol=0,atol=1e-12):raise ValueError('Real output X grid differs')
    result,meta=time_grid(spatial[:29],old_t,t);output['real_raw']=result
    residual=a-a.mean(axis=0,keepdims=True)
    spatial_power=np.sum(abs(np.fft.rfft(residual*np.hanning(len(a))[:,None],axis=0))**2,axis=1)
    k=np.fft.rfftfreq(len(a),dx)
    high=float(spatial_power[k>10].sum()/spatial_power.sum())
    records['real_raw']=dict(reference_specimen='II',reference_doi='10.5281/zenodo.10962520',shape_original=list(a.shape),shape_output=list(result.shape),units='Instrument export units, not calibrated V/m',
        spatial_operation=dict(source_step_m=dx,output_step_m=.05,up=1,down=10,filter_before_crop=True,
            window=['kaiser',8.6],padtype='line',original_traces=len(a),resampled_full_traces=len(spatial),retained_first_traces=29,
            original_residual_spatial_energy_above_new_nyquist_fraction=high,
            spatial_energy_protocol='Remove mean across profile, Hann along original full X, sum spatial spectrum energy over time; new spatial Nyquist is 10 cycles/m.'),
        temporal_operation=meta,contrast=contrast(result),bandwidth=bandwidth(result,50e-12))
    inputs[str(path.relative_to(ROOT))]=sha(path)
    for key in ('native_raw','fields_raw','real_raw'):
        if output[key].shape!=(29,320) or not np.isfinite(output[key]).all():raise ValueError('Invalid common array')
    np.savez_compressed(DEST/'standardized_comparison.npz',**output)
    report=dict(script_sha256=sha(Path(__file__)),scipy_version=scipy.__version__,common_grid=dict(
        traces=29,time_samples=320,trace_step_m=.05,time_step_s=50e-12,x_range_m=[0,1.4],time_range_ns=[0,15.95],
        time_interval_description='[0,16 ns), half-open; last centre 15.95 ns; no extrapolated real sample at 16 ns'),
        arrays=records,layer_contrast={mode:layers(mode) for mode,_ in MODES},inputs_sha256=inputs,
        npz_sha256=sha(DEST/'standardized_comparison.npz'),filter=dict(window=['kaiser',8.6],padtype='line',
        implementation='scipy.signal.resample_poly: zero-phase low-pass polyphase FIR with delay compensation; default filter length from installed SciPy',
        documentation='https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html'),
        limitations=[
            'Shared sample spacing and plotting units do not make material, target depth, source or antenna response identical.',
            'No peak alignment, time scaling, source deconvolution, temporal band-pass, AGC or independent trace normalisation was applied.',
            'Both origins remain first exported sample. Real instrument latency is unknown; no conversion to depth.',
            'Anti-alias filtering intentionally removes fine spatial detail from the real instrument. Keep its original-resolution image alongside.',
            'Line padding is an explicit boundary assumption for the finite record; features at profile/time edges should not be used as validation targets.',
            'Physical validation available now: object/no-object paired control, PML sensitivity, material/geometry integrity. Instrument similarity requires matched measured scene and calibration; no similarity percentage is produced.'])
    (DEST/'validation_v2.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(DEST/'standardized_comparison.npz'),'contrast':{k:v['contrast'] for k,v in records.items()},'real_spatial_energy_above_new_nyquist_fraction':high},indent=2))


if __name__=='__main__':main()
