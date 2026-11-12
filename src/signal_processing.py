"""Coherent frequency processing of saved point-source FDTD records."""
from pathlib import Path
import json
import h5py
import numpy as np

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
    # Last 5% cosine taper and 1e-3 Tikhonov scale are processing choices,
    # not physical antenna calibration; record both in exported metadata.
    taper=np.ones(len(t));m=max(2,round(.05*len(t)));taper[-m:]=.5*(1+np.cos(np.linspace(0,np.pi,m)))
    phase=np.exp(-2j*np.pi*t[:,None]*freq[None,:])
    Y=(y*taper)@phase*dt
    S=source@np.exp(-2j*np.pi*st[:,None]*freq[None,:])*dt
    reg=(1e-3*np.max(np.abs(S)))**2
    H=Y*np.conj(S)[None,:]/(np.abs(S)[None,:]**2+reg)
    window=np.hanning(len(freq));nfft=4096  # display/interpolation size, not new physical resolution
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
