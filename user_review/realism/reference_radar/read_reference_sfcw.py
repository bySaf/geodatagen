"""Reproduce measured SFCW reference arrays/plots; no fabricated complex phase.

Run using project's .venv/bin/python. Add --include-soil only to explicitly
read the SHA256-verified SAL dataset through the restricted deserializer.
Pandas is only needed for this option, installed into local _deps for isolation.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import pickle
import struct
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def filter_band(a, dt, low=400e6, high=900e6):
    """Analysis-only band filter: cosine 50 MHz edges, zero padding, no rescale."""
    n = 2 ** int(np.ceil(np.log2(a.shape[-1] * 4)))
    f = np.fft.rfftfreq(n, dt)
    w = np.zeros_like(f)
    inside = (f >= low) & (f <= high)
    w[inside] = 1
    edge = 50e6
    k = inside & (f < low + edge)
    w[k] = .5 - .5 * np.cos(np.pi * (f[k] - low) / edge)
    k = inside & (f > high - edge)
    w[k] = .5 - .5 * np.cos(np.pi * (high - f[k]) / edge)
    out = np.fft.irfft(np.fft.rfft(a, n=n, axis=1) * w, n=n, axis=1)
    return out[:, :a.shape[1]].astype(np.float32)


def draw_bscan(ax, a, time_s, x_m, title):
    # Display-only profile background subtraction: preserves saved original array.
    shown = a - np.mean(a, axis=0, keepdims=True)
    clip = np.percentile(np.abs(shown), 99)
    ax.imshow(shown.T, origin='upper', extent=[x_m[0], x_m[-1], time_s[-1]*1e9, 0],
              aspect='auto', cmap='gray', vmin=-clip, vmax=clip, interpolation='nearest')
    ax.set(title=title, xlabel='Расстояние по профилю, м', ylabel='Время в экспорте прибора, нс')


def concrete():
    records = []
    names = ['I', 'II', 'III']
    files = sorted((BASE/'raw/SFCW').rglob('*.sgy'), key=lambda p: len(p.parent.name.split('Specimen ')[1].split()[0]))
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
    for name, path, ax0, ax1 in zip(names, files, axs[0], axs[1]):
        raw = path.read_bytes()
        ns = struct.unpack_from('>H', raw, 3220)[0]
        format_code = struct.unpack_from('>H', raw, 3224)[0]
        if format_code != 5:
            raise ValueError(f'Expected IEEE float32 SEG-Y, found code {format_code}')
        record_bytes = 240 + 4 * ns
        if (len(raw) - 3600) % record_bytes:
            raise ValueError('SEG-Y record length is inconsistent')
        nt = (len(raw) - 3600) // record_bytes
        a = np.stack([np.frombuffer(raw, '>f4', count=ns, offset=3600+j*record_bytes+240)
                      for j in range(nt)]).astype(np.float32)
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite SEG-Y amplitudes')
        # Export CSV is authoritative: 40.96 GHz, 2 scans/cm. SEG-Y dt=24 is
        # rounded/nonstandard GPR time metadata and MUST NOT be read as 24 us.
        csv_text = path.with_suffix('.csv').read_text(encoding='utf-16')
        def csv_value(key):
            for line in csv_text.splitlines():
                parts = line.split('\t')
                if parts[0] == key:
                    return float(parts[1])
            raise ValueError(f'Missing export field {key}')
        fs = csv_value('Sampling Rate [GHz]') * 1e9
        dx = .01 / csv_value('Repetition Rate [scans/cm]')
        dt = 1/fs
        t = np.arange(ns)*dt
        x = np.arange(nt)*dx
        filtered = filter_band(a, dt)
        np.savez_compressed(BASE/f'concrete_specimen_{name}.npz',
            amplitude=a, amplitude_400_900MHz=filtered, time_s=t, x_m=x,
            axis_order=np.array('trace,time'), amplitude_units=np.array('instrument export units; unspecified'),
            acquisition_type=np.array('measured SFCW time-domain instrument export'),
            native_band_hz=np.array([400e6, 6000e6]))
        power = np.mean(np.abs(np.fft.rfft(a-a.mean(axis=0), axis=1))**2, axis=0)
        freq = np.fft.rfftfreq(ns, dt)
        fraction = float(power[(freq >= 400e6)&(freq <= 900e6)].sum()/power.sum())
        rec = dict(specimen=name, file=str(path.relative_to(BASE)), sha256=sha256(path),
            trace_count=nt, time_samples=ns, sampling_rate_hz=fs, dt_s=dt,
            nominal_trace_spacing_m=dx, inferred_last_x_m=float(x[-1]),
            spectrum_peak_hz=float(freq[np.argmax(power)]),
            background_subtracted_energy_fraction_400_900MHz=fraction,
            position_note='SEG-Y coordinates are zero; x is inferred from nominal 2 scans/cm. All traces retained.',
            count_note='CSV reports 1.25 m (I), 1.85 m (II/III); inferred endpoints I/II exceed these by 5 mm.',
            sample_note='Actual export:655 samples. Article reportedly states653; actual export retained.',
            time_zero_note='Time0 is the first exported sample, not a calibrated surface/reflection zero. No shift or depth conversion was applied.')
        records.append(rec)
        draw_bscan(ax0, a, t, x, f'Образец {name}: экспорт SFCW 0.4–6 ГГц')
        draw_bscan(ax1, filtered, t, x, f'Образец {name}: анализ в полосе 0.4–0.9 ГГц')
    fig.suptitle('Реальные измерения Proceq GP8800: арматура в бетоне\nПоказ: вычитание среднего профиля, клиппинг 99%; без изменения осей и частот', fontsize=13)
    fig.savefig(BASE/'reference_bscans_all.png'); plt.close(fig)
    # Compact panel for direct comparison. Same data and honest axes.
    d=np.load(BASE/'concrete_specimen_III.npz')
    fig,axs=plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
    draw_bscan(axs[0],d['amplitude'],d['time_s'],d['x_m'],'Реальный SFCW: арматура в бетоне\nИсходный экспорт прибора 0.4–6 ГГц')
    draw_bscan(axs[1],d['amplitude_400_900MHz'],d['time_s'],d['x_m'],'Тот же экспорт: фильтр 0.4–0.9 ГГц\nНе новая съёмка в этой полосе')
    fig.suptitle('Показ: вычтен средний профиль, клиппинг 99%; t=0 — начало экспорта, не поверхность',fontsize=10)
    fig.savefig(BASE/'reference_bscan.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5),constrained_layout=True)
    draw_bscan(ax,d['amplitude'],d['time_s'],d['x_m'],
        'Реальные данные SFCW: арматура в бетоне\nProceq GP8800, номинально 0.4–6 ГГц')
    fig.suptitle('Solla et al., Zenodo10962520, образец III\nВычтен средний профиль; клиппинг 99%; нуль времени не калиброван',fontsize=9)
    fig.savefig(BASE/'reference_bscan_native.png');plt.close(fig)
    return records


class SensorData:
    """Inert state holder for the authors' pickled acquisition metadata."""


class RestrictedUnpickler(pickle.Unpickler):
    allowed = {'pandas.core.frame.DataFrame','pandas.core.internals.managers.BlockManager',
        'pandas._libs.internals._unpickle_block','numpy.core.numeric._frombuffer',
        'numpy.dtype','builtins.slice','numpy.core.multiarray._reconstruct','numpy.ndarray',
        'pandas.core.indexes.base._new_Index','pandas.core.indexes.base.Index',
        'pandas.core.indexes.range.RangeIndex','numpy.core.multiarray.scalar'}
    def find_class(self, module, name):
        if (module, name)==('SensorData', 'SensorData'):
            return SensorData
        if module+'.'+name not in self.allowed:
            raise ValueError(f'Unapproved pickle class: {module}.{name}')
        return getattr(importlib.import_module(module), name)


def soil():
    # Never deserialize a changed/unknown pickle. No pickle read occurs by default.
    expected='42fe50e669db89e77ed80b27b4f62a068f306f0b1a91ee87da8f189c43ce72de'
    if sha256(BASE/'Radar_data_emi_data.pkl') != expected:
        raise ValueError('SAL dataset SHA256 mismatch; refusing pickle deserialization')
    sys.path.insert(0,str(BASE/'_deps'))
    with (BASE/'Radar_data_emi_data.pkl').open('rb') as stream:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',DeprecationWarning)
            data=RestrictedUnpickler(stream).load()
    values=data[[f'bin_{j}' for j in range(400)]].to_numpy()
    if values.dtype.kind=='c':
        raise ValueError('Unexpected complex data: re-evaluate the phase availability')
    # Paper states endpoints and 400 steps but actual instrument frequency vector
    # is not in this export. Keep ordinal bins authoritative, nominal grid labelled.
    nominal_freq=np.linspace(1.3e9,2.9e9,400)
    np.savez_compressed(BASE/'soil_golfcourse_frequency_readings.npz',
        readings=values, frequency_bin=np.arange(400), nominal_frequency_hz=nominal_freq,
        x_m=data.dist_x.to_numpy(), y_m=data.dist_y.to_numpy(),
        latitude=data.radar_inter_lat.to_numpy(), longitude=data.radar_inter_long.to_numpy(),
        site=data.radar_file_name.to_numpy(dtype=str),
        apparent_conductivity_ECaR1=data.ECaR1.to_numpy(),
        original_row_index=data.old_index.to_numpy(),
        units=np.array('radar instrument reading; units and quadrature convention unspecified'),
        not_a_time_bscan=np.array(True))
    fig,axs=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    for ax,site in zip(axs[:2],['FWY14','FWY16']):
        s=data[data.radar_file_name==site]
        vmin,vmax=np.quantile(s.ECaR1,[.005,.995])
        im=ax.scatter(s.dist_x,s.dist_y,c=s.ECaR1,s=5,cmap='viridis',vmin=vmin,vmax=vmax)
        ax.set(xlabel='Локальная координата X, м',ylabel='Локальная координата Y, м',title=f'{site}: измеренный ECaR1',aspect='equal')
        fig.colorbar(im,ax=ax,extend='both',label='EMI, единицы экспорта; шкала 0.5–99.5%')
    subset=values[data.radar_file_name.to_numpy()=='FWY16'][:200]
    im=axs[2].imshow((subset-subset.mean(axis=0)).T,origin='lower',aspect='auto',cmap='RdBu_r',
        extent=[0,subset.shape[0]-1,0,399],vmin=-np.percentile(abs(subset-subset.mean(axis=0)),99),
        vmax=np.percentile(abs(subset-subset.mean(axis=0)),99))
    axs[2].set(xlabel='Порядковый номер измерения FWY16',ylabel='Номер частотного отсчёта',
        title='Частотные показания SFCW\nЭто НЕ временной B-скан')
    fig.suptitle('Реальный травяной участок Fontana: SFCW 1.3–2.9 ГГц\nEMI включает фон трактора. Нет комплексной фазы и истинного разреза ε по глубине',fontsize=13)
    fig.savefig(BASE/'soil_golfcourse_measured_context.png');plt.close(fig)
    record=dict(rows=len(data),frequency_bins=400,site_counts=data.radar_file_name.value_counts().to_dict(),
        finite_readings=bool(np.isfinite(values).all()),sha256=sha256(BASE/'Radar_data_emi_data.pkl'),
        original_radar_columns=list(data.radar_data.iloc[0].columns),
        limitation='400 real-valued readings, no documented complex phase/IQ; not converted to time or depth.',
        license='No explicit repository license found. Public data published by authors for research; do not assume a redistribution license.',
        preprocessing_note='Repository has3508 samples; paper3472 after trimming upper/lower0.5% EMI tails; untrimmed published values retained.',
        frequency_axis_note='Frequency-bin index authoritative; nominal linspace1.3–2.9GHz based on paper, not calibrated sweepvector.')
    (BASE/'soil_data_summary.json').write_text(json.dumps(record,indent=2,ensure_ascii=False))
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-soil',action='store_true',help='Explicitly parse the checksum-verified public SAL pickle using an allowlist')
    args=parser.parse_args()
    records=concrete()
    sal=soil() if args.include_soil else json.loads((BASE/'soil_data_summary.json').read_text())
    sources=dict(
        concrete=dict(dataset='https://zenodo.org/records/10962520',doi='10.5281/zenodo.10962520',
            paper='https://doi.org/10.1016/j.prostr.2024.09.250',
            creator='Mercedes Solla; SFCW data provided by Screening Eagle Technologies',
            license='GPL-3.0-or-later',archive_md5=hashlib.md5((BASE/'SFCW.zip').read_bytes()).hexdigest(),
            expected_archive_md5='db35ab89b13301cd2b871e11ccf39c72',
            instrument='Proceq GP8800, bistatic ground-coupled SFCW',native_band_hz=[400e6,6000e6],
            tx_rx_spacing_m=.036,trace_spacing_m=.005,time_window_s=16e-9,
            material='Concrete specimens with reinforcing bars; NOT park soil',
            processing='Instrument time-domain SEG-Y export. Internal phase/calibration/windowing not fully specified. Local plots subtract profile mean and clip at99th percentile; low-band files add documented cosine filter.',
            quantitative_comparison_limit='Different material, antenna and electronics. Shared filter band permits diagnostic comparison only, not A-scan agreement or calibration validation.',
            profiles=records),
        soil=dict(dataset='https://opensource.silicon-austria.com/xuc/soil-analysis-machine-learning-stepped-frequency-gpr',
            paper='https://arxiv.org/html/2404.15961v2',
            commit=json.loads((BASE/'sal_commit.json').read_text())[0]['id'],
            authors='Chunlei Xu et al., Silicon Austria Labs / Geoprospectors',
            band_hz=[1.3e9,2.9e9],steps=400,sweep_duration_s=.070,
            instrument='Geoprospectors prototype, bistatic air-coupled Vivaldi',
            tx_rx_feedpoint_spacing_m=.60,ground_clearance_m=.15,tilt_from_vertical_deg=23,
            scene='Fontana Golf Club, fairways14and16, Austria,25May2023',**sal))
    (BASE/'sources.json').write_text(json.dumps(sources,indent=2,ensure_ascii=False))
    print(json.dumps({'concrete_profiles':records,'soil':sal},indent=2,ensure_ascii=False))
