"""Independent material and saved-signal checks."""
from pathlib import Path
import json
import numpy as np
import h5py

def _corr(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else None


def soil_statistics(scene_dir, frequency):
    from gprMax.materials import calculate_water_properties
    scene_dir = Path(scene_dir)
    meta = json.loads((scene_dir/'soil.json').read_text())
    database = json.loads((scene_dir/meta['materials']).read_text())['materials']
    with h5py.File(scene_dir/meta['geometry']) as f:
        voxels = f['data'][:]
        keys = [k.decode() if isinstance(k, bytes) else str(k) for k in f['material_keys'][:]]
    if voxels.min() < 0 or voxels.max() >= len(keys):
        raise ValueError('Voxel material index outside material_keys')
    er, debye_delta = [], []
    for key in keys:
        m = database[key]
        eps = float(m['base']['relative_permittivity'])
        for pole in m.get('poles', []):
            eps += pole['relative_permittivity_difference'] / (1+(2*np.pi*frequency*pole['relaxation_time_s'])**2)
        er.append(eps)
        debye_delta.append(m['poles'][0]['relative_permittivity_difference'] if m['model']=='debye' and len(m.get('poles', []))==1 else np.nan)
    er = np.asarray(er); debye_delta = np.asarray(debye_delta)
    # Match gprMax Peplinski water defaults: 25 °C, zero salinity; no local measurement.
    water_inf, water_static, _, _ = calculate_water_properties(25, 0)
    custom = None
    if meta.get('generator') == 'custom_crim_archie':
        with np.load(scene_dir/meta['custom_fields']) as f:
            custom = {name: f[name].copy() for name in ('water_fraction','layer_id')}
    layers = []
    for i, layer in enumerate(meta['layers']):
        k0, k1 = np.rint(np.array([layer['z_bottom_m'], layer['z_top_m']])/meta['dx']).astype(int)
        indices = voxels[:, :, k0:k1]
        record = dict(layer=i, z_bottom_m=layer['z_bottom_m'], z_top_m=layer['z_top_m'])
        if 'sand' in layer:
            if custom is not None:
                # Rough interfaces must be selected by their true labels, not
                # a rectangular crop using the nominal layer depths.
                mask = custom['layer_id'] == i+1
                theta = np.where(mask, custom['water_fraction'], np.nan)
                values = theta[np.isfinite(theta)]
                if not values.size:
                    raise ValueError(f'No custom soil voxels in layer {i}')
                adjacent = [_corr(np.take(theta,range(theta.shape[a]-1),axis=a),
                                  np.take(theta,range(1,theta.shape[a]),axis=a)) for a in range(3)]
                # Subsample/shuffle comparison is a deterministic diagnostic (seed17),
                # not a geological threshold or statistical significance test.
                sample = values[::max(1,len(values)//100000)]
                shuffled = _corr(sample,np.random.default_rng(17).permutation(sample))
                record.update(soil_voxels=int(len(values)), theta_mean=float(values.mean()),
                              theta_std=float(values.std()), theta_min=float(values.min()),
                              theta_max=float(values.max()), lag1_xyz=adjacent,
                              shuffled_correlation=shuffled,
                              correlation_exceeds_shuffled=bool(shuffled is not None and all(v is not None and v>abs(shuffled)+.05 for v in adjacent)),
                              finite=True, nonconstant=bool(values.std()>0),
                              moisture_source='continuous generated porosity*saturation, object voxels excluded',
                              layer_bounds='nominal; actual mask follows rough interface',
                              material_model='nondispersive CRIM/Archie, quantized for FDTD')
                layers.append(record)
                continue
            # Invert exactly the installed native Peplinski Debye-amplitude relation.
            # IDs themselves do NOT represent moisture or even ordered material bins.
            # Peplinski empirical beta-imag coefficients, doi:10.1109/36.387598;
            # exponent 0.65 is the mixing-model alpha, not a fitted scene value.
            b2 = 1.33797-.603*layer['sand']-.166*layer['clay']
            theta_lut = (debye_delta/(water_static-water_inf))**(.65/b2)
            theta = theta_lut[indices]
            values = theta[np.isfinite(theta)]
            if not values.size: raise ValueError(f'No soil voxels in layer {i}')
            edges = np.linspace(layer['water_min'], layer['water_max'], layer.get('materials',20)+1)
            bins = (edges[:-1]+edges[1:])/2
            recovered = np.unique(values)
            residual = np.min(abs(recovered[:,None]-bins[None,:]),axis=1).max()
            if residual > 1e-8: raise ValueError('Moisture reconstruction does not match native layer bins')
            adjacent = [_corr(np.take(theta,range(theta.shape[a]-1),axis=a),np.take(theta,range(1,theta.shape[a]),axis=a)) for a in range(3)]
            rng = np.random.default_rng(17)
            sample = values[::max(1,len(values)//100000)]
            shuffled = _corr(sample,rng.permutation(sample))
            record.update(soil_voxels=int(len(values)), excluded_object_voxels=int(theta.size-len(values)),
                          theta_mean=float(values.mean()),theta_std=float(values.std()),theta_min=float(values.min()),theta_max=float(values.max()),
                          theta_bin_reconstruction_error=float(residual),lag1_xyz=adjacent,shuffled_correlation=shuffled,
                          correlation_exceeds_shuffled=all(v is not None and v>abs(shuffled)+.05 for v in adjacent),
                          finite=True,nonconstant=bool(values.std()>0))
        else:
            record.update(constant_material=True, er=layer['er'], sigma=layer.get('sigma',0))
        layers.append(record)
    return meta, voxels, er, layers



def check_signal(path, tx, rx, config_hash):
    """Validate every finished row and its position; locked files raise OSError."""
    with h5py.File(path, 'r') as f:
        n = len(tx)
        if f.attrs.get('config_sha256') != config_hash: raise ValueError('configuration hash mismatch')
        for key, expected in [('tx_m', tx), ('rx_m', rx)]:
            if key not in f or not np.array_equal(f[key][:], expected): raise ValueError(f'{key} does not match path coordinates')
        mask = f['complete'][:]
        if mask.shape != (n,) or mask.dtype.kind != 'b': raise ValueError('invalid complete mask')
        done = int(mask.sum())
        if 'Ey' not in f:
            if done: raise ValueError('completed rows without signal')
            return dict(completed=0, total=n, valid=False)
        a = f['Ey']
        t = f['time_s'][:]
        dt = float(f.attrs['dt'])
        if a.ndim != 2 or a.shape != (n, len(t)) or len(t) < 2: raise ValueError('signal/time shape mismatch')
        if not np.isfinite(t).all() or not np.isfinite(dt) or dt <= 0: raise ValueError('invalid time axis')
        if not np.allclose(t, np.arange(len(t))*dt, rtol=1e-10, atol=1e-18): raise ValueError('time axis disagrees with dt')
        seconds = f['seconds'][:]
        if seconds.shape != (n,) or not np.isfinite(seconds[mask]).all() or np.any(seconds[mask] <= 0): raise ValueError('invalid completed timing metadata')
        for start in range(0, n, 32):
            rows = a[start:start+32][mask[start:start+32]]
            if not np.isfinite(rows).all(): raise ValueError(f'nonfinite signal after row {start}')
            if len(rows) and np.any(np.max(np.abs(rows), axis=1) == 0): raise ValueError(f'zero completed signal after row {start}')
        if f.attrs.get('units') != 'V/m' or f.attrs.get('axis_order') != 'scan,time': raise ValueError('missing signal units/axis order')
        return dict(completed=done, total=n, samples=len(t), dt_s=dt, valid=done == n)
