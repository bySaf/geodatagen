"""Inspect native material maps and real solver records; incomplete data stay explicit."""
from pathlib import Path
import json
import h5py
import numpy as np
from scipy.spatial import cKDTree


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


def report_scene(scene_dir, frequency=4e8, plot=True, overlap_radius=.15):
    scene_dir = Path(scene_dir)
    meta, voxels, er, layers = soil_statistics(scene_dir, frequency)
    objects = json.loads((scene_dir/'objects.json').read_text()) if (scene_dir/'objects.json').exists() else []
    paths = []
    for filename in sorted(scene_dir.glob('path_*.npz')):
        with np.load(filename) as data:
            paths.append((filename.stem.removeprefix('path_'), data['tx'].copy(),data['rx'].copy()))
    xy = [p[1][:,:2] for p in paths]
    overlaps = [[float(np.mean(cKDTree(b).query(a)[0]<=overlap_radius)) for b in xy] for a in xy]
    checks = []
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        fig,ax = plt.subplots(figsize=(10,5))
        z = max(0, int(round((meta['surface_z']-.36)/meta['dx'])))
        im=ax.imshow(er[voxels[:,:,z]].T,origin='lower',extent=(0,meta['domain'][0],0,meta['domain'][1]),aspect='equal',cmap='viridis')
        fig.colorbar(im,ax=ax,label=f'Relative permittivity at {frequency/1e6:g} MHz')
        for name,tx,rx in paths: ax.plot(tx[:,0],tx[:,1],lw=.6,label=name)
        # Highlight the actual samples used by the same directed overlap metric.
        # Proximity to another path is not an electromagnetic sensitivity area.
        labelled = False
        for i, points in enumerate(xy):
            nearby = np.zeros(len(points), dtype=bool)
            for j, other in enumerate(xy):
                if i != j:
                    nearby |= cKDTree(other).query(points)[0] <= overlap_radius
            if nearby.any():
                ax.scatter(points[nearby,0],points[nearby,1],s=8,facecolors='none',
                           edgecolors='magenta',linewidths=.5,
                           label=f'Within {overlap_radius:g} m of another path' if not labelled else None)
                labelled = True
        for obj in objects:
            ax.add_patch(Circle(obj['center_m'][:2],obj['radius_m'],fill=False,ec='red',lw=1.5))
            ax.annotate(obj['type'],obj['center_m'][:2],fontsize=6)
        ax.set(xlabel='x, m',ylabel='y, m',title=f'{meta["name"]}: horizontal slice z={z*meta["dx"]:.2f} m; objects projected')
        if paths: ax.legend(loc='upper right',fontsize=7)
        fig.tight_layout();fig.savefig(scene_dir/'scene_paths.png',dpi=150);plt.close(fig)
    for name,tx,rx in paths:
        fn=scene_dir/f'bscan_{name}.h5'
        entry=dict(path=name,positions=len(tx),completed=0,has_solver_data=False)
        if fn.exists():
            with h5py.File(fn) as f:
                mask=f['complete'][:].astype(bool)
                a=f['Ey'][:] if 'Ey' in f else None
                times=f['time_s'][:] if 'time_s' in f else None
                entry.update(completed=int(mask.sum()),all_complete=bool(mask.all()),has_solver_data=a is not None,
                             coordinates_match=bool(np.array_equal(f['tx_m'][:],tx) and np.array_equal(f['rx_m'][:],rx)))
                if a is not None:
                    entry.update(samples_per_scan=a.shape[1],finite_completed=bool(np.isfinite(a[mask]).all()),
                                 nonzero_completed=bool(np.all(np.any(a[mask]!=0,axis=1))),
                                 timing_monotonic=bool(times is not None and np.all(np.diff(times)>0)))
            if plot and a is not None and mask.any():
                fig,axes=plt.subplots(1,2,figsize=(10,4)); selected=int(np.flatnonzero(mask)[len(np.flatnonzero(mask))//2])
                axes[0].plot(times*1e9,a[selected]);axes[0].set(xlabel='Time, ns',ylabel='Ey, V/m',title=f'A-scan {selected}')
                display=np.ma.array(a.T,mask=np.broadcast_to(~mask,a.T.shape));v=float(np.percentile(abs(a[mask]),99)) or 1
                axes[1].imshow(display,aspect='auto',cmap='seismic',vmin=-v,vmax=v,extent=(-.5,len(tx)-.5,times[-1]*1e9,0))
                axes[1].set(xlabel='Scan index (ordered along trajectory)',ylabel='Time, ns',title=f'{name}: {mask.sum()}/{len(tx)} complete; 99% colour scale')
                fig.tight_layout();fig.savefig(scene_dir/f'bscan_{name}.png',dpi=150);plt.close(fig)
                late = times >= 6e-9
                if late.sum() >= 2:
                    # Crop only in time. No background subtraction, trace gain,
                    # alignment or per-trace normalisation changes the samples.
                    late_values = a[:,late]
                    late_scale = float(np.percentile(abs(late_values[mask]),99)) or 1
                    late_image = np.ma.array(late_values.T,mask=np.broadcast_to(~mask,late_values.T.shape))
                    fig,ax = plt.subplots(figsize=(10,4))
                    im = ax.imshow(late_image,aspect='auto',cmap='seismic',vmin=-late_scale,vmax=late_scale,
                                   extent=(-.5,len(tx)-.5,times[late][-1]*1e9,times[late][0]*1e9))
                    fig.colorbar(im,ax=ax,label='Raw Ey, V/m; 99% display limit')
                    ax.set(xlabel='Scan index (ordered along trajectory)',ylabel='Time, ns',
                           title=f'{name}: late-time display, {mask.sum()}/{len(tx)} complete; blank = pending')
                    fig.text(.5,.015,'Time crop >= 6 ns only; raw HDF5 unchanged; no background subtraction or trace normalisation.',ha='center',fontsize=8)
                    fig.tight_layout(rect=(0,.045,1,1));fig.savefig(scene_dir/f'bscan_{name}_late.png',dpi=150);plt.close(fig)
                    entry['late_display'] = dict(start_ns=float(times[late][0]*1e9),colour_limit_V_per_m=late_scale,
                                                 scaling='global 99th percentile of completed late samples; display only')
        checks.append(entry)
    validation=dict(scene=meta['name'],frequency_hz=frequency,soil_layers=layers,objects=len(objects),
                    paths=checks,overlap_radius_m=overlap_radius,directed_overlap=overlaps,
                    completed_ascans=sum(p['completed'] for p in checks),
                    limitations='Material statistics and output integrity only; this does not prove FDTD convergence or real-world accuracy.')
    (scene_dir/'validation.json').write_text(json.dumps(validation,indent=2,allow_nan=False))
    return validation


def report(data, config):
    """Report all configured scenes without turning partial simulations into success."""
    data = Path(data)
    results = [report_scene(data/preset['name'],config['grid']['frequency'],overlap_radius=config['overlap_radius']) for preset in config['presets']]
    summary = dict(scenes=results,completed_ascans=sum(r['completed_ascans'] for r in results),
                   planned_ascans=len(config['presets'])*3*config['trajectory']['n'])
    summary['all_outputs_complete'] = bool(results) and all(
        len(r['paths'])==3 and all(p.get('positions')==config['trajectory']['n'] and p.get('all_complete',False) and p.get('finite_completed',False)
                                  and p.get('nonzero_completed',False) and p.get('coordinates_match',False)
                                  and p.get('timing_monotonic',False) for p in r['paths']) for r in results)
    (data/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    return summary
