"""One immutable Peplinski scene, then exact voxel crops for all antenna positions."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import hashlib
import tempfile

import h5py
import numpy as np

# g/cm³: sand/mineral particle-density assumption used in the official Peplinski
# example, not a universal measured soil value. Porosity = 1 - bulk/particle.
# https://docs.gprmax.com/en/latest/input_hash_cmds.html#soil-peplinski
# Original constitutive model: Peplinski et al. (1995), doi:10.1109/36.387598.
from .physical_constants import MINERAL_DENSITY_G_CM3


GENERATORS = ('native_peplinski', 'custom_crim_archie')


def _geometry_environment():
    """Share Matplotlib's font cache across scenes and independent processes.

    gprMax imports plotting code even for geometry-only runs. Giving each new
    scene a fresh MPLCONFIGDIR repeats font discovery. Respect a usable caller
    setting, then try the project cache and a user-specific temporary cache if
    the installation is read-only. Matplotlib handles concurrent cache writers.
    Paths must be absolute because the child process changes its directory.
    """
    env = dict(os.environ)
    candidates = []
    if env.get('MPLCONFIGDIR'):
        candidates.append(Path(env['MPLCONFIGDIR']).expanduser().resolve())
    candidates.extend((Path(__file__).resolve().parents[1] / '.cache/matplotlib',
                       Path(tempfile.gettempdir()) / f'geodatagen-matplotlib-{os.getuid() if hasattr(os, "getuid") else "user"}'))
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            # Actual write probe also handles ACLs/read-only mounts; os.access
            # alone cannot establish that the subprocess can create its cache.
            with tempfile.TemporaryFile(dir=candidate):
                pass
        except OSError:
            continue
        env['MPLCONFIGDIR'] = str(candidate)
        return env
    # Leave fallback handling to Matplotlib if even the temporary directory is
    # unwritable; this optimization must not introduce a new failure mode.
    env.pop('MPLCONFIGDIR', None)
    return env


def custom_generator_fingerprint():
    """Version both the provided algorithm and its local geometry adapter."""
    base = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ('soil.py', 'supervisor_fields.py', 'physical_constants.py'):
        digest.update(name.encode())
        digest.update((base / name).read_bytes())
    return digest.hexdigest()


def prepare_scene(directory, preset, grid, seed, object_lines=(), generator=None):
    """Build once with native gprMax fractals; preserve Debye materials on disk.

    preset['layers'] is top-down. Each layer gives thickness and either
    sand/clay/bulk_density/water_min/water_max or constant er/sigma.
    The last layer fills all remaining depth; dimensions are rounded to dx.
    """
    generator = generator or preset.get('generator', 'native_peplinski')
    if generator not in GENERATORS:
        raise ValueError(f'Unknown soil generator {generator!r}; choose one of {GENERATORS}')
    if generator == 'custom_crim_archie':
        return _prepare_custom_scene(directory, preset, grid, seed, object_lines)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    dx = float(grid['dx'])
    domain = np.round(np.asarray(grid['domain']) / dx).astype(int) * dx
    surface = round(grid['surface_z'] / dx) * dx
    if not 0 < surface < domain[2]:
        raise ValueError('Soil surface must be inside domain')
    fmt = lambda xyz: ' '.join(f'{v:.9g}' for v in xyz)
    # Legacy fallback 12 ns/2 threads is a runtime setting, not a ground property;
    # validated scene configs explicitly specify both before this function runs.
    lines = [f"#title: {preset.get('name', 'soil')} fixed global scene seed {seed}",
             f'#domain: {fmt(domain)}', f'#dx_dy_dz: {dx} {dx} {dx}',
             f"#time_window: {grid.get('time_window', 12e-9)}",
             f"#num_threads: {grid.get('threads', 2)}"]
    top = surface
    layers = []
    for i, layer in enumerate(preset['layers']):
        bottom = max(0, round((top - layer['thickness']) / dx) * dx)
        if i == len(preset['layers']) - 1:
            bottom = 0
        if bottom >= top:
            raise ValueError('Layer must contain at least one voxel')
        name = f'layer_{i}'
        corners = fmt((0, 0, bottom, domain[0], domain[1], top))
        if 'sand' in layer:
            s, c = layer['sand'], layer['clay']
            rho, theta0, theta1 = layer['bulk_density'], layer['water_min'], layer['water_max']
            if not (s >= 0 and c >= 0 and s + c <= 1 and 0 <= theta0 < theta1 < 1 - rho / MINERAL_DENSITY_G_CM3):
                raise ValueError('Invalid soil fractions, moisture range or porosity')
            # Twenty material bins and fractal dimension 1.5 are configurable
            # discretization/statistical defaults, not measured site properties.
            # Unit weighting selects isotropic generation absent an explicit prior.
            n = int(layer.get('materials', 20))
            weighting = layer.get('weighting', [1, 1, 1])
            lines += [f'#soil_peplinski: {s} {c} {rho} {MINERAL_DENSITY_G_CM3} {theta0} {theta1} {name}',
                      f"#fractal_box: {corners} {layer.get('fractal_dimension', 1.5)} {fmt(weighting)} {n} {name} box_{i} {seed+i} n"]
        else:
            # mu_r=1 and magnetic loss=0 assume nonmagnetic civil materials.
            # Missing sigma=0 means an explicitly lossless dielectric approximation.
            lines += [f"#material: {layer['er']} {layer.get('sigma', 0)} 1 0 {name}",
                      f'#box: {corners} {name} n']
        layers.append(dict(layer, z_bottom_m=bottom, z_top_m=top, seed=seed+i))
        top = bottom
    lines.extend(object_lines)
    # Only the solver's canonical geometry is written here; the unified exporter
    # produces scene.vti with physical arrays. Avoid a redundant full VTK volume.
    lines += [f'#geometry_objects_write: 0 0 0 {fmt(domain)} global']
    inp = directory / 'scene.in'
    inp.write_text('\n'.join(lines) + '\n')
    env = _geometry_environment()
    with (directory / 'geometry.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'gprMax', str(inp), '--geometry-only'],
                       check=True, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT)
    geo = directory / 'global.h5'
    with h5py.File(geo) as f:
        database = f.attrs['MaterialDatabase']
        if isinstance(database, bytes): database = database.decode()
        shape = list(f['data'].shape)
    meta = dict(name=preset.get('name'), generator='native_peplinski', seed=seed, dx=dx, domain=domain.tolist(),
                surface_z=surface, shape=shape, layers=layers, geometry='global.h5',
                materials=f'{database}.json', geometry_sha256=sha256(geo),
                materials_sha256=sha256(directory / f'{database}.json'))
    (directory / 'soil.json').write_text(json.dumps(meta, indent=2))
    return meta


def _prepare_custom_scene(directory, preset, grid, seed, object_lines):
    """Import the provided CRIM/Archie field through gprMax's own geometry writer.

    Explicit water conductivity is required: it cannot be inferred from the
    native Peplinski inputs. Fixed engineering layers retain their er/sigma.
    """
    from dataclasses import asdict
    from .supervisor_fields import Layer, Config, generate, material_palette

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    dx = float(grid['dx'])
    domain = np.round(np.asarray(grid['domain']) / dx).astype(int) * dx
    shape = tuple(int(n) for n in np.rint(domain / dx))
    surface = round(grid['surface_z'] / dx) * dx
    options = dict(preset.get('custom_field', {}))
    permitted = {'bins_epsilon', 'bins_log_sigma', 'water_epsilon_r', 'max_fft_cells', 'surface_roughness_rms_m'}
    if set(options) - permitted:
        raise ValueError(f'Unknown custom_field options: {sorted(set(options)-permitted)}')
    if options.get('surface_roughness_rms_m', 0) != 0:
        raise ValueError('This pipeline keeps the air/soil surface flat for fixed-height Tx/Rx; use layer roughness instead')
    cfg = Config(shape=shape, spacing_m=(dx,)*3, surface_z_m=surface, seed=seed, **options)
    layers, metadata_layers, constants = [], [], {}
    top = surface
    for i, original in enumerate(preset['layers']):
        bottom = max(0, round((top - original['thickness']) / dx) * dx)
        if i == len(preset['layers'])-1:
            bottom = 0
        if bottom >= top:
            raise ValueError('Layer must contain at least one voxel')
        if 'sand' in original:
            rho = float(original['bulk_density'])
            lo, hi = float(original['water_min']), float(original['water_max'])
            porosity = 1-rho/MINERAL_DENSITY_G_CM3
            if not (0 < porosity < 1 and 0 <= lo < hi < porosity):
                raise ValueError('Invalid density or moisture range for custom field')
            custom = dict(original.get('custom', {}))
            if 'water_conductivity_s_m' not in custom:
                raise ValueError('custom_crim_archie requires layer.custom.water_conductivity_s_m; it is not determined by Peplinski inputs')
            # Map uniform [lo,hi] moisture moments to saturation: mean=(lo+hi)/2,
            # standard deviation=(hi-lo)/sqrt(12), divided by nominal porosity.
            # This matches two moments, not the resulting field distribution.
            params = dict(name=f'layer_{i}', thickness_m=top-bottom,
                          porosity=porosity, porosity_sd=0.0,
                          saturation=(lo+hi)/(2*porosity),
                          saturation_sd=(hi-lo)/(np.sqrt(12)*porosity))
            params.update(custom)
            params['name'], params['thickness_m'] = f'layer_{i}', top-bottom
            for key in ('correlation_m', 'roughness_corr_m'):
                if key in params:
                    params[key] = tuple(params[key])
            layer = Layer(**params)
        else:
            # Placeholder fields are replaced by the exact engineering material.
            # A flat lower boundary preserves asphalt/base thicknesses. Values
            # .3/.2/.05 below are discarded placeholder porosity/saturation/water
            # conductivity; actual er/sigma overwrite them and pore fields are NaN.
            layer = Layer(f'layer_{i}', top-bottom, .3, 0, .2, 0, .05,
                          roughness_rms_m=0)
            constants[i+1] = (float(original['er']), float(original.get('sigma', 0)))
        layers.append(layer)
        metadata_layers.append(dict(original, z_bottom_m=bottom, z_top_m=top,
                                    custom_resolved=asdict(layer), seed=seed))
        top = bottom
    fields = generate(tuple(layers), cfg)
    for lid, (er, sigma) in constants.items():
        mask = fields.layer_id == lid
        fields.epsilon_r[mask], fields.sigma_s_m[mask] = er, sigma
        fields.porosity[mask], fields.saturation[mask] = np.nan, np.nan
    ids, keys, materials = material_palette(fields)
    prefix = 'custom_seed_materials'
    palette = directory / (prefix+'.json')
    palette.write_text(json.dumps(dict(schema='gprMax-material-database', schema_version=1,
        database=dict(id=prefix, name='Provided correlated CRIM/Archie soil', version='1.0.0'),
        materials=materials), indent=2)+'\n')
    geometry = directory/'custom_seed_geometry.h5'
    with h5py.File(geometry, 'w') as f:
        f.attrs['dx_dy_dz'] = np.asarray(cfg.spacing_m)
        f['data'] = ids
        f.create_dataset('material_keys', data=np.asarray(keys, dtype=object), dtype=h5py.string_dtype())
    fmt = lambda xyz: ' '.join(f'{v:.9g}' for v in xyz)
    lines = [f"#title: {preset.get('name', 'soil')} custom CRIM/Archie seed {seed}",
             f'#domain: {fmt(domain)}', f'#dx_dy_dz: {dx} {dx} {dx}',
             f"#time_window: {grid.get('time_window', 12e-9)}",
             f"#num_threads: {grid.get('threads', 2)}",
             f'#geometry_objects_read: 0 0 0 {geometry.name} {prefix} n',
             *object_lines,
             f'#geometry_objects_write: 0 0 0 {fmt(domain)} global']
    inp = directory/'scene.in'
    inp.write_text('\n'.join(lines)+'\n')
    env = _geometry_environment()
    with (directory/'geometry.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'gprMax', str(inp), '--geometry-only'],
                       check=True, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT)
    geo = directory/'global.h5'
    with h5py.File(geo) as f:
        dbname = f.attrs['MaterialDatabase']
        if isinstance(dbname, bytes): dbname = dbname.decode()
        exported_ids = f['data'][:]
        exported_keys = [k.decode() if isinstance(k, bytes) else str(k) for k in f['material_keys'][:]]
    db = json.loads((directory/f'{dbname}.json').read_text())['materials']
    # gprMax may renumber or deduplicate equal materials. Identify soil entries
    # by the original palette key recorded by its geometry writer.
    is_custom = np.array([
        db[k].get('metadata', {}).get('source_database', {}).get('database_id') == prefix
        and db[k].get('metadata', {}).get('source_database', {}).get('entry_key') in keys
        for k in exported_keys])
    retained = is_custom[exported_ids]
    fields.porosity[~retained] = np.nan
    fields.saturation[~retained] = np.nan
    fields.layer_id[~retained & (fields.layer_id > 0)] = -1
    continuous = directory/'custom_fields.npz'
    np.savez_compressed(continuous, epsilon_r=fields.epsilon_r, sigma_s_m=fields.sigma_s_m,
                       porosity=fields.porosity, saturation=fields.saturation,
                       water_fraction=fields.porosity*fields.saturation,
                       layer_id=fields.layer_id, retained_soil_mask=retained,
                       spacing_m=np.asarray(cfg.spacing_m))
    meta = dict(name=preset.get('name'), generator='custom_crim_archie', seed=seed, dx=dx,
                domain=domain.tolist(), surface_z=surface, shape=list(shape), layers=metadata_layers,
                geometry='global.h5', materials=f'{dbname}.json', geometry_sha256=sha256(geo),
                materials_sha256=sha256(directory/f'{dbname}.json'),
                custom_fields=continuous.name, custom_fields_sha256=sha256(continuous),
                generator_sha256=custom_generator_fingerprint(), custom_field_config=asdict(cfg),
                material_model='nondispersive CRIM + Archie-type conduction',
                assumptions=['Nominal porosity = 1 - bulk_density/2.66 unless overridden.',
                             'Saturation mean and standard deviation derive from native moisture midpoint and uniform-range standard deviation; resulting Gaussian fields need not remain within native water_min/water_max.',
                             'Water conductivity and Archie/clay terms are explicit assumptions, not inferred from native soil fractions.',
                             'Layer intervals are nominal; internal interfaces may be rough.',
                             'epsilon_r/sigma_s_m in custom_fields.npz are continuous pre-object soil fields; use global.h5 for quantized actual object geometry.',
                             'No real-soil calibration or accuracy improvement over Peplinski is claimed.'])
    (directory/'soil.json').write_text(json.dumps(meta, indent=2)+'\n')
    return meta


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def crop_scene(scene_dir, output_path, origin_xy, window_xy=(1.2, 1.2)):
    """Copy fixed global voxels and all material definitions; no regenerated noise.

    Returns actual snapped global origin and the material database stem needed by
    #geometry_objects_read. Full z extent is preserved. No edge padding is hidden:
    windows outside the saved domain raise ValueError.
    """
    scene_dir, output_path = Path(scene_dir), Path(output_path)
    meta = json.loads((scene_dir / 'soil.json').read_text())
    dx = meta['dx']
    start = np.rint(np.asarray(origin_xy) / dx).astype(int)
    counts = np.rint(np.asarray(window_xy) / dx).astype(int)
    if np.any(start < 0) or np.any(start + counts > np.asarray(meta['shape'])[:2]):
        raise ValueError(f'Crop outside scene: origin={origin_xy}, window={window_xy}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(scene_dir / meta['geometry']) as source, h5py.File(output_path, 'w') as dest:
        data = source['data'][start[0]:start[0]+counts[0], start[1]:start[1]+counts[1], :]
        dest.create_dataset('data', data=data)
        dest.create_dataset('material_keys', data=source['material_keys'][:])
        for k in ('dx_dy_dz', 'MaterialDatabase', 'MaterialDatabaseSchemaVersion'):
            dest.attrs[k] = source.attrs[k]
    materials = scene_dir / meta['materials']
    target = output_path.parent / materials.name
    if target.resolve() != materials.resolve() and not target.exists():
        try: os.link(materials, target)
        except OSError: shutil.copyfile(materials, target)
    return dict(origin_xy=(start * dx).tolist(), domain=(counts * dx).tolist() + [meta['domain'][2]],
                material_database=materials.stem, voxel_sha256=hashlib.sha256(data.tobytes()).hexdigest())
