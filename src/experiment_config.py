"""Resolve separate scene and survey inputs into a reproducible experiment."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import numpy as np

from .physical_constants import MINERAL_DENSITY_G_CM3, GEOMETRY_CLEARANCE_M, OBJECT_CLEARANCE_M
from .layout import load_object_coordinates, MATERIALS
from .soil import GENERATORS, custom_generator_fingerprint


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'Nonfinite JSON number: {value}')
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def number(value, name, minimum=None, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if minimum is not None and (value <= minimum if strict else value < minimum):
        raise ValueError(f'{name} must be {">" if strict else ">="} {minimum}')
    return value


def integer(value, name, minimum=0):
    number(value, name, minimum)
    if int(value) != value:
        raise ValueError(f'{name} must be an integer')
    return int(value)


def random_objects(spec, grid, default_seed):
    """Place spheres in declared XY bounds, independently of the survey path."""
    allowed = {'seed', 'clearance_m', 'bounds_m', 'rules', 'method', 'min_distance_m'}
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError('Unknown object_generation fields')
    seed = integer(spec.get('seed', default_seed), 'object_generation.seed')
    clearance = number(spec.get('clearance_m', OBJECT_CLEARANCE_M), 'clearance_m', 0)
    pad = max(GEOMETRY_CLEARANCE_M, grid['pml_thickness'])
    bounds = np.asarray(spec.get('bounds_m', [[pad, grid['domain'][0]-pad], [pad, grid['domain'][1]-pad]]), dtype=float)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError('bounds_m requires [[xmin,xmax],[ymin,ymax]]')
    if np.any(bounds[:, 0] < pad) or np.any(bounds[:, 1] > np.asarray(grid['domain'][:2])-pad):
        raise ValueError('Object generation bounds must stay inside the scene and outside PML')
    rng = np.random.default_rng(seed)
    method = spec.get('method', 'uniform')
    if method not in ('uniform', 'sobol', 'poisson_disk'):
        raise ValueError('method must be uniform, sobol or poisson_disk')
    exclusion = spec.get('min_distance_m', 0)
    if method == 'poisson_disk':number(exclusion, 'min_distance_m', 0, True)
    elif 'min_distance_m' in spec:raise ValueError('min_distance_m applies only to poisson_disk')
    result = []
    rules = spec.get('rules', [])
    if not isinstance(rules, list):
        raise ValueError('object_generation.rules must be a list')
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != {'type', 'count', 'radius_m', 'depth_m'} or rule['type'] not in MATERIALS:
            raise ValueError('Each object rule requires type, count, radius_m, depth_m')
        count = rule['count']
        if isinstance(count, list):
            if len(count) != 2:
                raise ValueError('count must be an integer or [minimum,maximum]')
            lo, hi = [integer(x, 'count') for x in count]
            if lo > hi:
                raise ValueError('count minimum exceeds maximum')
            count = int(rng.integers(lo, hi+1))
        else:
            count = integer(count, 'count')
        ranges = []
        for key in ('radius_m', 'depth_m'):
            values = rule[key]
            if not isinstance(values, list) or len(values) != 2:
                raise ValueError(f'{key} requires [minimum,maximum]')
            lo, hi = [number(x, key, 0, True) for x in values]
            if hi < lo:
                raise ValueError(f'{key} minimum exceeds maximum')
            ranges.append((lo, hi))
        if ranges[0][0] < grid['dx']:
            raise ValueError('Object radius must be at least one geometry cell')
        from .sampling import proposals
        # A fixed burial depth is a valid object constraint. Sample a dummy
        # unit interval in that dimension and replace it by the exact depth;
        # spatial X/Y bounds still need nonzero extent.
        fixed_depth = ranges[1][0] == ranges[1][1]
        depth_bounds = [0, 1] if fixed_depth else list(ranges[1])
        candidates = iter(proposals([*bounds.tolist(), depth_bounds], max(1, count * 2000), int(rng.integers(2**31)), method))
        for _ in range(count):
            for attempt in range(2000):
                r = float(rng.uniform(*ranges[0]))
                candidate = next(candidates)
                xy = candidate[:2]
                depth = ranges[1][0] if fixed_depth else float(candidate[2])
                c = np.r_[xy, grid['surface_z']-depth]
                if np.any(c-r < pad) or np.any(c+r > np.asarray(grid['domain'])-pad) or c[2]+r > grid['surface_z']-clearance:
                    continue
                if any(np.linalg.norm(c-o['center_m']) < max(r+o['radius_m']+clearance, exclusion) for o in result):
                    continue
                result.append(dict(type=rule['type'], center_m=c.tolist(), radius_m=r))
                break
            else:
                raise ValueError('Cannot place all objects: reduce count/radii or enlarge bounds')
    return result, clearance


def load_settings(path, mode=None, trajectory_path=None):
    try:
        return _load_settings(path, mode, trajectory_path)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f'Missing or malformed scene configuration field: {exc}') from exc


def _load_settings(path, mode=None, trajectory_path=None):
    from .trajectories import build_trajectory
    from .experiment_solver import validate_positions

    source_path = Path(path).resolve()
    raw = read_json(source_path)
    if not isinstance(raw, dict):
        raise ValueError('Scene config must be a JSON object')
    if raw.get('schema_version', 2) != 2:
        raise ValueError('Unsupported scene schema_version; expected 2')
    if 'presets' in raw:
        raise ValueError('Use a scene from examples/scenes/ instead of legacy project.json')
    cfg = deepcopy(raw)
    cfg['schema_version'] = 2
    cfg['generator'] = mode or raw.get('generator', 'custom_crim_archie')
    if cfg['generator'] not in GENERATORS:
        raise ValueError(f'generator must be one of {GENERATORS}')
    cfg['seed'] = integer(cfg.get('seed', 0), 'seed')
    g, s, preset = cfg['grid'], cfg['source'], cfg['preset']
    if not isinstance(g['domain'], list) or len(g['domain']) != 3:
        raise ValueError('grid.domain requires three dimensions')
    for key, value in [*(('domain', v) for v in g['domain']), *((key, g[key]) for key in ('dx', 'solver_dx', 'time_window', 'pml_thickness'))]:
        number(value, key, 0, True)
    g['threads'] = integer(g.get('threads', 2), 'threads', 1)
    number(g['surface_z'], 'surface_z', 0, True)
    if g['surface_z'] >= g['domain'][2]:
        raise ValueError('Ground surface must be inside the domain')
    for value in [*g['domain'], g['surface_z']]:
        if not np.isclose(value/g['dx'], round(value/g['dx'])):
            raise ValueError('Domain and surface must align with geometry dx')
    factor = g['dx']/g['solver_dx']
    if factor < 1 or not np.isclose(factor, round(factor)):
        raise ValueError('solver_dx must divide geometry dx')
    if not np.isclose(g['pml_thickness']/g['solver_dx'], round(g['pml_thickness']/g['solver_dx'])):
        raise ValueError('PML thickness must align with solver_dx')
    for key in ('frequency_hz', 'moment_Am'):
        number(s[key], key, 0, True)
    number(s['height_m'], 'height_m', 0)
    number(s['tx_rx_offset_m'], 'tx_rx_offset_m', 0)
    from gprMax.waveforms import Waveform
    if s['waveform'] not in Waveform.types or s['waveform'] == 'user':
        raise ValueError('Select a built-in gprMax waveform')
    g['frequency'] = s['frequency_hz']
    layers = preset['layers']
    if not isinstance(layers, list) or not layers:
        raise ValueError('preset.layers must be a nonempty list')
    for layer in layers:
        number(layer['thickness'], 'layer thickness', 0, True)
        if not np.isclose(layer['thickness']/g['dx'], round(layer['thickness']/g['dx'])):
            raise ValueError('Layer thickness must align with geometry dx')
        if 'sand' in layer:
            sand, clay = [number(layer[k], k, 0) for k in ('sand', 'clay')]
            if sand+clay > 1:
                raise ValueError('Sand and clay fractions must sum to at most 1')
            density = number(layer['bulk_density'], 'bulk_density', 0, True)
            lo, hi = [number(layer[k], k, 0) for k in ('water_min', 'water_max')]
            if density >= MINERAL_DENSITY_G_CM3 or not 0 <= lo < hi < 1-density/MINERAL_DENSITY_G_CM3:
                raise ValueError('Moisture range must fit the soil porosity')
            if cfg['generator'] == 'custom_crim_archie' and 'water_conductivity_s_m' not in layer.get('custom', {}):
                raise ValueError('Model fields requires layer.custom.water_conductivity_s_m')
        else:
            number(layer['er'], 'er', 1)
            number(layer.get('sigma', 0), 'sigma', 0)
    if not np.isclose(sum(x['thickness'] for x in layers), g['surface_z']):
        raise ValueError('Layer thicknesses must add up to surface_z')
    if 'objects' in raw and 'object_generation' in raw:
        raise ValueError('Choose objects OR object_generation, not both')
    clearance = OBJECT_CLEARANCE_M
    if 'object_generation' in raw:
        objects, clearance = random_objects(raw['object_generation'], g, cfg['seed'])
    else:
        objects = raw.get('objects', [])
        if 'target' in raw and 'objects' not in raw:
            objects = [dict(type='metal', **raw['target'])]
    with tempfile.TemporaryDirectory(prefix='geodatagen-objects-') as temp:
        f = Path(temp)/'objects.json'
        f.write_text(json.dumps({'scene': objects}))
        checked = load_object_coordinates(f, ['scene'], g['domain'], g['surface_z'], clearance=clearance)['scene']
    for obj in checked:
        c, r = np.asarray(obj['center_m']), obj['radius_m']
        if r < g['dx'] or np.any(c-r < g['pml_thickness']) or np.any(c+r > np.asarray(g['domain'])-g['pml_thickness']):
            raise ValueError('Objects must resolve on the geometry grid and stay outside PML')
    cfg['experiment_objects'] = checked
    if trajectory_path is not None:
        route_path = Path(trajectory_path).resolve()
        route, base_dir = read_json(route_path), route_path.parent
    elif 'trajectory' in raw:
        route, base_dir = raw['trajectory'], source_path.parent
    elif 'profile' in raw:
        p = raw['profile']
        route = dict(type='line', start=[p['x_start_m'], p['y_m']], end=[p['x_end_m'], p['y_m']], step_m=p['step_m'])
        base_dir = source_path.parent
    else:
        raise ValueError('Specify a trajectory JSON with --trajectory')
    positions = build_trajectory(route, g, s, base_dir=base_dir)
    cfg['trajectory'] = positions['spec']
    cfg['simulation'] = cfg.get('simulation', {'mode': 'fixed'})
    cfg['profile'] = dict(n=len(positions['tx_m']), y_m=float(np.median(positions['tx_m'][:, 1])),
                          step_m=float(np.median(np.linalg.norm(np.diff(positions['tx_m'], axis=0), axis=1))) if len(positions['tx_m']) > 1 else g['solver_dx'])
    validate_positions(cfg, positions)
    sweep = cfg['sfcw']
    lo, hi, step = [number(sweep[k], k, 0, True) for k in ('fmin_hz', 'fmax_hz', 'df_hz')]
    if not lo < hi or hi/step >= 2048 or not all(np.isclose(v/step, round(v/step)) for v in (lo, hi)):
        raise ValueError('SFCW frequencies must have increasing, df-aligned bounds inside the 4096-point spectrum')
    if 1/step < g['time_window']:
        raise ValueError('SFCW period must cover the FDTD record')
    if cfg['generator'] == 'custom_crim_archie':
        cfg['generator_fingerprint'] = custom_generator_fingerprint()
    return cfg, positions
