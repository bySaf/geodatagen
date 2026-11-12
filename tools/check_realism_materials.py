#!/usr/bin/env python3
"""Check completed scene geometry/materials; never open live B-scan files.

Run from the project: .venv/bin/python tools/check_realism_materials.py
Only the JSON report is written. Input geometry and field files remain unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import json

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EPSILON_0 = 8.8541878128e-12


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(min=float(a.min()), mean=float(a.mean()),
                std=float(a.std()), max=float(a.max()))


def material_statistics(scene, frequency_hz):
    database = scene/'global_materials.json'
    geometry = scene/'global.h5'
    db = json.loads(database.read_text())['materials']
    with h5py.File(geometry, 'r') as h:
        ids = h['data'][:]
        keys = [k.decode() if isinstance(k, bytes) else str(k)
                for k in h['material_keys'][:]]
    counts = np.bincount(ids.ravel(), minlength=len(keys))
    weights, er_values, sigma_base, sigma_eff = [], [], [], []
    omega = 2*np.pi*frequency_hz
    for index, key in enumerate(keys):
        material = db[key]
        base = material['base']
        er = float(base['relative_permittivity'])
        sigma = float(base['electric_conductivity_s_per_m'])
        # In these two scenes all soil materials satisfy er>1 and finite sigma;
        # free space has er=1 and PEC has infinite sigma. Empty palette entries
        # are excluded. Weight by actual geometry occupancy, not palette size.
        if counts[index] == 0 or not np.isfinite(sigma) or er <= 1:
            continue
        if material['model'] not in ('constant', 'debye'):
            raise ValueError(f'Unsupported loss model: {material["model"]}')
        ep, loss = er, sigma
        for pole in material.get('poles', []):
            delta = float(pole['relative_permittivity_difference'])
            tau = float(pole['relaxation_time_s'])
            wt = omega*tau
            ep += delta/(1+wt**2)
            loss += EPSILON_0*omega*delta*wt/(1+wt**2)
        weights.append(int(counts[index]))
        er_values.append(ep)
        sigma_base.append(sigma)
        sigma_eff.append(loss)
    if not weights:
        raise ValueError(f'No soil materials in {scene}')
    return dict(
        frequency_hz=frequency_hz, soil_voxels=sum(weights),
        geometry_sha256=checksum(geometry), materials_sha256=checksum(database),
        epsilon_prime_voxel_weighted_mean=float(np.average(er_values, weights=weights)),
        sigma_base_voxel_weighted_mean_s_m=float(np.average(sigma_base, weights=weights)),
        sigma_effective_voxel_weighted_mean_s_m=float(np.average(sigma_eff, weights=weights)),
        selection='Occupied finite-conductivity materials with base epsilon_r>1; excludes air and PEC.',
        material_basis='Actual quantized material palette assigned to frozen geometry.')


def moisture_statistics(scene):
    meta = json.loads((scene/'soil.json').read_text())
    with np.load(scene/'custom_fields.npz', allow_pickle=False) as f:
        # Export provenance mask can also include the custom palette's air
        # entry. Positive layer_id is required to count physical soil cells.
        lid = f['layer_id']
        retained = f['retained_soil_mask'].astype(bool) & (lid > 0)
        phi, saturation, theta = f['porosity'], f['saturation'], f['water_fraction']
        finite = np.isfinite(phi) & np.isfinite(saturation) & np.isfinite(theta)
        mask = retained & finite
        violations = (theta < 0) | (theta > phi + 1e-7)
        if not mask.any():
            raise ValueError('No retained finite soil fields')
        layers = []
        for i, layer in enumerate(meta['layers']):
            m = mask & (lid == i+1)
            if not m.any():
                continue
            lo, hi = layer['water_min'], layer['water_max']
            x = theta[m]
            layers.append(dict(layer=i, soil_voxels=int(m.sum()),
                source_water_min=lo, source_water_max=hi,
                nominal_theta_midpoint=(lo+hi)/2,
                nominal_uniform_range_sd=(hi-lo)/np.sqrt(12),
                actual_theta=stats(x), actual_porosity=stats(phi[m]),
                actual_saturation=stats(saturation[m]),
                voxels_outside_source_water_interval=int(np.count_nonzero((x<lo)|(x>hi))),
                theta_exceeds_porosity_count=int(np.count_nonzero(violations[m]))))
        result = dict(
            source_sha256=checksum(scene/'custom_fields.npz'),
            retained_soil_voxels=int(retained.sum()), checked_soil_voxels=int(mask.sum()),
            nonfinite_retained_soil_voxels=int(np.count_nonzero(retained & ~finite)),
            theta_negative_count=int(np.count_nonzero((theta<0) & mask)),
            theta_exceeds_porosity_count=int(np.count_nonzero((theta>phi+1e-7) & mask)),
            saturation_outside_0_1_count=int(np.count_nonzero(((saturation<0)|(saturation>1)) & mask)),
            theta_equals_phi_times_s_max_abs_error=float(np.max(np.abs(theta[mask]-phi[mask]*saturation[mask]))),
            comparison_tolerance=1e-7,
            actual_theta=stats(theta[mask]), actual_porosity=stats(phi[mask]),
            actual_saturation=stats(saturation[mask]), layers=layers,
            field_basis='Continuous generated theta=porosity*saturation; object/air cells excluded. FDTD uses the separately quantized epsilon/sigma palette.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, default=ROOT/'user_review/realism/pilot')
    parser.add_argument('--output', type=Path, default=ROOT/'user_review/realism/material_validation.json')
    parser.add_argument('--frequency-hz', type=float, default=600e6)
    args = parser.parse_args()
    if not np.isfinite(args.frequency_hz) or args.frequency_hz <= 0:
        raise ValueError('Frequency must be positive and finite')
    materials = {mode: material_statistics(args.pilot/mode/'scene', args.frequency_hz)
                 for mode in ('native_peplinski', 'custom_crim_archie')}
    moisture = moisture_statistics(args.pilot/'custom_crim_archie/scene')
    report = dict(
        checker_sha256=checksum(Path(__file__)),
        frequency_hz=args.frequency_hz, epsilon_0_F_per_m=EPSILON_0,
        material_statistics=materials, moisture_validation=moisture,
        loss_definition={
            'epsilon_prime': 'epsilon_inf + sum(delta_epsilon / (1 + (omega*tau)^2))',
            'sigma_effective_s_m': 'sigma_base + epsilon_0*omega*sum(delta_epsilon*omega*tau / (1 + (omega*tau)^2))',
            'interpretation': 'Effective conductivity includes Debye dielectric losses at the stated frequency. Base sigma alone is not total Peplinski loss. Neither conductivity alone equals propagation attenuation.'},
        water_parameter_semantics={
            'native_peplinski': 'water_min/max are hard limits for the soil-moisture material palette; represented values are bin centers.',
            'custom_crim_archie': 'Bounds are converted to nominal midpoint and uniform-range SD, then to saturation mean/SD using nominal porosity. Gaussian correlated porosity and saturation fields yield theta=phi*S; this theta need not stay inside the source interval.',
            'physical_clipping': 'Generator clips porosity to[0.03,0.8] and saturation to[0.001,1]; thus0<=theta<=porosity.',
            'realization_mean': 'Nominal means are field-generation parameters, not enforced empirical means within each irregular layer mask. Porosity/saturation correlation also affects theta.',
            'recommended_caption_ru': 'Номинальные настройки и фактические статистики реализации: Peplinski использует жёсткий диапазон влажности; модель полей переводит этот диапазон в среднее и разброс.'},
        limitations=['Both material models are assumptions, not measurements of a particular park.',
            'Compare epsilon_prime and sigma_effective at a common stated frequency and with voxel weighting.',
            'Different loss laws and moisture distributions mean this is not an isolated comparison of spatial-field geometry.'],
        inputs_read='Frozen scene global.h5, global_materials.json, custom_fields.npz, soil.json only; no B-scan files opened.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),
        material_statistics=materials,
        theta_exceeds_porosity_count=moisture['theta_exceeds_porosity_count'],
        checked_soil_voxels=moisture['checked_soil_voxels']), indent=2))


if __name__ == '__main__':
    main()
