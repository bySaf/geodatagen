"""Small frozen fixtures exercise real crop/cache I/O without an expensive solver."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from src import experiment_solver as solver
from src.soil import sha256
import experiment
import experiment_outputs as outputs


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.cfg = dict(schema_version=2, seed=1, generator='native_peplinski', experiment_objects=[],
                        grid=dict(domain=[7.,4.8,1.2], dx=.2, solver_dx=.1, pml_thickness=.2, surface_z=.8, time_window=1e-8, threads=1),
                        source=dict(waveform='ricker', frequency_hz=600e6, moment_Am=.02, height_m=.1),
                        profile=dict(n=3, step_m=.2, y_m=1.), simulation=dict(mode='moving_window', window_xy=[1.2,1.2]))
        tx = np.array([[1.,.8,.9], [1.,1.,.9], [1.4,1.2,.9]])
        self.positions = solver.normalized_positions(dict(tx_m=tx, rx_m=tx+[.1,0,0], distance_m=[0,.2,0], route_id=[0,0,1], segment_id=[0,1,0], route_names=['route α','route β']))
        scene = self.root/'scene'; scene.mkdir()
        with h5py.File(scene/'global.h5', 'w') as h:
            h['data'] = np.arange(35*24*6, dtype=np.int16).reshape(35,24,6)
            h['material_keys'] = np.asarray(['air'], dtype='S3')
            h.attrs['dx_dy_dz'] = [.2]*3
            h.attrs['MaterialDatabase'] = 'materials'; h.attrs['MaterialDatabaseSchemaVersion'] = 1
        (scene/'materials.json').write_text('{"materials": {}}')
        self.meta = dict(dx=.2, domain=[7.,4.8,1.2], shape=[35,24,6], geometry='global.h5', materials='materials.json',
                         geometry_sha256=sha256(scene/'global.h5'), materials_sha256=sha256(scene/'materials.json'))
        (scene/'soil.json').write_text(json.dumps(self.meta))
        (self.root/'config.json').write_text(json.dumps(self.cfg))
        np.savez_compressed(self.root/'positions.npz', **self.positions)
        self.calls = []

    def tearDown(self):self.tmp.cleanup()

    def fake_run(self, inputfile, **kwargs):
        path = Path(inputfile); self.calls.append(path.read_text())
        with h5py.File(path.with_suffix('.h5'), 'w') as h:
            h['rxs/rx1/Ey'] = np.arange(1,17, dtype=np.float32)
            h.attrs['dt'] = 1e-10; h.attrs['gprMax'] = 'test'

    def run_fake(self, limit=None):
        fake = types.SimpleNamespace(run=self.fake_run)
        with patch.dict(sys.modules, {'gprMax': fake}), patch.object(solver, 'source_values', side_effect=lambda cfg, dt, n: ((np.arange(n)+.5)*dt, np.ones(n))):
            return solver.run_survey(self.cfg, self.positions, self.root, limit=limit)

    def test_moving_refines_only_crop_and_reuses_checked_cache(self):
        origin = solver.crop_origins(self.cfg, self.positions)[0]
        geometry, _, domain = solver.prepare_geometry(self.cfg, self.root, self.meta, origin)
        with h5py.File(geometry) as h:
            self.assertEqual(h['data'].shape, (12,12,12))
            np.testing.assert_allclose(h.attrs['dx_dy_dz'], [.1]*3)
        self.assertEqual(domain, [1.2,1.2,1.2]); self.assertFalse((self.root/'run/geometry.h5').exists())
        with patch.object(solver, 'crop_scene') as crop:
            self.assertEqual(solver.prepare_geometry(self.cfg, self.root, self.meta, origin)[0], geometry)
            crop.assert_not_called()
        with h5py.File(geometry, 'r+') as h:h['data'][0,0,0] = -1
        with patch.object(solver, 'crop_scene', wraps=solver.crop_scene) as crop:
            solver.prepare_geometry(self.cfg, self.root, self.meta, origin); crop.assert_called_once()

    def test_limit_stop_resume_metadata_and_frozen_inputs(self):
        before = {name: sha256(self.root/name) for name in ('config.json','positions.npz','scene/soil.json')}
        output = self.run_fake(limit=1)
        with h5py.File(output) as h:
            np.testing.assert_array_equal(h['complete'][:], [True,False,False])
            np.testing.assert_allclose(h['distance_m'][:], [0,.2,0])
            self.assertEqual(h['route_names'].asstr()[1], 'route β')
            self.assertFalse(h.attrs['fixed_domain']); self.assertEqual(h.attrs['geometry_sha256'], self.meta['geometry_sha256'])
            first = h['Ey'][0].copy()
        (self.root/'STOP').touch(); self.run_fake(); self.assertEqual(len(self.calls), 1)
        (self.root/'STOP').unlink(); self.run_fake(limit=1); self.assertEqual(len(self.calls), 2)
        self.run_fake(); self.run_fake(); self.assertEqual(len(self.calls), 3)
        with h5py.File(output) as h:
            self.assertTrue(h['complete'][:].all()); np.testing.assert_array_equal(h['Ey'][0], first)
        self.assertEqual(before, {name: sha256(self.root/name) for name in before})
        self.assertIn('#hertzian_dipole: y', self.calls[0]); self.assertIn('#waveform: ricker 0.19999999999999998 600000000.0 pulse', self.calls[0])

    def test_sweep_preserves_units_and_safe_route_arrays(self):
        path = self.run_fake()
        solver.effective_sfcw(path, fmin=100e6, fmax=300e6, df=100e6)
        with np.load(self.root/'effective_sfcw.npz', allow_pickle=False) as data:
            np.testing.assert_allclose(data['distance_m'], [0,.2,0])
            self.assertEqual(data['route_names'][1], 'route β')
            self.assertEqual(data['transfer'].shape, (3,3))
            self.assertTrue(np.isfinite(data['response']).all())
        meta = json.loads((self.root/'effective_sfcw.json').read_text())
        self.assertEqual(meta['transfer_units'], '(V/m)/(A*m)')
        self.assertEqual(meta['response_units'], '(V/m)/(A*m*s)')

    def test_scene_tampering_rejected_before_solver(self):
        with h5py.File(self.root/'scene/global.h5', 'r+') as h:h['data'][0,0,0] = -1
        with self.assertRaisesRegex(ValueError, 'geometry/materials'):self.run_fake()
        self.assertEqual(self.calls, [])

    def test_bad_crop_is_rejected_before_preparation(self):
        bad = copy.deepcopy(self.positions); bad['tx_m'][:,0] = .2; bad['rx_m'][:,0] = .3
        with patch.object(experiment, 'prepare_scene') as create:
            with self.assertRaisesRegex(ValueError, 'crop outside'):experiment.prepare(self.root/'bad', self.cfg, bad)
            create.assert_not_called()
        bad = copy.deepcopy(self.positions); bad['rx_m'][:,0] += 1.
        with self.assertRaisesRegex(ValueError, 'moving-window PML'):solver.validate_positions(self.cfg, bad)

    def test_legacy_digest_and_saved_coordinate_resume(self):
        self.cfg.pop('schema_version'); self.cfg['simulation'] = dict(mode='fixed')
        expected = hashlib.sha256(json.dumps(self.cfg, sort_keys=True).encode()).hexdigest()
        self.assertEqual(solver.signature(self.cfg), expected)
        with h5py.File(self.root/'bscan.h5', 'w') as h:
            h.attrs['config_sha256'] = expected
            h['complete'] = [False]*3; h['seconds'] = [0.]*3
            h['tx_m'] = self.positions['tx_m']; h['rx_m'] = self.positions['rx_m']
        self.run_fake(limit=1)
        with h5py.File(self.root/'bscan.h5') as h:self.assertEqual(h.attrs['config_sha256'], expected)
        changed = copy.deepcopy(self.positions); changed['rx_m'][0,0] += .1
        with self.assertRaisesRegex(ValueError, 'rx_m'):
            solver.run_survey(self.cfg, changed, self.root, limit=1)

    def test_frozen_manifest_catches_position_changes(self):
        (self.root/'frozen.json').write_text(json.dumps({'positions.npz': sha256(self.root/'positions.npz')}))
        experiment.verify_frozen(self.root)
        np.savez_compressed(self.root/'positions.npz', tx_m=self.positions['tx_m'])
        with self.assertRaisesRegex(ValueError, 'positions.npz'):experiment.verify_frozen(self.root)

    def test_route_background_distance_plot_csv_and_slice(self):
        a = np.array([[1.,2.], [3.,4.], [100.,200.]])
        bg = experiment.route_background(a, np.ones(3, dtype=bool), self.positions['route_id'])
        np.testing.assert_allclose(bg, [[-1,-1],[1,1],[0,0]])
        self.run_fake(limit=1); experiment.plot(self.root)
        self.assertTrue((self.root/'bscan.png').is_file())
        experiment.write_trajectory_csv(self.root, self.positions)
        content = (self.root/'trajectory.csv').read_text()
        self.assertIn('requested_rx_z_m', content); self.assertIn('route α', content)
        self.assertAlmostEqual(outputs.representative_y(self.root, self.cfg), 1.)


if __name__ == '__main__':unittest.main()
