"""Backend decisions are explicit; optimization preserves exact voxel indices."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from src import backends, experiment_solver as solver
from src import soil
from src.soil import crop_scene


class BackendTests(unittest.TestCase):
    def test_geometry_cache_is_shared_and_respects_valid_caller_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.dict(os.environ, {'MPLCONFIGDIR': str(root/'caller')}):
                env = soil._geometry_environment()
                self.assertEqual(Path(env['MPLCONFIGDIR']), (root/'caller').resolve())
            (root/'not-a-directory').write_text('occupied')
            with patch.dict(os.environ, {'MPLCONFIGDIR': str(root/'not-a-directory')}), patch.object(soil, '__file__', str(root/'src/soil.py')):
                first = soil._geometry_environment()['MPLCONFIGDIR']
                self.assertEqual(first, soil._geometry_environment()['MPLCONFIGDIR'])
                self.assertEqual(Path(first), (root/'.cache/matplotlib').resolve())

    def test_unavailable_explicit_gpu_raises_instead_of_falling_back(self):
        available = {name: dict(available=name == 'cpu', reason='test unavailable') for name in ('cpu','metal','cuda')}
        with patch.object(backends, 'probe_backends', return_value=available):
            self.assertEqual(backends.resolve_backend('auto'), 'cpu')
            with self.assertRaisesRegex(ValueError, 'unavailable'):backends.resolve_backend('metal')
            with self.assertRaisesRegex(ValueError, 'unavailable'):backends.resolve_backend('cuda')

    def test_auto_selects_a_discovered_device_and_options_match_api(self):
        available = {name: dict(available=name != 'cuda') for name in ('cpu','metal','cuda')}
        with patch.object(backends, 'probe_backends', return_value=available):
            self.assertEqual(backends.resolve_backend('auto'), 'metal')
        self.assertEqual(backends.solver_options('metal'), {'metal': True, 'gpu_precision': 'single'})
        self.assertEqual(backends.solver_options('cuda'), {'gpu': [0], 'gpu_precision': 'single'})
        self.assertEqual(backends.solver_options('cpu'), {})
        with self.assertRaises(ValueError):backends.resolve_backend('gpu')
        with self.assertRaises(ValueError):backends.solver_options('auto')

    def test_crop_refinement_matches_old_two_file_path_and_preserves_palette(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); scene = root/'scene'; scene.mkdir()
            values = np.arange(8*10*6, dtype=np.int32).reshape(8,10,6)
            with h5py.File(scene/'global.h5', 'w') as h:
                h['data'] = values; h['material_keys'] = np.asarray(['soil','pec'], dtype='S4')
                h.attrs['dx_dy_dz'] = [.1]*3
                h.attrs['MaterialDatabase'] = 'materials'; h.attrs['MaterialDatabaseSchemaVersion'] = 1
            (scene/'materials.json').write_text('{}')
            (scene/'soil.json').write_text(json.dumps(dict(dx=.1, shape=list(values.shape), domain=[.8,1.,.6], geometry='global.h5', materials='materials.json')))
            crop_scene(scene, root/'old/coarse.h5', [.2,.3], [.4,.5])
            solver._refine(root/'old/coarse.h5', root/'old/refined.h5', 2, .05)
            solver._refine(scene/'global.h5', root/'fused.h5', 2, .05, slices=np.s_[2:6,3:8,:])
            with h5py.File(root/'old/refined.h5') as old, h5py.File(root/'fused.h5') as fused:
                np.testing.assert_array_equal(old['data'][:], fused['data'][:])
                np.testing.assert_array_equal(old['material_keys'][:], fused['material_keys'][:])
                np.testing.assert_array_equal(old.attrs['dx_dy_dz'], fused.attrs['dx_dy_dz'])
                self.assertEqual(fused['data'].shape, (8,10,12))
                self.assertEqual(fused.attrs['MaterialDatabase'], 'materials')


if __name__ == '__main__':unittest.main()
