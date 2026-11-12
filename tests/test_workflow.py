"""Portable export checks: preserve voxel axes, dispersion and experiment isolation."""
import base64
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import geodatagen
import experiment_outputs as outputs


class WorkflowTests(unittest.TestCase):
    def test_names_cannot_escape_output_folder(self):
        for name in ('../other', '/tmp/output', 'x/y', '.', 'x y', ''):
            with self.assertRaises(ValueError):geodatagen.experiment_path(name)
        self.assertEqual(geodatagen.experiment_path('park-02').parent, ROOT / 'experiments')

    def test_symlink_cannot_redirect_experiment(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(geodatagen, 'ROOT', Path(tmp)):
            base = Path(tmp) / 'experiments'; base.mkdir()
            (base / 'existing').mkdir()
            (base / 'alias').symlink_to(base / 'existing', target_is_directory=True)
            with self.assertRaises(ValueError):geodatagen.experiment_path('alias')

    def test_vti_preserves_cell_order_and_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'scene.vti'
            a = np.arange(24, dtype=np.float32).reshape(2,3,4)
            outputs.write_vti(p, {'epsilon_real': a}, [.02,.03,.04], 600e6)
            tree = ET.parse(p)
            image = tree.find('ImageData')
            self.assertEqual(image.attrib['WholeExtent'], '0 2 0 3 0 4')
            node = tree.find('.//CellData/DataArray')
            raw = base64.b64decode(node.text)
            self.assertEqual(struct.unpack('<Q', raw[:8])[0], a.nbytes)
            actual = np.frombuffer(raw[8:], dtype='<f4').reshape(a.shape, order='F')
            np.testing.assert_array_equal(actual, a)

    def test_material_values_and_objects_follow_exported_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); scene = root/'scene'; scene.mkdir()
            cfg = {'source': {'frequency_hz': 600e6}, 'grid': {'surface_z': .2}}
            (root/'config.json').write_text(json.dumps(cfg))
            (scene/'soil.json').write_text(json.dumps({'geometry': 'global.h5', 'materials': 'materials.json'}))
            materials = {
                'soil': {'name': 'soil', 'model': 'debye', 'base': {'relative_permittivity': 4, 'electric_conductivity_s_per_m': .01}, 'poles': [{'relative_permittivity_difference': 8, 'relaxation_time_s': 1e-10}]},
                'air': {'name': 'free_space', 'model': 'constant', 'base': {'relative_permittivity': 1, 'electric_conductivity_s_per_m': 0}},
                'metal': {'name': 'pec', 'model': 'constant', 'base': {'relative_permittivity': 1, 'electric_conductivity_s_per_m': 'inf'}},
            }
            (scene/'materials.json').write_text(json.dumps({'materials': materials}))
            ids = np.array([[[0,1,1]], [[2,0,1]]], dtype=np.int16)
            with h5py.File(scene/'global.h5', 'w') as h:
                h['data'] = ids; h['material_keys'] = np.asarray(list(materials), dtype='S5'); h.attrs['dx_dy_dz'] = [.1,.1,.1]
            _, fields, _, _, _ = outputs.geometry_fields(root)
            expected = 4 + 8/(1+(2*np.pi*600e6*1e-10)**2)
            self.assertAlmostEqual(float(fields['epsilon_real'][0,0,0]), expected, places=5)
            self.assertGreater(fields['conductivity_effective_S_m'][0,0,0], .01)
            self.assertTrue(np.isnan(fields['epsilon_real'][1,0,0]))
            self.assertEqual(fields['object_type'][1,0,0], 1)
            self.assertEqual(fields['object_type'][0,0,1], 2)
            self.assertEqual(fields['object_type'][0,0,2], 0)

    def test_docs_command_does_not_import_solver_or_open_when_requested(self):
        with patch.object(geodatagen.webbrowser, 'open') as browser:
            self.assertEqual(geodatagen.main(['docs', '--no-open']), 0)
            browser.assert_not_called()


if __name__ == '__main__':unittest.main()
