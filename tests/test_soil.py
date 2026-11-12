"""Integration checks for immutable geometry and dispersive material transport."""
from pathlib import Path
import json
import tempfile
import unittest

import h5py
import numpy as np
from src.soil import prepare_scene, crop_scene


class SoilIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='gpr-soil-test-')
        cls.root = Path(cls.tmp.name)
        cls.preset = dict(name='test', layers=[dict(thickness=.2, sand=.5, clay=.2,
            bulk_density=1.6, water_min=.05, water_max=.15, materials=5)])
        cls.grid = dict(domain=[.4,.4,.3], dx=.01, surface_z=.2, time_window=2e-9, threads=2)
        cls.objects = ['#sphere: 0.2 0.2 0.12 0.03 pec n']
        for name in ('a','b'):
            prepare_scene(cls.root/name, cls.preset, cls.grid, 42, cls.objects)
        cls.crop = crop_scene(cls.root/'a', cls.root/'crop'/'geometry.h5', [.05,.05], [.3,.3])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_seed_reproduces_entire_scene(self):
        with h5py.File(self.root/'a'/'global.h5') as a, h5py.File(self.root/'b'/'global.h5') as b:
            np.testing.assert_array_equal(a['data'][:], b['data'][:])
            np.testing.assert_array_equal(a['material_keys'][:], b['material_keys'][:])

    def test_crop_has_exact_same_voxels_and_object_labels(self):
        with h5py.File(self.root/'a'/'global.h5') as a, h5py.File(self.root/'crop'/'geometry.h5') as c:
            np.testing.assert_array_equal(a['data'][5:35,5:35,:], c['data'][:])
            keys = [x.decode() for x in c['material_keys'][:]]
            pec = next(i for i,k in enumerate(keys) if k.endswith('_pec'))
            self.assertGreater(np.count_nonzero(c['data'][:]==pec), 0)
            self.assertEqual(int(c['data'][15,15,12]), pec)

    def test_material_database_preserves_dispersion_exactly(self):
        name = self.crop['material_database']+'.json'
        self.assertEqual((self.root/'a'/name).read_bytes(), (self.root/'crop'/name).read_bytes())
        db = json.loads((self.root/'crop'/name).read_text())
        dispersive = [m for m in db['materials'].values() if m['model']=='debye']
        self.assertTrue(dispersive)
        self.assertTrue(all(m['poles'][0]['relaxation_time_s']>0 for m in dispersive))

    def test_out_of_bounds_crop_rejected(self):
        with self.assertRaises(ValueError):
            crop_scene(self.root/'a',self.root/'bad.h5', [.2,.2], [.3,.3])

class CustomSoilIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='gpr-custom-soil-test-')
        cls.root = Path(cls.tmp.name)
        cls.preset = dict(name='custom_test', layers=[
            dict(thickness=.08, er=6, sigma=.001),
            dict(thickness=.22, sand=.5, clay=.2, bulk_density=1.5,
                 water_min=.1, water_max=.2,
                 custom=dict(water_conductivity_s_m=.12, porosity_sd=.01))])
        cls.grid = dict(domain=[.4,.4,.4], dx=.01, surface_z=.3,
                        time_window=2e-9, threads=2)
        cls.objects = ['#sphere: 0.2 0.2 0.12 0.03 pec n']
        for name in ('a','b'):
            prepare_scene(cls.root/name, cls.preset, cls.grid, 29, cls.objects,
                          generator='custom_crim_archie')
        cls.crop = crop_scene(cls.root/'a', cls.root/'crop'/'geometry.h5',
                              [.05,.05], [.3,.3])

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_reproducible_geometry_and_real_object(self):
        with h5py.File(self.root/'a'/'global.h5') as a, h5py.File(self.root/'b'/'global.h5') as b:
            np.testing.assert_array_equal(a['data'][:],b['data'][:])
            np.testing.assert_array_equal(a['material_keys'][:],b['material_keys'][:])
            keys = [k.decode() for k in a['material_keys'][:]]
            self.assertTrue(keys[int(a['data'][20,20,12])].endswith('_pec'))
        with np.load(self.root/'a'/'custom_fields.npz') as f:
            self.assertEqual(int(f['layer_id'][20,20,12]),-1)
            self.assertTrue(np.isnan(f['water_fraction'][20,20,12]))

    def test_crop_and_constant_material_retained(self):
        with h5py.File(self.root/'a'/'global.h5') as source, h5py.File(self.root/'crop'/'geometry.h5') as crop:
            np.testing.assert_array_equal(source['data'][5:35,5:35,:],crop['data'][:])
            self.assertIn('MaterialDatabase',crop.attrs)
            keys = [k.decode() for k in source['material_keys'][:]]
            lid = int(source['data'][20,20,25])
        db = json.loads((self.root/'a'/'global_materials.json').read_text())['materials']
        self.assertEqual(db[keys[lid]]['base']['relative_permittivity'],6)
        self.assertAlmostEqual(db[keys[lid]]['base']['electric_conductivity_s_per_m'],.001,places=9)
        self.assertEqual((self.root/'a'/'global_materials.json').read_bytes(),
                         (self.root/'crop'/'global_materials.json').read_bytes())

    def test_report_uses_custom_moisture_instead_of_debye_inversion(self):
        from src.report import soil_statistics
        meta, _, _, layers = soil_statistics(self.root/'a',4e8)
        self.assertEqual(meta['generator'],'custom_crim_archie')
        self.assertTrue(layers[0]['constant_material'])
        self.assertGreater(layers[1]['theta_std'],0)
        self.assertTrue(layers[1]['finite'])
        self.assertNotIn('theta_bin_reconstruction_error',layers[1])
        self.assertTrue(all(v>.2 for v in layers[1]['lag1_xyz']))

    def test_explicit_conductivity_and_known_generator_required(self):
        preset = dict(name='bad',layers=[dict(thickness=.3, sand=.5, clay=.2,
                         bulk_density=1.5, water_min=.1, water_max=.2)])
        with self.assertRaisesRegex(ValueError,'water_conductivity'):
            prepare_scene(self.root/'bad',preset,self.grid,1,generator='custom_crim_archie')
        with self.assertRaisesRegex(ValueError,'Unknown soil generator'):
            prepare_scene(self.root/'bad',preset,self.grid,1,generator='typo')


if __name__=='__main__': unittest.main()
