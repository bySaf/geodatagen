"""Material/signals exports retain physical conventions and portable array layouts."""
import base64
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('exports',ROOT/'tools/experiment_outputs.py')
exports=importlib.util.module_from_spec(spec);spec.loader.exec_module(exports)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        scene=self.root/'scene';scene.mkdir()
        self.cfg=dict(grid=dict(surface_z=.2,domain=[.4,.3,.3]),source=dict(frequency_hz=600e6,height_m=.02),
            preset=dict(layers=[dict(thickness=.2,er=4,sigma=.01)]),profile=dict(n=3,y_m=.15),
            generator='native_peplinski',seed=1,experiment_objects=[dict(type='metal',material='pec',center_m=[.2,.15,.1],radius_m=.04)])
        (self.root/'config.json').write_text(json.dumps(self.cfg))
        self.meta=dict(geometry='geometry.h5',materials='materials.json',layers=[dict(z_bottom_m=0,z_top_m=.2)])
        (scene/'soil.json').write_text(json.dumps(self.meta))
        db={'soil':dict(name='soil',model='debye',base=dict(relative_permittivity=4,electric_conductivity_s_per_m=.01,relative_permeability=1.2,magnetic_conductivity_s_per_m=.003),poles=[dict(relative_permittivity_difference=8,relaxation_time_s=1e-10)]),
            'pec':dict(name='pec',model='constant',base=dict(relative_permittivity=1,electric_conductivity_s_per_m='inf')),
            'air':dict(name='free_space',model='constant',base=dict(relative_permittivity=1,electric_conductivity_s_per_m=0))}
        (scene/'materials.json').write_text(json.dumps(dict(materials=db)))
        ids=np.zeros((4,3,3),dtype='i4');ids[:,:,2]=2;ids[2,1,1]=1
        with h5py.File(scene/'geometry.h5','w') as h:
            h['data']=ids;h['material_keys']=np.asarray(list(db),dtype='S5');h.attrs['dx_dy_dz']=[.1]*3
        tx=np.array([[.1,.1,.22],[.2,.1,.22],[.2,.2,.22]])
        np.savez_compressed(self.root/'positions.npz',tx_m=tx,rx_m=tx+[.02,0,0],distance_m=[0,.1,.2],route_id=[0,0,0],segment_id=[0,1,1],route_names=np.array(['turn']))

    def tearDown(self):self.tmp.cleanup()

    def test_complex_materials_and_native_layer_labels(self):
        _,f,_,freq,_=exports.geometry_fields(self.root)
        omega=2*np.pi*freq;wt=omega*1e-10
        self.assertAlmostEqual(float(f['epsilon_real'][0,0,0]),4+8/(1+wt*wt),places=5)
        self.assertAlmostEqual(float(f['epsilon_imag'][0,0,0]),8*wt/(1+wt*wt)+.01/(omega*exports.EPSILON_0),places=5)
        np.testing.assert_allclose(f['epsilon_imag']*omega*exports.EPSILON_0,f['conductivity_effective_S_m'],rtol=1e-6)
        self.assertAlmostEqual(float(f['permeability_relative'][0,0,0]),1.2,places=6)
        self.assertEqual(f['layer_id'][0,0,0],1);self.assertEqual(f['layer_id'][2,1,1],-1)
        self.assertEqual(f['layer_id'][0,0,2],0);self.assertNotIn('water_fraction',f)
        self.assertTrue(np.isnan(f['epsilon_real'][2,1,1]));self.assertEqual(f['object_mask'][2,1,1],1)

    def test_numpy_centres_and_partial_signals(self):
        with h5py.File(self.root/'bscan.h5','w') as h:
            h['complete']=[True,False,True];h['Ey']=np.array([[1,2],[99,99],[3,4]],dtype='f4')
            h['time_s']=[0,1e-10];h['source_time_s']=[.5e-10,1.5e-10];h['source_current_A']=[0,1]
        exports.export_numpy(self.root)
        with np.load(self.root/'exports/geometry.npz',allow_pickle=False) as g:
            np.testing.assert_allclose(g['x_m'],[.05,.15,.25,.35]);self.assertEqual(g['epsilon_real'].shape,(4,3,3))
            self.assertEqual(g['material_keys'].tolist(),['soil','pec','air'])
            for k in g.files:self.assertFalse(g[k].dtype.hasobject)
        with np.load(self.root/'exports/signals.npz',allow_pickle=False) as h:
            self.assertTrue(np.isnan(h['Ey'][1]).all());self.assertEqual(h['route_names'][0],'turn')
            self.assertEqual(h['Ey'][2,1],4)
        meta=json.loads((self.root/'exports/metadata.json').read_text())
        self.assertEqual(meta['signals']['complete_count'],2)
        self.assertIn('epsilon_imag',meta['fields'])
        self.assertEqual((self.root/'exports/materials.json').read_bytes(),(self.root/'scene/materials.json').read_bytes())

    def test_signal_refresh_reuses_geometry_export(self):
        exports.export_numpy(self.root)
        before=(self.root/'exports/geometry.npz').read_bytes()
        with h5py.File(self.root/'bscan.h5','w') as h:
            h['complete']=[False,False,False]
        with patch.object(exports,'geometry_fields',side_effect=AssertionError('Volume should remain cached')):
            exports.export_numpy(self.root,signals_only=True)
        self.assertEqual(before,(self.root/'exports/geometry.npz').read_bytes())
        with np.load(self.root/'exports/signals.npz',allow_pickle=False) as h:
            self.assertEqual(h['Ey'].shape,(3,0));self.assertFalse(h['complete'].any())

    def test_no_fake_signals_before_calculation(self):
        exports.export_numpy(self.root)
        self.assertFalse((self.root/'exports/signals.npz').exists())
        self.assertIsNone(json.loads((self.root/'exports/metadata.json').read_text())['signals'])

    def test_custom_continuous_fields_mask_targets(self):
        self.meta['custom_fields']='custom.npz';(self.root/'scene/soil.json').write_text(json.dumps(self.meta))
        shape=(4,3,3);retained=np.ones(shape,dtype=bool);retained[:,:,2]=False;retained[2,1,1]=False
        np.savez(self.root/'scene/custom.npz',layer_id=np.ones(shape,dtype='i4'),retained_soil_mask=retained,
            porosity=np.full(shape,.4),saturation=np.full(shape,.5),water_fraction=np.full(shape,.2),
            epsilon_r=np.full(shape,7),sigma_s_m=np.full(shape,.01))
        _,f,_,_,_=exports.geometry_fields(self.root)
        self.assertTrue(np.isnan(f['water_fraction'][2,1,1]));self.assertTrue(np.isnan(f['porosity'][0,0,2]))
        self.assertAlmostEqual(float(f['water_fraction'][0,0,0]),.2,places=6)
        self.assertEqual(f['continuous_epsilon_real'][0,0,0],7)

    def test_vtp_overlays_have_valid_connectivity(self):
        exports.export_overlays(self.root,self.cfg)
        for name,cellkind in [('trajectory.vtp','Lines'),('objects.vtp','Polys')]:
            tree=ET.parse(self.root/'scene'/name);p=tree.find('.//Piece');n=int(p.attrib['NumberOfPoints'])
            node=p.find(f'{cellkind}/DataArray[@Name="connectivity"]')
            raw=base64.b64decode(node.text);self.assertEqual(struct.unpack('<Q',raw[:8])[0],len(raw)-8)
            indices=np.frombuffer(raw[8:],dtype='<i8')
            self.assertTrue((indices>=0).all());self.assertTrue((indices<n).all())
        self.assertEqual(int(ET.parse(self.root/'scene/trajectory.vtp').find('.//Piece').attrib['NumberOfLines']),2)

    def test_init_preview_contains_layers_objects_routes_without_signals(self):
        exports.export_geometry(self.root)
        exports.report(self.root,dict(completed=0,total=3,running=False))
        text=(self.root/'index.html').read_text()
        for token in ['soil_sections.png','Слои грунта','Объекты','Маршруты','Сигналы ещё не рассчитаны','exports/geometry.npz']:
            self.assertIn(token,text)
        self.assertNotIn('href="exports/signals.npz"',text)
        self.assertTrue((self.root/'scene/trajectory.vtp').exists())


if __name__=='__main__':unittest.main()
