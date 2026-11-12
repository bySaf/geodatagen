import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial.distance import pdist
from src.sampling import sample_coordinates
from src.experiment_config import load_settings, random_objects

ROOT = Path(__file__).resolve().parents[1]


class SamplingTests(unittest.TestCase):
    def test_methods_reproducible_bounded_and_poisson_separated(self):
        for method in ('uniform', 'sobol', 'poisson_disk'):
            kw = {'min_distance_m': .2} if method == 'poisson_disk' else {}
            p = sample_coordinates([[1,3], [2,4]], 32, 42, method, **kw)
            np.testing.assert_array_equal(p, sample_coordinates([[1,3],[2,4]], 32,42,method,**kw))
            self.assertTrue(np.all(p >= [1,2]) and np.all(p <= [3,4]))
            self.assertFalse(np.array_equal(p,sample_coordinates([[1,3],[2,4]],32,43,method,**kw)))
            if method == 'poisson_disk':self.assertGreaterEqual(pdist(p).min(), .2)

    def test_sobol_balanced_quadrants(self):
        p=sample_coordinates([[0,1],[0,1]],32,42,'sobol')
        for x in (0,1):
            for y in (0,1):self.assertEqual(np.sum(((p[:,0]>=.5)==bool(x)) & ((p[:,1]>=.5)==bool(y))),8)

    def test_impossible_or_invalid_requests_fail(self):
        for args in [(0,42,'uniform',None),(3,-1,'uniform',None),(2,42,'poisson_disk',3),(2,42,'poisson_disk',0)]:
            with self.assertRaises(ValueError):sample_coordinates([[0,1],[0,1]],*args)

    def test_object_minimum_distance_and_random_route_replay(self):
        cfg,pos=load_settings(ROOT/'examples/scenes/park-poisson.json',trajectory_path=ROOT/'examples/trajectories/random-poisson.json')
        centers=np.array([o['center_m'] for o in cfg['experiment_objects']])
        self.assertGreaterEqual(pdist(centers).min(),.5)
        self.assertEqual(len(pos['tx_m']),32)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'scene.json';path.write_text(json.dumps(cfg))
            _,again=load_settings(path)
            np.testing.assert_array_equal(pos['tx_m'],again['tx_m'])

    def test_every_scene_number_has_provenance(self):
        def numeric(value,path=''):
            if isinstance(value,dict):
                for key,item in value.items():
                    if key!='_provenance':yield from numeric(item,path+'/'+key)
            elif isinstance(value,list):
                for i,item in enumerate(value):yield from numeric(item,path+'/'+str(i))
            elif isinstance(value,(int,float)) and not isinstance(value,bool):yield path
        for path in (ROOT/'examples/scenes').glob('*.json'):
            scene=json.loads(path.read_text())
            self.assertEqual(set(numeric(scene)),set(scene['_provenance']),path.name)

    def test_fixed_depth_and_radius_are_valid_for_all_object_samplers(self):
        scene = json.loads((ROOT/'examples/scenes/park-random.json').read_text())
        for method in ('uniform', 'sobol', 'poisson_disk'):
            spec = dict(method=method, bounds_m=[[1,5],[1,3]], rules=[
                dict(type='metal',count=3,radius_m=[.05,.05],depth_m=[.35,.35])])
            if method == 'poisson_disk':spec['min_distance_m'] = .5
            objects, _ = random_objects(spec, scene['grid'], 42)
            self.assertEqual(len(objects), 3)
            for obj in objects:
                self.assertEqual(obj['radius_m'], .05)
                self.assertAlmostEqual(obj['center_m'][2], scene['grid']['surface_z']-.35)


if __name__ == '__main__':unittest.main()
