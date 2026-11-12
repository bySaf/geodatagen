import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('experiment',ROOT/'tools/experiment.py')
experiment=importlib.util.module_from_spec(spec);spec.loader.exec_module(experiment)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='gpr-experiment-test-')
        self.root=Path(self.tmp.name)
        self.source=json.loads((ROOT/'examples/scenes/park-pilot.json').read_text())

        self.source['trajectory']=json.loads((ROOT/'examples/trajectories/pilot-line.json').read_text())

    def tearDown(self):self.tmp.cleanup()

    def configuration(self,value=None):
        path=self.root/'settings.json';path.write_text(json.dumps(self.source if value is None else value))
        return experiment.settings(path,'native_peplinski')

    def test_protects_completed_datasets_and_ancestors(self):
        for p in [ROOT,ROOT/'data/final/new',ROOT/'user_review/realism/pilot']:
            with self.assertRaises(ValueError):experiment.destination(p)
        self.assertEqual(experiment.destination(self.root/'new'),(self.root/'new').resolve())

    def test_multiple_materials_and_empty_background(self):
        self.source['objects']=[dict(type=kind,center_m=[x,.6,.45],radius_m=.06)
                                for kind,x in [('metal',.7),('air',1.2),('dielectric',1.7)]]
        cfg,tx,rx=self.configuration()
        self.assertEqual([v['material'] for v in cfg['experiment_objects']],['pec','free_space','inclusion'])
        self.assertEqual(len(tx),29)
        self.source['objects']=[]
        self.assertEqual(self.configuration()[0]['experiment_objects'],[])

    def test_invalid_surface_overlap_and_subvoxel_targets_rejected(self):
        cases=[ [dict(type='metal',center_m=[1.2,.6,.79],radius_m=.06)],
                [dict(type='metal',center_m=[1.2,.6,.45],radius_m=.01)],
                [dict(type='metal',center_m=[1.2,.6,.45],radius_m=.1)]*2 ]
        for objects in cases:
            self.source['objects']=objects
            with self.assertRaises(ValueError):self.configuration()

    def test_changed_config_cannot_overwrite_frozen_experiment(self):
        cfg,tx,rx=self.configuration();folder=self.root/'frozen';folder.mkdir()
        (folder/'config.json').write_text(json.dumps(cfg))
        changed=dict(cfg,seed=cfg['seed']+1)
        with patch.object(experiment,'prepare_scene') as create:
            with self.assertRaises(ValueError):experiment.prepare(folder,changed,tx,rx)
            create.assert_not_called()

    def test_status_does_not_read_hdf5_while_running(self):
        folder=self.root/'running';folder.mkdir()
        (folder/'progress.json').write_text(json.dumps(dict(completed=2,total=29)))
        with experiment.locked(folder),patch.object(experiment.h5py,'File') as h5:
            result=experiment.status(folder)
            self.assertTrue(result['running']);self.assertEqual(result['completed'],2)
            h5.assert_not_called()


if __name__=='__main__':unittest.main()
