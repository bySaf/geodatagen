import unittest
import tempfile
import json
from pathlib import Path
from src.layout import load_object_coordinates

class LayoutTests(unittest.TestCase):
    def test_explicit_coordinates_preserve_position_and_derive_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'coordinates.json'
            p.write_text(json.dumps({'park':[dict(type='metal_sphere', center_m=[2,1,.45],
                radius_m=.08, depth_center_m=999, depth_top_m=999)],'field':[]}))
            result=load_object_coordinates(p,['park','field'],[7,4.8,1.2],.9,control_scenes=['field'])
            o=result['park'][0]
            self.assertEqual(o['center_m'],[2,1,.45])
            self.assertEqual(o['material'],'pec')
            self.assertAlmostEqual(o['depth_center_m'],.45)
            self.assertAlmostEqual(o['depth_top_m'],.37)
            self.assertEqual(result['field'],[])
            # Exported canonical records are valid input, with identical results.
            p.write_text(json.dumps(result))
            self.assertEqual(result,load_object_coordinates(p,['park','field'],[7,4.8,1.2],.9))

    def test_explicit_coordinates_reject_invalid_geometry_and_metadata(self):
        base=dict(type='metal_sphere',center_m=[2,1,.45],radius_m=.08,id='target')
        invalid=[]
        for change in [dict(center_m=[float('nan'),1,.45]),dict(center_m=[True,1,.45]),
                       dict(center_m=[.1,1,.45]),dict(center_m=[2,1,.85]),
                       dict(center_m=[2,1,.22]),dict(radius_m=-1),dict(radius_m=float('inf')),
                       dict(type='unknown'),dict(material='free_space'),dict(id=''),dict(raduis_m=.1)]:
            invalid.append({'park':[dict(base,**change)],'field':[]})
        invalid += [{'park':[base],'field':[base]}, {'park':[base]},
                    {'park':[base],'field':[],'unknown':[]},
                    {'park':[base,dict(base,id='other',center_m=[2.15,1,.45])],'field':[]},
                    {'park':[base,dict(base,center_m=[3,1,.45])],'field':[]}]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'coordinates.json'
            for spec in invalid:
                with self.subTest(spec=spec):
                    p.write_text(json.dumps(spec))
                    with self.assertRaises(ValueError):
                        load_object_coordinates(p,['park','field'],[7,4.8,1.2],.9,control_scenes=['field'])
            p.write_text('{"park": [], "park": [], "field": []}')
            with self.assertRaises(ValueError):
                load_object_coordinates(p,['park','field'],[7,4.8,1.2],.9)
