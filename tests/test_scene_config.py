"""Scene/route resolution must be deterministic before geometry generation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.experiment_config import load_settings


ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT / "examples" / "scenes"
ROUTES = ROOT / "examples" / "trajectories"


class SceneConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="scene-config-test-")
        self.directory = Path(self.temporary.name)
        self.pilot = json.loads((SCENES / "park-pilot.json").read_text())

    def tearDown(self):
        self.temporary.cleanup()

    def load(self, scene, route="pilot-line.json", **kwargs):
        path = self.directory / "scene.json"
        path.write_text(json.dumps(scene))
        return load_settings(path, trajectory_path=ROUTES / route, **kwargs)

    def test_random_objects_are_reproducible_and_independent_of_route(self):
        scene = json.loads((SCENES / "park-random.json").read_text())
        first, _ = self.load(scene, "line.json")
        second, _ = self.load(scene, "snake.json")
        self.assertEqual(first["experiment_objects"], second["experiment_objects"])
        self.assertEqual(len({obj["id"] for obj in first["experiment_objects"]}),
                         len(first["experiment_objects"]))
        self.assertEqual(sum(obj["type"] == "metal" for obj in first["experiment_objects"]), 3)
        scene["seed"] += 1
        same_objects, _ = self.load(scene, "line.json")
        self.assertEqual(first["experiment_objects"], same_objects["experiment_objects"])
        scene["object_generation"]["seed"] += 1
        different, _ = self.load(scene, "line.json")
        self.assertNotEqual(first["experiment_objects"], different["experiment_objects"])

    def test_explicit_and_generated_objects_are_mutually_exclusive(self):
        scene = deepcopy(self.pilot)
        scene["objects"] = []
        scene["object_generation"] = {"rules": []}
        with self.assertRaisesRegex(ValueError, "objects OR object_generation"):
            self.load(scene)

    def test_invalid_geometry_is_rejected_before_simulation(self):
        changes = [
            lambda c: c["grid"].update(domain=[2.41, 1.2, 1.2]),
            lambda c: c["grid"].update(solver_dx=.015),
            lambda c: c["grid"].update(pml_thickness=.123),
            lambda c: c["grid"].update(surface_z=1.2),
            lambda c: c["preset"]["layers"][0].update(thickness=.12),
            lambda c: c["source"].update(height_m=.001),
        ]
        for change in changes:
            scene = deepcopy(self.pilot)
            change(scene)
            with self.subTest(scene=scene), self.assertRaises(ValueError):
                self.load(scene)

    def test_generator_override_preserves_scene_objects_source_and_route(self):
        model, model_route = self.load(self.pilot)
        native, native_route = self.load(self.pilot, mode="native_peplinski")
        self.assertEqual(model["generator"], "custom_crim_archie")
        self.assertEqual(native["generator"], "native_peplinski")
        self.assertIn("generator_fingerprint", model)
        self.assertEqual(native["experiment_objects"], model["experiment_objects"])
        self.assertEqual(native["source"], model["source"])
        self.assertEqual(native["preset"], model["preset"])
        np.testing.assert_array_equal(native_route["tx_m"], model_route["tx_m"])
        with self.assertRaisesRegex(ValueError, "generator"):
            self.load(self.pilot, mode="unknown")

    def test_separate_scene_and_csv_snapshot_load_without_original_inputs(self):
        scene_path = self.directory / "new-scene.json"
        scene_path.write_text(json.dumps(self.pilot))
        route_directory = self.directory / "survey"
        route_directory.mkdir()
        csv_path = route_directory / "measured.csv"
        csv_path.write_text("x_m,y_m\n.5,.6\n.7,.6\n.9,.7\n")
        route_path = route_directory / "route.json"
        route_path.write_text(json.dumps({"type": "csv", "file": "measured.csv", "step_m": .05}))
        cfg, positions = load_settings(scene_path, trajectory_path=route_path)
        self.assertEqual(cfg["trajectory"]["type"], "points")
        self.assertNotIn("file", cfg["trajectory"])
        self.assertEqual(cfg["profile"]["n"], len(positions["tx_m"]))
        frozen = self.directory / "frozen-config.json"
        frozen.write_text(json.dumps(cfg))
        for path in (scene_path, route_path, csv_path):
            path.unlink()
        replay_cfg, replay = load_settings(frozen)
        self.assertEqual(replay_cfg["experiment_objects"], cfg["experiment_objects"])
        for key in ("tx_m", "rx_m", "requested_tx_m", "requested_rx_m", "distance_m", "route_id", "segment_id"):
            np.testing.assert_array_equal(positions[key], replay[key], err_msg=key)

    def test_json_rejects_duplicate_keys_and_nonfinite_numbers(self):
        path = self.directory / "bad.json"
        for contents in ('{"seed": 1, "seed": 2}', '{"seed": NaN}', '{"seed": Infinity}'):
            path.write_text(contents)
            with self.subTest(contents=contents), self.assertRaises(ValueError):
                load_settings(path, trajectory_path=ROUTES / "pilot-line.json")


if __name__ == "__main__":
    unittest.main()
