import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.trajectories import build_trajectory


ROOT = Path(__file__).resolve().parents[1]
GRID = dict(domain=[7, 4.8, 1.2], dx=.02, solver_dx=.01, surface_z=.8, pml_thickness=.12)
SOURCE = dict(height_m=.02, tx_rx_offset_m=.1)


class TrajectoryTests(unittest.TestCase):
    def build(self, spec, **kwargs):
        return build_trajectory(spec, kwargs.pop("grid", GRID), kwargs.pop("source", SOURCE), **kwargs)

    def assert_replays(self, route):
        replay = self.build(json.loads(json.dumps(route["spec"])))
        for key in ("tx_m", "rx_m", "requested_tx_m", "requested_rx_m", "distance_m", "route_id", "segment_id"):
            np.testing.assert_array_equal(route[key], replay[key], err_msg=key)

    def test_line_spacing_endpoint_altitude_and_receiver(self):
        route = self.build(dict(type="line", start=[1, 1], end=[1.25, 1], step_m=.1))
        np.testing.assert_allclose(route["requested_tx_m"][:, 0], [1, 1.1, 1.2, 1.25])
        np.testing.assert_allclose(route["tx_m"][:, 2], .82)
        np.testing.assert_allclose(route["rx_m"] - route["tx_m"], np.tile([.1, 0, 0], (4, 1)))
        np.testing.assert_allclose(route["distance_m"], [0, .1, .2, .25])
        self.assert_replays(route)

    def test_waypoint_corners_segments_and_diagonal_distance(self):
        route = self.build(dict(type="waypoints", points=[[1, 1], [1.2, 1.2], [1.2, 1.4]], step_m=.1))
        points = route["requested_tx_m"]
        self.assertTrue(np.any(np.all(np.isclose(points[:, :2], [1.2, 1.2]), axis=1)))
        self.assertEqual(set(route["segment_id"]), {0, 1})
        distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route["tx_m"], axis=0), axis=1))]
        np.testing.assert_allclose(route["distance_m"], distances)
        self.assert_replays(route)

    def test_snake_rows_connectors_and_rounded_excursions(self):
        spec = dict(type="snake", start=[1, 1], row_length_m=1, rows=3, row_spacing_m=.2, step_m=.05)
        straight = self.build(dict(spec, turn="straight"))
        rounded = self.build(dict(spec, turn="rounded"))
        np.testing.assert_allclose(straight["requested_tx_m"][-1], [2, 1.4, .82])
        self.assertEqual(set(straight["segment_id"]), {-1, 0, 1, 2})
        self.assertAlmostEqual(straight["distance_m"][-1], 3.4)
        self.assertGreater(np.max(rounded["requested_tx_m"][:, 0]), 2.09)
        self.assertLess(np.min(rounded["requested_tx_m"][:, 0]), .91)
        np.testing.assert_allclose(rounded["requested_rx_m"] - rounded["requested_tx_m"],
                                   np.tile([.1, 0, 0], (len(rounded["tx_m"]), 1)))
        self.assert_replays(rounded)

    def test_snake_rotation_and_single_row(self):
        route = self.build(dict(type="snake", start=[2, 1], row_length_m=1, rows=2,
                               row_spacing_m=.2, step_m=.1, heading_deg=90))
        np.testing.assert_allclose(route["requested_tx_m"][-1], [1.8, 1, .82])
        single = self.build(dict(type="snake", start=[1, 1], row_length_m=1, rows=1,
                                row_spacing_m=.2, step_m=.1, turn="rounded"))
        self.assertEqual(set(single["segment_id"]), {0})

    def test_explicit_points_preserve_samples_altitudes_and_snap(self):
        points = [[1.003, 1.003, .823], [1.207, 1.093, .837], [1.4, 1.3, .85]]
        route = self.build(dict(type="points", points=points))
        np.testing.assert_array_equal(route["requested_tx_m"], points)
        np.testing.assert_allclose(route["tx_m"], np.rint(np.array(points) / .01) * .01)
        self.assert_replays(route)

    def test_dedup_retains_final_endpoint_and_matching_metadata(self):
        points = [[1, 1], [1.001, 1], [1.011, 1], [1.012, 1]]
        route = self.build(dict(type="points", points=points))
        np.testing.assert_allclose(route["requested_tx_m"][:, 0], [1, 1.012])
        np.testing.assert_allclose(route["tx_m"][:, 0], [1, 1.01])
        np.testing.assert_array_equal(route["segment_id"], [0, 2])
        self.assertEqual(route["metadata"]["routes"][0]["removed_consecutive_duplicates"], 2)
        self.assert_replays(route)
        one = self.build(dict(type="points", points=[[1, 1], [1.001, 1]]))
        self.assertEqual(one["tx_m"].shape, (1, 3))
        self.assertEqual(one["distance_m"].tolist(), [0])
        self.assertAlmostEqual(one["requested_tx_m"][0, 0], 1.001)

    def test_arbitrary_route_count_distance_reset_and_no_cross_route_dedup(self):
        spec = {"routes": [dict(type="line", name=f"line_{i}", start=[1, 1], end=[1.2, 1], step_m=.1)
                           for i in range(4)]}
        route = self.build(spec)
        self.assertEqual(route["route_names"], [f"line_{i}" for i in range(4)])
        np.testing.assert_array_equal(route["route_id"], np.repeat(np.arange(4), 3))
        np.testing.assert_allclose(route["distance_m"], np.tile([0, .1, .2], 4))
        self.assert_replays(route)

    def test_csv_exact_and_resampled_snapshot_survives_file_removal(self):
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "samples.csv"
            csv_path.write_text("x_m,y_m,z_m\n1,1,.83\n1.2,1,.83\n1.2,1.2,.85\n")
            exact = self.build(dict(type="csv", file="samples.csv"), base_dir=temporary)
            sampled = self.build(dict(type="csv", file="samples.csv", step_m=.05), base_dir=temporary)
            self.assertEqual(len(exact["tx_m"]), 3)
            self.assertGreater(len(sampled["tx_m"]), 3)
            self.assertEqual(sampled["spec"]["type"], "points")
            self.assertNotIn("file", sampled["spec"])
            csv_path.unlink()
            self.assert_replays(exact)
            self.assert_replays(sampled)

    def test_csv_header_validation_bad_rows_and_nonfinite_values(self):
        invalid = ["", "x,y\n1,1\n", "x_m,y_m,y_m\n1,1,1\n", "x_m,y_m\n",
                   "x_m,y_m\n1\n", "x_m,y_m\n1,1,2\n", "x_m,y_m\nno,1\n",
                   "x_m,y_m\n1,nan\n", "x_m,y_m,z_m\n1,1,\n"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "points.csv"
            for contents in invalid:
                with self.subTest(contents=contents):
                    path.write_text(contents)
                    with self.assertRaises(ValueError):
                        self.build(dict(type="csv", file=str(path)))

    def test_invalid_specs_are_rejected(self):
        invalid = [None, {}, {"routes": []}, {"routes": [{"type": "points", "points": [[1, 1]]}], "name": "x"},
                   dict(type="line", start=[1, 1], end=[1, 1], step_m=.1),
                   dict(type="line", start=[1, 1], end=[2, 1], step_m=0),
                   dict(type="line", start=[True, 1], end=[2, 1], step_m=.1),
                   dict(type="line", start=[float("nan"), 1], end=[2, 1], step_m=.1),
                   dict(type="points", points=[]), dict(type="points", points=[[1, 1]], extra=True),
                   dict(type="points", points=[[1, 1]], segment_ids=[False]),
                   dict(type="waypoints", points=[[1, 1]], step_m=.1),
                   dict(type="waypoints", points=[[1, 1], [1, 1]], step_m=.1),
                   dict(type="snake", start=[1, 1], row_length_m=1, rows=1.5, row_spacing_m=.2, step_m=.1),
                   dict(type="snake", start=[1, 1], row_length_m=1, rows=2, row_spacing_m=0, step_m=.1),
                   dict(type="snake", start=[1, 1], row_length_m=1, rows=2, row_spacing_m=.2, step_m=.1, turn="x")]
        for spec in invalid:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                self.build(spec)
        repeated = {"routes": [dict(type="points", name="same", points=[[1, 1]])] * 2}
        with self.assertRaises(ValueError):
            self.build(repeated)

    def test_bounds_include_receiver_requested_and_snapped_positions(self):
        for point in ([.11, 1], [6.81, 1], [1, 4.7], [1, 1, .8], [1, 1, 1.09]):
            with self.subTest(point=point), self.assertRaises(ValueError):
                self.build(dict(type="points", points=[point]))
        # The requested altitude is above ground, but rounds down to the surface.
        with self.assertRaises(ValueError):
            self.build(dict(type="points", points=[[1, 1, .804]]))
        grid = dict(GRID, pml_thickness=.123)
        with self.assertRaises(ValueError):
            self.build(dict(type="points", points=[[.124, 1]]), grid=grid)

    def test_all_example_files_fit_their_domains_and_replay(self):
        directory = ROOT / "examples" / "trajectories"
        for path in sorted(directory.glob("*.json")):
            with self.subTest(path=path.name):
                grid = copy.deepcopy(GRID)
                source = copy.deepcopy(SOURCE)
                if path.name == "pilot-line.json":
                    grid["domain"] = [2.4, 1.2, 1.2]
                elif path.name == "smoke.json":
                    grid.update(domain=[1, .6, .8], surface_z=.5)
                    source.update(height_m=.04, tx_rx_offset_m=.08)
                route = self.build(json.loads(path.read_text()), grid=grid, source=source, base_dir=directory)
                self.assertGreater(len(route["tx_m"]), 1)
                for number in range(len(route["route_names"])):
                    positions = route["tx_m"][route["route_id"] == number]
                    self.assertTrue(np.all(np.linalg.norm(np.diff(positions, axis=0), axis=1) > 0))
                replay = self.build(route["spec"], grid=grid, source=source)
                np.testing.assert_array_equal(route["tx_m"], replay["tx_m"])


if __name__ == "__main__":
    unittest.main()
