"""Independent survey routes in metres, sampled and snapped to the solver grid.

``step_m`` is a maximum spacing before snapping: every line/waypoint edge and
snake piece includes its endpoint, so the final interval can be shorter. Snake
turns are sampled along their arc. The receiver always has the source's fixed
offset in the *global +X direction*, including on reversed rows and turns.

Each route is processed independently. Consecutive samples in the same Tx grid
cell are collapsed; the first sample represents an interior run, and the final
sample represents the final run, preserving the requested route endpoint. A
route that fits in one cell has one sample. Requested route endpoints remain in
``spec`` even if snapping collapses them. Distance is measured between retained,
snapped Tx samples and starts at zero for each route. A joint belongs to its
outgoing segment; the last endpoint belongs to the last segment. Snake rows are
numbered from zero and connectors/rounded turns use segment_id=-1.
"""
from collections.abc import Mapping
import csv
from pathlib import Path

import numpy as np


def _number(value, label, *, positive=False, nonnegative=False):
    if (isinstance(value, (bool, str)) or not np.isscalar(value)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _point(value, default_z, label):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) not in (2, 3):
        raise ValueError(f"{label} requires [x, y] or [x, y, z]")
    point = [_number(v, f"{label} coordinate") for v in value]
    if len(point) == 2:
        point.append(default_z)
    return np.asarray(point, dtype=float)


def _points(value, default_z, label, minimum=1):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < minimum:
        raise ValueError(f"{label} requires at least {minimum} point(s)")
    return np.asarray([_point(p, default_z, f"{label}[{i}]") for i, p in enumerate(value)])


def _distances(length, step):
    # Explicit endpoints avoid arange overshoot and keep corners exactly.
    values = np.arange(int(np.floor(length / step)) + 1, dtype=float) * step
    if np.isclose(values[-1], length, rtol=1e-12, atol=1e-12):
        values[-1] = length
    else:
        values = np.r_[values, length]
    return values


def _line_points(start, end, step):
    length = float(np.linalg.norm(end - start))
    if length <= 1e-12:
        raise ValueError("Adjacent route endpoints must be distinct")
    return start + _distances(length, step)[:, None] * (end - start) / length


def _join(pieces):
    """Assign a shared joint to the outgoing piece, with no repeated endpoints."""
    points, segments = [], []
    for i, (samples, segment) in enumerate(pieces):
        samples = samples if i == len(pieces) - 1 else samples[:-1]
        points.append(samples)
        segments.append(np.full(len(samples), segment, dtype=np.int64))
    return np.concatenate(points), np.concatenate(segments)


def _waypoints(points, step):
    return _join([(_line_points(a, b, step), i)
                  for i, (a, b) in enumerate(zip(points[:-1], points[1:]))])


def _csv_points(file, base_dir, default_z):
    if not isinstance(file, str) or not file.strip():
        raise ValueError("csv file must be a nonempty path string")
    path = Path(file).expanduser()
    if not path.is_absolute():
        path = Path(base_dir or ".") / path
    points = []
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream, strict=True)
            fields = reader.fieldnames
            if (not fields or len(set(fields)) != len(fields)
                    or set(fields) not in ({"x_m", "y_m"}, {"x_m", "y_m", "z_m"})):
                raise ValueError("CSV header must contain x_m,y_m and optional z_m only")
            for row in reader:
                if None in row or any(v is None or not v.strip() for v in row.values()):
                    raise ValueError(f"Malformed CSV row {reader.line_num}")
                try:
                    point = [float(row[k]) for k in ("x_m", "y_m")]
                    point.append(float(row["z_m"]) if "z_m" in fields else default_z)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Malformed CSV coordinate on row {reader.line_num}") from exc
                points.append(_point(point, default_z, f"CSV row {reader.line_num}"))
    except (OSError, csv.Error) as exc:
        raise ValueError(f"Cannot read trajectory CSV {path}: {exc}") from exc
    if not points:
        raise ValueError("CSV trajectory must contain at least one coordinate row")
    return np.asarray(points)


def _route(spec, default_z, base_dir, name):
    kind = spec.get("type")
    allowed = {
        "line": {"start", "end", "step_m"},
        "snake": {"start", "row_length_m", "rows", "row_spacing_m", "step_m", "turn", "heading_deg"},
        "waypoints": {"points", "step_m"},
        "points": {"points", "segment_ids"},
        "csv": {"file", "step_m"},
        "random": {"bounds_m", "count", "seed", "method", "min_distance_m", "order"},
    }
    if not isinstance(kind, str) or kind not in allowed:
        raise ValueError(f"{name}: type must be line, snake, waypoints, points, csv or random")
    unknown = set(spec) - allowed[kind] - {"type", "name"}
    if unknown:
        raise ValueError(f"{name}: unknown trajectory fields: {sorted(unknown)}")
    canonical = {"type": kind, "name": name}
    step = None
    if kind in {"line", "snake", "waypoints"} or "step_m" in spec:
        step = _number(spec.get("step_m"), f"{name}.step_m", positive=True)
    if kind == "random":
        from .sampling import sample_coordinates, nearest_order
        bounds = spec.get('bounds_m')
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError('Random trajectory bounds_m requires X and Y ranges')
        count = spec.get('count')
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError('Random trajectory count must be a positive integer')
        xy = sample_coordinates(bounds, count, spec.get('seed', 0), spec.get('method', 'uniform'), spec.get('min_distance_m'))
        order = spec.get('order', 'nearest')
        if order not in ('nearest', 'sample'):raise ValueError('order must be nearest or sample')
        if order == 'nearest':xy = nearest_order(xy)
        points = np.c_[xy, np.full(len(xy), default_z)]
        segments = np.arange(len(points), dtype=np.int64)
        canonical.update(type='points', points=points.tolist(), segment_ids=segments.tolist())
    elif kind == "line":
        start = _point(spec.get("start"), default_z, f"{name}.start")
        end = _point(spec.get("end"), default_z, f"{name}.end")
        points = _line_points(start, end, step)
        segments = np.zeros(len(points), dtype=np.int64)
        canonical.update(start=start.tolist(), end=end.tolist(), step_m=step)
    elif kind == "snake":
        start = _point(spec.get("start"), default_z, f"{name}.start")
        length = _number(spec.get("row_length_m"), f"{name}.row_length_m", positive=True)
        pitch = _number(spec.get("row_spacing_m"), f"{name}.row_spacing_m", positive=True)
        rows = _number(spec.get("rows"), f"{name}.rows", positive=True)
        if not rows.is_integer():
            raise ValueError(f"{name}.rows must be a positive integer")
        rows = int(rows)
        turn = spec.get("turn", "straight")
        if turn not in ("straight", "rounded"):
            raise ValueError(f"{name}.turn must be straight or rounded")
        heading = _number(spec.get("heading_deg", 0), f"{name}.heading_deg")
        angle = np.deg2rad(heading)
        forward = np.array([np.cos(angle), np.sin(angle), 0.])
        across = np.array([-np.sin(angle), np.cos(angle), 0.])
        pieces = []
        for row in range(rows):
            sign = 1 if row % 2 == 0 else -1
            a = start + row * pitch * across + (length * forward if sign < 0 else 0)
            b = a + sign * length * forward
            pieces.append((_line_points(a, b, step), row))
            if row == rows - 1:
                continue
            if turn == "straight":
                samples = _line_points(b, b + pitch * across, step)
            else:
                radius = pitch / 2
                theta = _distances(np.pi * radius, step) / radius
                samples = (b + sign * radius * np.sin(theta)[:, None] * forward
                           + radius * (1 - np.cos(theta))[:, None] * across)
                samples[0], samples[-1] = b, b + pitch * across
            pieces.append((samples, -1))
        points, segments = _join(pieces)
        canonical.update(start=start.tolist(), row_length_m=length, rows=rows,
                         row_spacing_m=pitch, step_m=step, turn=turn, heading_deg=heading)
    elif kind == "waypoints":
        vertices = _points(spec.get("points"), default_z, f"{name}.points", minimum=2)
        points, segments = _waypoints(vertices, step)
        canonical.update(points=vertices.tolist(), step_m=step)
    else:
        if kind == "csv":
            points = _csv_points(spec.get("file"), base_dir, default_z)
        else:
            points = _points(spec.get("points"), default_z, f"{name}.points")
        if step is not None:
            if len(points) < 2:
                raise ValueError("CSV resampling requires at least two points")
            points, segments = _waypoints(points, step)
        else:
            segments = np.minimum(np.arange(len(points)), max(0, len(points) - 2)).astype(np.int64)
        if "segment_ids" in spec:
            ids = spec["segment_ids"]
            if (not isinstance(ids, (list, tuple)) or len(ids) != len(points)
                    or any(isinstance(v, bool) or not isinstance(v, int) or v < -1 for v in ids)):
                raise ValueError("segment_ids must contain one integer >= -1 per point")
            segments = np.asarray(ids, dtype=np.int64)
        # CSV is frozen to explicit samples and segment IDs; replay never reads
        # the original file and preserves resampled waypoint-edge identities.
        canonical.update(type="points", points=points.tolist(), segment_ids=segments.tolist())
    return points, segments, canonical


def build_trajectory(spec, grid, source, base_dir=None):
    """Return flattened arrays and a JSON-safe reproducible trajectory ``spec``.

    Accept one route or ``{"routes": [route, ...]}``. Each route may have a
    unique ``name``. Two-coordinate positions use surface_z + source.height_m;
    three-coordinate positions retain their supplied altitude. ``base_dir`` is
    the directory containing the trajectory JSON, used for relative CSV paths.
    Both requested and solver coordinates must be above the nominal surface
    and inside the non-PML volume. Snapping is NumPy round-to-nearest/even.

    ``requested_*_m`` correspond exactly to retained solver samples. Metadata
    records removed sample counts; the canonical spec keeps the full requested
    geometry, including explicit points collapsed during snapping.
    """
    if not isinstance(spec, Mapping):
        raise ValueError("Trajectory must be an object")
    if not isinstance(grid, Mapping) or not isinstance(source, Mapping):
        raise ValueError("grid and source must be objects")
    domain = _points([grid.get("domain")], 0, "grid.domain")[0]
    if len(grid.get("domain", [])) != 3 or np.any(domain <= 0):
        raise ValueError("grid.domain requires three positive dimensions")
    dx = _number(grid.get("solver_dx", grid.get("dx")), "grid.solver_dx", positive=True)
    pml = _number(grid.get("pml_thickness", .2), "grid.pml_thickness", nonnegative=True)
    if np.any(domain <= 2 * pml):
        raise ValueError("PML leaves no interior domain")
    surface = _number(grid.get("surface_z"), "grid.surface_z")
    height = _number(source.get("height_m"), "source.height_m", positive=True)
    offset = _number(source.get("tx_rx_offset_m"), "source.tx_rx_offset_m", nonnegative=True)
    if "routes" in spec:
        if set(spec) != {"routes"} or not isinstance(spec["routes"], list) or not spec["routes"]:
            raise ValueError("Multi-route spec requires a nonempty routes list and no other fields")
        specs = spec["routes"]
    else:
        specs = [spec]
    records, canonical, names, counts = [], [], [], []
    for route_id, local in enumerate(specs):
        if not isinstance(local, Mapping):
            raise ValueError(f"Route {route_id} must be an object")
        name = local.get("name", f"route_{route_id}")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Route names must be unique nonempty strings")
        requested_tx, segment_ids, resolved = _route(local, surface + height, base_dir, name)
        requested_rx = requested_tx + [offset, 0, 0]
        cells = np.rint(requested_tx / dx)
        tx, rx = cells * dx, np.rint(requested_rx / dx) * dx
        for label, points in (("requested Tx", requested_tx), ("requested Rx", requested_rx),
                              ("solver Tx", tx), ("solver Rx", rx)):
            if not np.isfinite(points).all():
                raise ValueError(f"{name}: {label} coordinates must be finite")
            if np.any(points < pml - 1e-10) or np.any(points > domain - pml + 1e-10):
                raise ValueError(f"{name}: {label} lies outside the non-PML domain")
            if np.any(points[:, 2] <= surface + 1e-10):
                raise ValueError(f"{name}: {label} must be above the nominal ground surface")
        keep = np.r_[True, np.any(np.diff(cells, axis=0) != 0, axis=1)]
        indices = np.flatnonzero(keep)
        # The last run represents the exact final requested endpoint.
        indices[-1] = len(tx) - 1
        count = len(indices)
        actual_tx = tx[indices]
        distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(actual_tx, axis=0), axis=1))]
        records.append(dict(tx_m=actual_tx, rx_m=rx[indices], requested_tx_m=requested_tx[indices],
                            requested_rx_m=requested_rx[indices], distance_m=distances,
                            route_id=np.full(count, route_id, dtype=np.int64),
                            segment_id=segment_ids[indices]))
        counts.append({"name": name, "requested_samples": len(tx), "retained_samples": count,
                       "removed_consecutive_duplicates": len(tx) - count})
        canonical.append(resolved)
        names.append(name)
    result = {key: np.concatenate([record[key] for record in records]) for key in records[0]}
    result.update(route_names=names, spec={"routes": canonical} if "routes" in spec else canonical[0],
                  metadata={"distance_basis": "cumulative snapped Tx distance within each route",
                            "receiver_offset_direction": "global +X", "routes": counts,
                            "duplicate_policy": "first sample of each Tx cell run; last sample of final run"})
    return result
