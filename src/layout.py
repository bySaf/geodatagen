"""Seeded sphere labels and constant-height survey paths; distances are metres."""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.spatial import cKDTree

MATERIALS = {"metal_sphere": "pec", "air_sphere": "free_space", "dielectric_sphere": "inclusion",
             "metal": "pec", "air": "free_space", "dielectric": "inclusion"}


def load_object_coordinates(spec_path, scene_names, domain, surface_z, control_scenes=(), clearance=.04):
    """Load explicit spheres from {scene_name: [object, ...]} JSON, in metres.

    Required object keys: type, center_m=[x,y,z], radius_m. Existing objects.json
    records are accepted too; depth fields are derived again from center_m.
    All scenes must be present, including empty control scenes. Coordinates are
    global scene coordinates (z increases upwards), not burial depths. Objects
    have the same boundary and pairwise clearance rules as generated objects.
    """
    def finite_number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    raw = json.loads(Path(spec_path).read_text(), object_pairs_hook=unique_keys)
    names = list(scene_names)
    if len(set(names)) != len(names):
        raise ValueError("Scene names must be unique")
    if not isinstance(raw, dict) or set(raw) != set(names):
        raise ValueError(f"Coordinate file must contain exactly these scenes: {names}")
    if len(domain) != 3 or not all(finite_number(v) and v > 0 for v in domain):
        raise ValueError("Domain must contain three finite positive dimensions")
    if not finite_number(surface_z) or not 0 < surface_z < domain[2]:
        raise ValueError("Surface z must lie inside the domain")
    if not finite_number(clearance) or clearance < 0:
        raise ValueError("Object clearance must be finite and nonnegative")
    allowed = {"id", "type", "material", "center_m", "radius_m", "depth_center_m", "depth_top_m"}
    result = {}
    for scene in names:
        entries = raw[scene]
        if not isinstance(entries, list):
            raise ValueError(f"{scene}: objects must be a list")
        if scene in control_scenes and entries:
            raise ValueError(f"{scene}: a control scene must have an empty object list")
        objects, ids = [], set()
        for index, entry in enumerate(entries):
            label = f"{scene} object {index}"
            if not isinstance(entry, dict) or set(entry) - allowed:
                raise ValueError(f"{label}: invalid object fields")
            kind, center, radius = entry.get("type"), entry.get("center_m"), entry.get("radius_m")
            if not isinstance(kind, str) or kind not in MATERIALS:
                raise ValueError(f"{label}: unknown object type")
            if "material" in entry and entry["material"] != MATERIALS[kind]:
                raise ValueError(f"{label}: material disagrees with object type")
            if not isinstance(center, list) or len(center) != 3 or not all(finite_number(v) for v in center):
                raise ValueError(f"{label}: center_m requires three finite coordinates")
            if not finite_number(radius) or radius <= 0:
                raise ValueError(f"{label}: radius_m must be finite and positive")
            identifier = entry.get("id", f"object_{index:03d}")
            if not isinstance(identifier, str) or not identifier.strip() or identifier in ids:
                raise ValueError(f"{label}: object id must be a unique nonempty string")
            c, r = np.asarray(center, dtype=float), float(radius)
            if (np.any(c[:2] - r < .2 - 1e-9)
                    or np.any(c[:2] + r > np.asarray(domain[:2]) - .2 + 1e-9)
                    or c[2] - r < .2 - 1e-9 or c[2] + r > surface_z - clearance + 1e-9):
                raise ValueError(f"{label}: sphere violates the domain or ground-surface clearance")
            if any(np.linalg.norm(c - o["center_m"]) < r + o["radius_m"] + clearance - 1e-9 for o in objects):
                raise ValueError(f"{label}: sphere intersects another object or its clearance")
            depth = float(surface_z - c[2])
            objects.append(dict(id=identifier, type=kind, material=MATERIALS[kind],
                                center_m=c.tolist(), radius_m=r, depth_center_m=depth,
                                depth_top_m=depth-r))
            ids.add(identifier)
        result[scene] = objects
    return result


def generate_objects(spec_path, seed, domain, surface_z, control=False, clearance=.04):
    """Read: type count_min count_max radius_min radius_max depth_min depth_max.

    Depth means centre below the ground surface. Reject configurations that cannot
    fit; a failed placement is never silently dropped. Objects stay 0.2 m from
    domain boundaries and do not intersect each other or the ground surface.
    """
    rng = np.random.default_rng(seed)
    objects = []
    for number, line in enumerate(Path(spec_path).read_text().splitlines(), 1):
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if len(fields) != 7 or fields[0] not in MATERIALS:
            raise ValueError(f"Invalid object specification on line {number}: {line}")
        kind = fields[0]
        lo, hi = map(int, fields[1:3])
        rlo, rhi, dlo, dhi = map(float, fields[3:])
        if not (0 <= lo <= hi and 0 < rlo <= rhi and 0 < dlo <= dhi):
            raise ValueError(f"Invalid ranges on line {number}")
        if control:
            continue
        for _ in range(int(rng.integers(lo, hi + 1))):
            for attempt in range(2000):
                radius = float(rng.uniform(rlo, rhi))
                depth = float(rng.uniform(dlo, dhi))
                centre = np.array([rng.uniform(.7 + radius, domain[0] - .7 - radius),
                                   rng.uniform(.7 + radius, domain[1] - .7 - radius),
                                   surface_z - depth])
                # Default survey covers two horizontal bands. Keep targets under
                # them, rather than filling unsurveyed margins with invisible labels.
                if domain[0] >= 7 and domain[1] >= 4.8:
                    centre[0] = rng.uniform(1.0, 5.5)
                    centre[1] = rng.uniform(.9, 1.6) if rng.random() < .5 else rng.uniform(2.7, 3.4)
                if np.any(centre[:2]-radius < .2) or np.any(centre[:2]+radius > np.asarray(domain[:2])-.2):
                    continue
                if centre[2] - radius < .2 or centre[2] + radius > surface_z - clearance:
                    continue
                if any(np.linalg.norm(centre - o["center_m"]) < radius + o["radius_m"] + clearance for o in objects):
                    continue
                objects.append({"id": f"object_{len(objects):03d}", "type": kind,
                                "material": MATERIALS[kind], "center_m": centre.tolist(),
                                "radius_m": radius, "depth_center_m": depth,
                                "depth_top_m": depth - radius})
                break
            else:
                raise ValueError(f"Could not place {kind}; enlarge domain or change ranges")
    return objects


def straight_path(start, direction, n, step):
    """Straight survey with a specified step; domain checking happens separately."""
    direction = np.asarray(direction, dtype=float)
    if np.linalg.norm(direction) == 0 or n < 2 or step <= 0:
        raise ValueError("Require a direction, n>=2 and positive step")
    return np.asarray(start, dtype=float) + np.arange(n)[:, None] * step * direction / np.linalg.norm(direction)


def _rounded_raster(n, step, rows, x0, y0, pitch):
    """Analytic straights + semicircular U-turns, sampled by arc length."""
    radius = pitch / 2
    length = ((n - 1) * step - (rows - 1) * np.pi * radius) / rows
    if length <= 0 or radius <= 0:
        raise ValueError("Rounded raster needs positive straight length and row pitch")
    segments = []
    for row in range(rows):
        y = y0 + row * pitch
        sign = 1 if row % 2 == 0 else -1
        start_x = x0 if sign == 1 else x0 + length
        segments.append((length, lambda s, x=start_x, yy=y, d=sign: (x + d*s, yy)))
        if row < rows - 1:
            end_x = x0 + length if sign == 1 else x0
            segments.append((np.pi*radius,
                             lambda s, x=end_x, yy=y, d=sign: (x + d*radius*np.sin(s/radius), yy+radius*(1-np.cos(s/radius)))))
    points = []
    segment = 0
    consumed = 0.
    for s in np.arange(n) * step:
        while segment < len(segments)-1 and s > consumed + segments[segment][0]:
            consumed += segments[segment][0]
            segment += 1
        points.append(segments[segment][1](s-consumed))
    return np.asarray(points)


def make_trajectories(domain=(7, 4.8, 1.2), surface_z=.9, dx=.02, n=500,
                      step=.1, antenna_height=.06, tx_rx_offset=.04,
                      start_x=.8, start_y=.8, row_pitch=.1, overlap_shift=.04,
                      separation=.9, margin=.65, paths=None):
    """Default two nearby rasters + one rounded raster, or three explicit paths.

    Explicit paths use shape='line' (start + direction or end) or
    shape='rounded_raster' (start, optional rows/row_pitch). They share n;
    individual step is optional. A line end sets spacing unless step is supplied,
    in which case both constraints must agree. Actual solver coordinates are
    snapped, while requested coordinates retain the continuous geometry.
    """
    if n < 2 or int(n) != n or step <= 0 or dx <= 0 or row_pitch <= 0:
        raise ValueError("Require integer n>=2 and positive step/dx/row_pitch")
    candidates = []
    if paths is None:
        columns = min(50, n)
        rows = int(np.ceil(n / columns))
        base = np.array([(start_x + (j if i % 2 == 0 else columns-1-j)*step, start_y+i*row_pitch)
                         for i in range(rows) for j in range(columns)])[:n]
        separated_y = start_y + (rows - 1)*row_pitch + separation
        curved = _rounded_raster(n, step, rows, start_x, separated_y, row_pitch)
        for name, shape, xy in [("overlap_a", "piecewise_straight", base),
                                ("overlap_b", "piecewise_straight", base+[0, overlap_shift]),
                                ("separate_curve", "rounded_raster", curved)]:
            candidates.append((name, shape, np.c_[xy, np.full(n, surface_z+antenna_height)], step))
    else:
        if len(paths) != 3:
            raise ValueError("Provide exactly three explicit paths")
        for index, spec in enumerate(paths):
            shape = spec["shape"]
            start = np.asarray(spec["start"], dtype=float)
            if start.shape == (2,):
                start = np.r_[start, surface_z+antenna_height]
            if start.shape != (3,):
                raise ValueError("Path start must have two or three coordinates")
            local_step = float(spec.get("step", step))
            if shape == "line":
                if ("end" in spec) == ("direction" in spec):
                    raise ValueError("A line requires exactly one of end or direction")
                if "end" in spec:
                    end = np.asarray(spec["end"], dtype=float)
                    if end.shape == (2,): end = np.r_[end, start[2]]
                    if end.shape != (3,): raise ValueError("Invalid line end")
                    direction = end-start
                    inferred = np.linalg.norm(direction)/(n-1)
                    if "step" in spec and not np.isclose(inferred, local_step):
                        raise ValueError("Line end disagrees with n and step")
                    local_step = inferred
                else:
                    direction = np.asarray(spec["direction"], dtype=float)
                    if direction.shape == (2,): direction = np.r_[direction, 0.]
                if direction.shape != (3,) or not np.isclose(direction[2],0):
                    raise ValueError("Survey directions must be horizontal")
                tx = straight_path(start,direction,n,local_step)
            elif shape == "rounded_raster":
                rows = int(spec.get("rows", np.ceil(n/min(50,n))))
                if rows < 1: raise ValueError("rows must be positive")
                xy = _rounded_raster(n,local_step,rows,*start[:2],float(spec.get("row_pitch",row_pitch)))
                tx = np.c_[xy,np.full(n,start[2])]
            else:
                raise ValueError(f"Unknown trajectory shape: {shape}")
            candidates.append((f"path_{index}",shape,tx,local_step))
    result = []
    margin = np.broadcast_to(np.asarray(margin,dtype=float),(2,))
    if np.any(margin < 0): raise ValueError("margin cannot be negative")
    for name,shape,tx,local_step in candidates:
        rx = tx + [tx_rx_offset,0,0]
        actual_tx, actual_rx = np.round(tx/dx)*dx, np.round(rx/dx)*dx
        for a in (tx,rx,actual_tx,actual_rx):
            if not np.isfinite(a).all(): raise ValueError("Nonfinite trajectory coordinates")
            if np.any(a[:,:2] < margin-1e-8) or np.any(a[:,:2] > np.asarray(domain[:2])-margin+1e-8):
                raise ValueError(f"{name} does not fit: enlarge domain or change path/margin")
            if np.any(a[:,2] <= surface_z) or np.any(a[:,2] >= domain[2]-.1):
                raise ValueError("Antenna must be above ground and below top boundary margin")
        result.append(dict(name=name,shape=shape,nominal_step_m=local_step,nominal_length_m=(n-1)*local_step,
                           requested_tx=tx,snapped_tx=actual_tx,requested_rx=rx,snapped_rx=actual_rx))
    return result


def overlap_matrix(paths, radius=.15):
    """Directed fraction of actual Tx xy samples within radius of another path."""
    points = [np.asarray(p["snapped_tx"])[:, :2] for p in paths]
    return [[float(np.mean(cKDTree(b).query(a)[0] <= radius)) for b in points] for a in points]


def write_layout(out, objects, paths, overlap_radius=.15):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'objects.json').write_text(json.dumps(objects, indent=2))
    metadata = []
    for path in paths:
        columns = ["scan_id"] + [f"{role}_{axis}_m" for role in ["tx", "rx", "requested_tx", "requested_rx"] for axis in "xyz"]
        values = np.c_[np.arange(len(path['snapped_tx'])), path['snapped_tx'], path['snapped_rx'], path['requested_tx'], path['requested_rx']]
        with (out/f"{path['name']}.csv").open('w', newline='') as f:
            writer = csv.writer(f); writer.writerow(columns); writer.writerows(values)
        metadata.append({k: v for k, v in path.items() if k not in ['requested_tx','requested_rx','snapped_tx','snapped_rx']})
    (out/'trajectories.json').write_text(json.dumps(dict(paths=metadata, overlap_radius_m=overlap_radius, directed_overlap=overlap_matrix(paths,overlap_radius)), indent=2))


def object_commands(objects):
    """Global-coordinate gprMax sphere commands, after soil has been created."""
    return ["#sphere: " + " ".join(f"{v:.8g}" for v in [*o["center_m"], o["radius_m"]]) + " " + o["material"] for o in objects]
