"""Seeded sphere labels and constant-height survey paths; distances are metres."""
from .physical_constants import GEOMETRY_CLEARANCE_M, OBJECT_CLEARANCE_M
from pathlib import Path
import json
import numpy as np

MATERIALS = {"metal_sphere": "pec", "air_sphere": "free_space", "dielectric_sphere": "inclusion",
             "metal": "pec", "air": "free_space", "dielectric": "inclusion"}


def load_object_coordinates(spec_path, scene_names, domain, surface_z, control_scenes=(), clearance=OBJECT_CLEARANCE_M):
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
            if (np.any(c[:2] - r < GEOMETRY_CLEARANCE_M - 1e-9)
                    or np.any(c[:2] + r > np.asarray(domain[:2]) - GEOMETRY_CLEARANCE_M + 1e-9)
                    or c[2] - r < GEOMETRY_CLEARANCE_M - 1e-9 or c[2] + r > surface_z - clearance + 1e-9):
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


def object_commands(objects):
    """Global-coordinate gprMax sphere commands, after soil has been created."""
    return ["#sphere: " + " ".join(f"{v:.8g}" for v in [*o["center_m"], o["radius_m"]]) + " " + o["material"] for o in objects]
