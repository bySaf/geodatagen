"""Coupled random soil fields adapted from the user-provided gpr_fields.py.

Source: /Users/davsf/Downloads/to_send/gpr_fields.py
Source SHA-256: c0521175dd50185a82b3bfc27246cef759761dee0530f463314a1d649b0870a1
Copied core: Layer/Pipe/Config/Fields, validation, Matérn-like FFT fields,
rough stratigraphy, coupled CRIM/Archie conversion and material quantization.
The original module's authorship is not established here. Its cited methods:
10.1109/JSTARS.2015.2468597, 10.1137/S1064827592240555, 10.2118/942054-G.

Local changes: retain the library core only; repair the invalid default surface
height; limit FFT allocation. Coordinate order is x,y,z; z increases upward.
CRIM/Archie is nondispersive and approximate, not an improved Peplinski model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


FIELD_NAMES = ("epsilon_r", "sigma_s_m", "resistivity_ohm_m",
               "porosity", "saturation", "layer_id")


@dataclass(frozen=True)
class Layer:
    name: str
    thickness_m: float  # nominal thickness; the last layer fills to z=0
    porosity: float
    porosity_sd: float
    saturation: float
    saturation_sd: float
    water_conductivity_s_m: float
    grain_permittivity: float = 4.5
    archie_a: float = 1.0
    archie_m: float = 1.5
    archie_n: float = 2.0
    surface_conductivity_s_m: float = 0.0  # calibrate for clay
    correlation_m: tuple[float, float, float] = (0.10, 0.10, 0.035)
    roughness_rms_m: float = 0.008
    roughness_corr_m: tuple[float, float] = (0.15, 0.12)


@dataclass(frozen=True)
class Pipe:
    """Hollow cylinder parallel to y, extending across the entire y-domain."""

    x_m: float
    z_m: float
    outer_radius_m: float
    wall_m: float
    wall_epsilon_r: float = 3.0
    wall_sigma_s_m: float = 1e-5
    fill_epsilon_r: float = 1.0
    fill_sigma_s_m: float = 0.0


@dataclass(frozen=True)
class Config:
    shape: tuple[int, int, int] = (200, 1, 150)
    spacing_m: tuple[float, float, float] = (0.005, 0.005, 0.005)
    surface_z_m: float = 0.6
    surface_roughness_rms_m: float = 0.000
    water_epsilon_r: float = 80.0  # approximate, nondispersive
    seed: int = 17
    bins_epsilon: int = 5
    bins_log_sigma: int = 5
    max_fft_cells: int = 50_000_000


@dataclass
class Fields:
    epsilon_r: np.ndarray
    sigma_s_m: np.ndarray
    porosity: np.ndarray
    saturation: np.ndarray
    layer_id: np.ndarray  # 0=air, 1..N=soil, N+1=pipe wall, N+2=fill
    config: Config
    layer_names: tuple[str, ...]

    @property
    def resistivity_ohm_m(self) -> np.ndarray:
        """Air and other zero-conductivity cells have infinite resistivity."""
        return np.divide(1., self.sigma_s_m,
                         out=np.full_like(self.sigma_s_m, np.inf),
                         where=self.sigma_s_m > 0)


def _check(layers: tuple[Layer, ...], cfg: Config, pipe: Pipe | None) -> None:
    if not layers:
        raise ValueError("At least one soil layer is required")
    if len(cfg.shape) != 3 or len(cfg.spacing_m) != 3 or any(n < 1 for n in cfg.shape):
        raise ValueError("shape must contain three positive cell counts")
    if any(d <= 0 or not np.isfinite(d) for d in cfg.spacing_m):
        raise ValueError("spacing_m must contain positive finite lengths")
    ztop = cfg.shape[2] * cfg.spacing_m[2]
    if not 0 < cfg.surface_z_m < ztop:
        raise ValueError("surface_z_m must lie inside the domain")
    if cfg.surface_roughness_rms_m < 0:
        raise ValueError("surface roughness must be nonnegative")
    if cfg.bins_epsilon < 1 or cfg.bins_log_sigma < 1:
        raise ValueError("material bin counts must be positive")
    if cfg.water_epsilon_r <= 1:
        raise ValueError("water_epsilon_r must exceed 1")
    for l in layers:
        if (l.thickness_m <= 0 or not 0 < l.porosity < 1
                or not 0 <= l.saturation <= 1 or l.porosity_sd < 0
                or l.saturation_sd < 0 or l.water_conductivity_s_m <= 0
                or l.grain_permittivity <= 1 or l.archie_a <= 0
                or l.archie_m <= 0 or l.archie_n <= 0
                or l.surface_conductivity_s_m < 0
                or any(c <= 0 for c in l.correlation_m + l.roughness_corr_m)
                or l.roughness_rms_m < 0):
            raise ValueError(f"Invalid layer parameters for {l.name}")
    if pipe and (pipe.outer_radius_m <= 0 or not 0 < pipe.wall_m < pipe.outer_radius_m
                 or pipe.wall_epsilon_r < 1 or pipe.fill_epsilon_r < 1
                 or pipe.wall_sigma_s_m < 0 or pipe.fill_sigma_s_m < 0):
        raise ValueError("Invalid pipe parameters")


def correlated_field(shape: tuple[int, ...], spacing: tuple[float, ...],
                     correlation: tuple[float, ...], rng: np.random.Generator,
                     max_fft_cells: int = 16_000_000, smoothness: float = 1.0
                     ) -> np.ndarray:
    """Unit-variance, Matérn-like Gaussian field from an FFT spectral filter.

    Padding and cropping reduce periodic seams. Correlation lengths specify
    approximate scales, not fitted covariance ranges. Singleton axes are
    invariant; the returned field remains in x,y,z order.
    """
    if not (len(shape) == len(spacing) == len(correlation)):
        raise ValueError("shape, spacing and correlation lengths must match")
    active = [i for i, n in enumerate(shape) if n > 1]
    if not active:
        return np.zeros(shape, dtype=np.float32)
    padded = tuple(2 * n if n > 1 else 1 for n in shape)
    if np.prod(padded, dtype=np.int64) > max_fft_cells:
        raise ValueError("FFT field too large; increase max_fft_cells or tile the model")
    spectrum_arg = np.ones(padded, dtype=np.float64)
    for axis in active:
        k = np.fft.fftfreq(padded[axis], spacing[axis])
        view = [1] * len(shape)
        view[axis] = len(k)
        spectrum_arg += (2 * np.pi * correlation[axis] * k.reshape(view)) ** 2
    psd_sqrt = spectrum_arg ** (-(smoothness + len(active) / 2) / 2)
    noise = rng.standard_normal(padded)
    field = np.fft.ifftn(np.fft.fftn(noise) * psd_sqrt).real
    field = field[tuple(slice(n // 2, n // 2 + n) if n > 1 else slice(None)
                        for n in shape)]
    sd = field.std()
    return ((field - field.mean()) / sd if sd > 1e-12
            else np.zeros(shape)).astype(np.float32)


def generate(layers: tuple[Layer, ...], cfg: Config = Config(),
             pipe: Pipe | None = None) -> Fields:
    """Generate one fixed 3D realization of εr, σ, ρ, porosity and saturation.

    Layers have independent multiscale random fields. Within a layer, shared
    latent fields couple porosity, moisture and pore-water conductivity.
    """
    _check(layers, cfg, pipe)
    rng = np.random.default_rng(cfg.seed)
    nx, ny, nz = cfg.shape
    dx, dy, dz = cfg.spacing_m
    xy_shape, xy_spacing = (nx, ny), (dx, dy)
    surface_noise = correlated_field(xy_shape, xy_spacing, (0.16, 0.16), rng,
                                     cfg.max_fft_cells)
    surface_z = cfg.surface_z_m + np.clip(cfg.surface_roughness_rms_m * surface_noise,
                                         -3 * cfg.surface_roughness_rms_m,
                                         3 * cfg.surface_roughness_rms_m)
    z = (np.arange(nz, dtype=np.float32) + 0.5) * dz
    depth = surface_z[:, :, None] - z[None, None, :]
    epsilon = np.ones(cfg.shape, dtype=np.float32)
    sigma = np.zeros(cfg.shape, dtype=np.float32)
    porosity = np.full(cfg.shape, np.nan, dtype=np.float32)
    saturation = np.full(cfg.shape, np.nan, dtype=np.float32)
    layer_id = np.zeros(cfg.shape, dtype=np.int16)

    lower = np.zeros(xy_shape, dtype=np.float32)
    for i, layer in enumerate(layers):
        if i == len(layers) - 1:
            mask = depth >= lower[:, :, None]
        else:
            rough = correlated_field(xy_shape, xy_spacing, layer.roughness_corr_m,
                                     rng, cfg.max_fft_cells) * layer.roughness_rms_m
            # Keep all layer boundaries ordered, even for unusual random seeds.
            limit = 0.22 * min(layer.thickness_m, layers[i + 1].thickness_m)
            upper = lower + layer.thickness_m + np.clip(rough, -limit, limit)
            mask = (depth >= lower[:, :, None]) & (depth < upper[:, :, None])
            lower = upper
        if not mask.any():
            continue
        shared = correlated_field(cfg.shape, cfg.spacing_m, layer.correlation_m,
                                  rng, cfg.max_fft_cells)
        local = correlated_field(cfg.shape, cfg.spacing_m,
                                 tuple(max(2 * d, c * 0.45)
                                       for d, c in zip(cfg.spacing_m, layer.correlation_m)),
                                 rng, cfg.max_fft_cells)
        p = np.clip(layer.porosity + layer.porosity_sd *
                    (0.75 * shared + 0.66 * local), 0.03, 0.8)
        sw = np.clip(layer.saturation + layer.saturation_sd *
                     (0.85 * shared - 0.53 * local), 0.001, 1.0)
        pore_sigma = layer.water_conductivity_s_m * np.exp(0.15 * shared)
        root_epsilon = ((1 - p) * np.sqrt(layer.grain_permittivity)
                        + p * sw * np.sqrt(cfg.water_epsilon_r) + p * (1 - sw))
        eps_l = root_epsilon ** 2  # CRIM, low-loss approximation
        sig_l = (pore_sigma * p ** layer.archie_m * sw ** layer.archie_n
                 / layer.archie_a + layer.surface_conductivity_s_m)
        epsilon[mask] = eps_l[mask]
        sigma[mask] = sig_l[mask]
        porosity[mask] = p[mask]
        saturation[mask] = sw[mask]
        layer_id[mask] = i + 1

    if pipe:
        x = (np.arange(nx, dtype=np.float32) + 0.5) * dx
        r2 = ((x[:, None, None] - pipe.x_m) ** 2
              + (z[None, None, :] - pipe.z_m) ** 2)
        outer = np.broadcast_to(r2 <= pipe.outer_radius_m ** 2, cfg.shape)
        inner = np.broadcast_to(r2 < (pipe.outer_radius_m - pipe.wall_m) ** 2,
                                cfg.shape)
        if np.any(outer & (depth < 0)):
            raise ValueError("Pipe crosses the air/soil interface")
        if not np.any(outer & (layer_id > 0)):
            raise ValueError("Pipe lies outside the soil")
        epsilon[outer], sigma[outer] = pipe.wall_epsilon_r, pipe.wall_sigma_s_m
        epsilon[inner], sigma[inner] = pipe.fill_epsilon_r, pipe.fill_sigma_s_m
        layer_id[outer] = len(layers) + 1
        layer_id[inner] = len(layers) + 2
        porosity[outer], saturation[outer] = np.nan, np.nan
    return Fields(epsilon, sigma, porosity, saturation, layer_id, cfg,
                  ("air",) + tuple(l.name for l in layers)
                  + (("pipe_wall", "pipe_fill") if pipe else ()))


def material_palette(fields: Fields) -> tuple[np.ndarray, list[str], dict]:
    """Quantize ε and log(σ) within each layer into occupied 2D quantile bins."""
    ids = np.empty(fields.config.shape, dtype=np.int16)
    keys: list[str] = []
    materials: dict = {}
    for layer_num, name in enumerate(fields.layer_names):
        mask = fields.layer_id == layer_num
        if not mask.any():
            continue
        e = fields.epsilon_r[mask].astype(np.float64)
        s = fields.sigma_s_m[mask].astype(np.float64)
        if np.all(e == e[0]) and np.all(s == s[0]):
            code = np.zeros(e.size, dtype=np.int64)
        else:
            le = np.quantile(e, np.linspace(0, 1, fields.config.bins_epsilon + 1)[1:-1])
            ls = np.quantile(np.log10(np.maximum(s, 1e-12)),
                             np.linspace(0, 1, fields.config.bins_log_sigma + 1)[1:-1])
            code = (np.searchsorted(le, e, side="right") * fields.config.bins_log_sigma
                    + np.searchsorted(ls, np.log10(np.maximum(s, 1e-12)), side="right"))
        local_ids = np.empty(e.size, dtype=np.int16)
        for bin_num in np.unique(code):
            pick = code == bin_num
            key = f"layer{layer_num:02d}_bin{int(bin_num):03d}"
            if len(keys) >= 32767:
                raise ValueError("Too many materials for the int16 gprMax palette")
            local_ids[pick] = len(keys)
            keys.append(key)
            # Geometric conductivity mean preserves positive order of magnitude.
            mean_s = (float(np.exp(np.log(s[pick]).mean()))
                      if np.all(s[pick] > 0) else float(s[pick].mean()))
            materials[key] = {
                "name": f"{name} bin {int(bin_num)}", "model": "constant",
                "base": {"relative_permittivity": float(e[pick].mean()),
                         "electric_conductivity_s_per_m": mean_s,
                         "relative_permeability": 1.0,
                         "magnetic_conductivity_s_per_m": 0.0},
            }
        ids[mask] = local_ids
    if not np.all(np.isfinite(fields.epsilon_r)) or not np.all(np.isfinite(fields.sigma_s_m)):
        raise ValueError("Non-finite material field")
    return ids, keys, materials

