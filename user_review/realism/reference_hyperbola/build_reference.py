"""Render a measured SFCW example without resampling or fitted hyperbolas."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "reference_radar"
SOURCE = REFERENCE / "concrete_specimen_II.npz"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    with np.load(SOURCE) as data:
        raw = data["amplitude"].copy()
        x = data["x_m"].copy()
        time_s = data["time_s"].copy()
    assert raw.shape == (372, 655)
    assert np.isfinite(raw).all()
    assert np.allclose(np.diff(x), 0.005)
    assert np.allclose(np.diff(time_s), 1 / 40.96e9, rtol=1e-9, atol=1e-20)
    # Subtract one profile-average trace; no frequency filter, gain, interpolation,
    # time-zero shift, spatial smoothing, downsampling or depth conversion.
    background = raw.astype(np.float64).mean(axis=0)
    display = raw.astype(np.float64) - background[None, :]
    time_ns = time_s * 1e9
    lim = float(np.percentile(np.abs(display[:, time_ns <= 6]), 99))
    normalized = display / lim
    dx = float(np.median(np.diff(x)))
    dt = float(np.median(np.diff(time_ns)))
    extent = [x[0] - dx / 2, x[-1] + dx / 2,
              time_ns[-1] + dt / 2, time_ns[0] - dt / 2]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(13, 8.5), facecolor="white")
    grid = fig.add_gridspec(2, 1, hspace=0.44, top=0.84, bottom=0.17,
                           left=0.075, right=0.89, height_ratios=[1, 0.95])
    axs = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
    for ax in axs:
        im = ax.imshow(normalized.T, cmap="gray", vmin=-1, vmax=1,
                       extent=extent, interpolation="nearest", aspect="auto")
        ax.set_ylabel("Время, нс")
        ax.set_xlabel("Расстояние по профилю, м")
        ax.tick_params(labelsize=11)
    axs[0].set_xlim(0, 1.85)
    axs[0].set_ylim(6, 0)
    axs[0].set_title("Профиль целиком · показаны первые 6 нс", loc="left", fontsize=14, pad=11)
    axs[1].set_xlim(0.85, 1.35)
    axs[1].set_ylim(3.8, 1.1)
    axs[1].set_xticks(np.arange(0.85, 1.351, 0.05))
    axs[1].set_title("Увеличение участка 0,85–1,35 м · видны отдельные гиперболы", loc="left", fontsize=14, pad=11)
    color_ax = fig.add_axes([0.92, 0.26, 0.018, 0.47])
    bar = fig.colorbar(im, cax=color_ax, ticks=[-1, 0, 1])
    bar.set_label("Нормированная амплитуда", fontsize=11)
    fig.text(0.075, 0.965, "Реальные гиперболы: арматура в бетоне", fontsize=21, weight="bold", va="top")
    fig.text(0.075, 0.918, "Proceq GP8800 · SFCW 0,4–6 ГГц · образец II · исходный шаг 5 мм", fontsize=13, color="#454545", va="top")
    fig.text(0.075, 0.090, "Обработка для показа: вычтена средняя трасса; одна шкала контраста для обеих панелей.", fontsize=10.5, color="#454545")
    fig.text(0.075, 0.063, "Оси не пересчитаны в глубину. Ноль времени — начало экспорта прибора.", fontsize=10.5, color="#454545")
    fig.text(0.075, 0.025, "Источник: M. Solla / Screening Eagle Technologies, 2024 · doi:10.5281/zenodo.10962520 · GPL-3.0-or-later", fontsize=10, color="#454545")
    fig.savefig(HERE / "real_hyperbola.png", dpi=180)
    plt.close(fig)

    np.savez_compressed(HERE / "real_hyperbola.npz", amplitude_raw=raw,
                        amplitude_background_removed=display.astype(np.float32),
                        background_trace=background, x_m=x, time_s=time_s,
                        display_p99=lim, axis_order="trace,time",
                        amplitude_units="instrument export units (uncalibrated)")
    original = json.loads((REFERENCE / "sources.json").read_text())["concrete"]
    profile = next(p for p in original["profiles"] if p["specimen"] == "II")
    sgy = REFERENCE / profile["file"]
    assert sha256(sgy) == profile["sha256"]
    metadata = {
        "dataset_url": original["dataset"],
        "dataset_doi": original["doi"],
        "paper_url": original["paper"],
        "primary_catalog_url": "https://portalcientifico.uvigo.gal/documentos/668fc42bb9e7c03b01bd564e?lang=en",
        "source_verified_online": "2026-11-12",
        "creator": original["creator"], "license": original["license"],
        "license_file": "../GPL-3.0.txt",
        "specimen": "II", "instrument": original["instrument"],
        "material": "Laboratory reinforced concrete; real measurements, not simulation; not park soil",
        "band_hz": original["native_band_hz"],
        "tx_rx_spacing_m": original["tx_rx_spacing_m"],
        "input_npz": "../reference_radar/concrete_specimen_II.npz",
        "input_npz_sha256": sha256(SOURCE),
        "original_sgy": "../reference_radar/" + profile["file"],
        "original_sgy_sha256": sha256(sgy),
        "original_archive_md5": original["archive_md5"],
        "array_shape_trace_time": list(raw.shape),
        "trace_step_m": dx, "sample_interval_s": float(np.median(np.diff(time_s))),
        "raw_last_x_m": float(x[-1]), "raw_last_time_ns": float(time_ns[-1]),
        "x_axis": profile["position_note"], "length_note": "CSV says 1.85 m; 372 traces at 5 mm infer endpoint 1.855 m. All numerical traces retained; plot x-limit is 1.85 m.",
        "sample_count_note": profile["sample_note"], "time_zero_note": profile["time_zero_note"],
        "processing": {
            "input": "Original measured instrument time-domain export, not diagnostic 400–900 MHz filtered array",
            "background": "Arithmetic mean of all 372 traces at each sample subtracted from each trace",
            "normalization": "Shared symmetric +/-99th percentile of absolute background-removed values in 0–6 ns",
            "p99_export_units": lim,
            "display": "gray colormap, nearest-neighbour pixel display; no spatial/time resampling",
            "full_panel_x_m": [0, 1.85], "full_panel_time_ns": [0, 6],
            "zoom_x_m": [0.85, 1.35], "zoom_time_ns": [1.1, 3.8],
            "other_processing": "No extra frequency filter, gain, warp, fitted curve, time-zero correction or depth conversion"
        },
        "interpretation": "Measured hyperbolic reflections in a known rebar specimen. The figure does not assign a diameter or depth to each visible arch and is not a quantitative validation of the park model.",
        "output_npz_sha256": sha256(HERE / "real_hyperbola.npz"),
        "output_png_sha256": sha256(HERE / "real_hyperbola.png"),
    }
    (HERE / "sources.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    (HERE / "README.txt").write_text(
        "РЕАЛЬНЫЙ ПРИМЕР С ЯВНЫМИ ГИПЕРБОЛАМИ\n\n"
        "Бетон с арматурой, лабораторный образец II. Это реальные данные Proceq GP8800: "
        "ступенчатый SFCW-радар, 0,4–6 ГГц, шаг трасс 5 мм.\n\n"
        "real_hyperbola.png — профиль 0–1,85 м / 0–6 нс и увеличение 0,85–1,35 м / 1,1–3,8 нс. "
        "Светлые и тёмные дуги — измеренные отклики. Теоретические гиперболы поверх них не нарисованы.\n"
        "Для показа вычтена средняя трасса; контраст ограничен общим P99 в окне 0–6 нс. "
        "Шаг не увеличен до 5 см, сглаживания и дополнительного частотного фильтра нет. "
        "Это визуальный пример характерной формы, а не совпадение с моделью паркового грунта.\n\n"
        "real_hyperbola.npz содержит все исходные 372 × 655 отсчётов, временную ось, "
        "координаты по номинальному шагу и отдельно обработанные для показа значения. "
        "Конец численной оси 1,855 м; CSV указывает длину 1,85 м. "
        "Ноль времени — начало экспорта прибора, не поверхность.\n\n"
        "Источник: Mercedes Solla; данные SFCW предоставлены Screening Eagle Technologies.\n"
        "Датасет: https://zenodo.org/records/10962520\n"
        "Статья: https://doi.org/10.1016/j.prostr.2024.09.250\n"
        "Лицензия датасета: GPL-3.0-or-later (текст: ../GPL-3.0.txt).\n"
        "Детали происхождения, контрольные суммы и обработка: sources.json.\n"
        "Пересобрать из корня проекта: .venv/bin/python user_review/realism/reference_hyperbola/build_reference.py\n"
    )
    print(json.dumps({"output": str(HERE / "real_hyperbola.png"), "shape": raw.shape,
                      "p99": lim, "original_sgy_hash_ok": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
