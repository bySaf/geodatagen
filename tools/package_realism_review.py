#!/usr/bin/env python3
"""Package the completed park review, without modifying its live source files.

Run from the original project after all pilot and review generation has finished:
    .venv/bin/python tools/package_realism_review.py

Only this script's execution reads completed pilot HDF5 files. The script is
deliberately not an automatic wait loop and never starts or stops simulations.
"""
from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import posixpath
import re
import tempfile
from urllib.parse import unquote, urlparse
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "user_review" / "realism"
HOME_PAGE = SOURCE.parent / "index.html"
OUTPUT = PROJECT / "user_review" / "park-realism-review.zip"
BUNDLE = "park-realism-review"
MODES = ("native_peplinski", "custom_crim_archie")
SAL_IMAGE = "reference_radar/soil_golfcourse_measured_context.png"
SAL_URL = (
    "https://opensource.silicon-austria.com/xuc/"
    "soil-analysis-machine-learning-stepped-frequency-gpr"
)
FORBIDDEN_NAMES = {
    "Radar_data_emi_data.pkl",
    "soil_golfcourse_frequency_readings.npz",
    "soil_golfcourse_measured_context.png",
}
REQUIRED = [
    "index.html", "guide.html", "park-pilot.json", "ATTRIBUTION.txt", "GPL-3.0.txt",
    "soil_three_way.png", "bscan_three_way.png", "source_and_ascans.png", "pilot_route.png",
    "spectra_three_way.png", "target_checks.png",
    "comparison_metrics.json", "soil_comparison_metrics.json", "material_validation.json",
    "pilot/README.txt", "pilot/pilot_metrics.json", "pilot/source_processing_check.json",
    "soil_models.png", "soil_depth_profile.png", "processing_effect.png", "sampling_comparison.png",
    "effective_sweep_gray.png", "standardized_comparison.npz", "validation_v2.json",
    "synthetic_hyperbolas.png", "reference_hyperbola/real_hyperbola.png", "reference_hyperbola/sources.json",
    "reference_radar/README.txt", "reference_radar/sources.json",
    "reference_radar/read_reference_sfcw.py", "reference_radar/SFCW.zip",
    "reference_radar/reference_bscan_native.png",
]
for _mode in MODES:
    REQUIRED.extend(
        f"pilot/{_mode}/{name}"
        for name in (
            "progress.json", "config.json", "bscan.h5", "effective_sfcw.npz",
            "effective_sfcw.json", "center_checks.json", "scene/scene.in",
            "scene/global.h5", "scene/global_materials.json", "scene/soil.json",
        )
    )
REQUIRED.extend(f"reference_radar/concrete_specimen_{s}.npz" for s in ("I", "II", "III"))
REQUIRED.extend(
    f"pilot/custom_crim_archie/scene/{name}"
    for name in ("custom_seed_geometry.h5", "custom_seed_materials.json", "custom_fields.npz")
)


class PackagingError(RuntimeError):
    pass


class HtmlAudit(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ("href", "src") and value:
                self.links.append(value)

    def handle_data(self, data):
        self.text.append(data)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackagingError(f"Нельзя прочитать завершённый JSON: {path}: {exc}") from exc


def stamp(path: Path) -> tuple[int, int]:
    s = path.stat()
    return s.st_size, s.st_mtime_ns


def assert_source_file(path: Path) -> None:
    if not path.is_file():
        raise PackagingError(f"Файл ещё не готов: {path.relative_to(SOURCE)}")
    if path.is_symlink() or not path.resolve().is_relative_to(SOURCE.resolve()):
        raise PackagingError(f"Внешняя или символическая ссылка не упаковывается: {path}")
    if path.stat().st_size == 0:
        raise PackagingError(f"Пустой файл: {path.relative_to(SOURCE)}")


def ready_before_hdf5() -> tuple[str, tuple[int, int]]:
    """Do not open any pilot HDF5 until every expected final artifact is ready."""
    missing = [name for name in REQUIRED if not (SOURCE / name).is_file()]
    if missing:
        raise PackagingError("Пилот ещё не готов. Отсутствуют:\n  " + "\n  ".join(missing))
    for name in REQUIRED:
        assert_source_file(SOURCE / name)
    for mode in MODES:
        p = read_json(SOURCE / "pilot" / mode / "progress.json")
        if p.get("completed") != 29 or p.get("total") != 29:
            raise PackagingError(f"{mode}: progress не подтверждает 29/29 трасс")
    html_path = SOURCE / "index.html"
    html_stamp = stamp(html_path)
    text = html_path.read_text(encoding="utf-8")
    if stamp(html_path) != html_stamp:
        raise PackagingError("index.html изменился во время проверки готовности")
    parsed = HtmlAudit()
    parsed.feed(text)
    plain = " ".join(" ".join(parsed.text).lower().split())
    pending_patterns = (
        r"расч[её]т.{0,90}продолжается",
        r"итоговые метрики будут добавлены",
        r"этот блок намеренно не содержит оценки",
    )
    if any(re.search(pattern, plain) for pattern in pending_patterns):
        raise PackagingError("В index.html остался текст ожидания незавершённого пилота")
    # A valid JSON document is required, not merely a file being written.
    for name in REQUIRED:
        if name.endswith(".json"):
            read_json(SOURCE / name)
    return text, html_stamp


def check_pilots() -> dict:
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise PackagingError("Запустите скрипт Python из .venv исходного проекта (нужны h5py и numpy)") from exc
    result = {}
    for mode in MODES:
        path = SOURCE / "pilot" / mode / "bscan.h5"
        before = stamp(path)
        try:
            with h5py.File(path, "r") as h:
                complete = h["complete"][:]
                if complete.shape != (29,) or not np.all(complete):
                    raise PackagingError(f"{mode}: HDF5 не подтверждает 29/29 complete")
                y = h["Ey"]
                if y.ndim != 2 or y.shape[0] != 29 or y.shape[1] < 2:
                    raise PackagingError(f"{mode}: неожиданная форма Ey: {y.shape}")
                if h["time_s"].shape != (y.shape[1],):
                    raise PackagingError(f"{mode}: длина оси времени не совпадает с Ey")
                for i in range(29):
                    trace = y[i]
                    if not np.isfinite(trace).all() or not np.any(trace != 0):
                        raise PackagingError(f"{mode}: некорректная или пустая трасса {i}")
                result[mode] = {"traces": 29, "samples": y.shape[1], "all_complete": True, "finite_nonzero": True}
        except (OSError, KeyError) as exc:
            raise PackagingError(f"{mode}: файл ещё пишется либо имеет неверную структуру: {exc}") from exc
        if stamp(path) != before:
            raise PackagingError(f"{mode}: HDF5 изменился во время проверки; повторите после завершения")
    return result


def bundled_html(original: str) -> bytes:
    # Change the bundle copy only. Do not edit the local report or distribute
    # the separately downloaded SAL data whose redistribution license is unclear.
    pattern = re.compile(
        r'<a\b[^>]*\bhref=(["\'])' + re.escape(SAL_IMAGE) + r'\1[^>]*>.*?</a>',
        flags=re.I | re.S,
    )
    matches = list(pattern.finditer(original))
    if not matches:
        return original.encode('utf-8')
    if len(matches) != 1:
        raise PackagingError(f"Ожидалась одна ссылка на SAL-изображение, найдено {len(matches)}")
    replacement = (
        f'<a href="{SAL_URL}">официальные данные травяного участка '
        '(внешний набор авторов; не включён в архив)</a>'
    )
    return pattern.sub(lambda _: replacement, original, count=1).encode("utf-8")


def collect_files() -> list[Path]:
    selected: set[Path] = set()
    for p in SOURCE.iterdir():
        if p.is_file() and p.suffix.lower() in {".html", ".png", ".npz", ".json", ".txt", ".md"}:
            selected.add(p)
    selected.update(p for p in (SOURCE/'examples').glob('*.json') if p.is_file())
    selected.update(p for p in (SOURCE/'reference_hyperbola').rglob('*')
                    if p.is_file() and p.name != 'candidates_concrete.png' and '__pycache__' not in p.parts)
    pilot = SOURCE / "pilot"
    for p in pilot.iterdir():
        if p.is_file() and p.suffix.lower() in {".json", ".txt", ".md"}:
            selected.add(p)
    for mode in MODES:
        folder = pilot / mode
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in {".h5", ".npz", ".json", ".txt", ".md", ".png"}:
                selected.add(p)
        # Full scene folder preserves relative references to native/custom seed
        # geometry and materials. The much larger per-trace run folder is omitted.
        selected.update(p for p in (folder / "scene").rglob("*") if p.is_file())
        for check in ("background_center", "pml_center"):
            selected.update(p for p in (folder/check).iterdir()
                            if p.is_file() and p.suffix.lower() in {".h5", ".npz", ".json"})
    radar = SOURCE / "reference_radar"
    for name in (
        "README.txt", "sources.json", "read_reference_sfcw.py", "SFCW.zip",
        "concrete_specimen_I.npz", "concrete_specimen_II.npz", "concrete_specimen_III.npz",
        "reference_bscan_native.png", "reference_bscan.png", "reference_bscans_all.png",
        "zenodo_record.json", "reproduction_log.json",
    ):
        p = radar / name
        if p.is_file():
            selected.add(p)
    for p in selected:
        assert_source_file(p)
        if p.name in FORBIDDEN_NAMES or any(x in p.relative_to(SOURCE).parts for x in ("_deps", "run", "__pycache__")):
            raise PackagingError(f"Запрещённый файл попал в отбор: {p}")
    return sorted(selected, key=lambda p: p.relative_to(SOURCE).as_posix())


def validate_archive(path: Path) -> tuple[int, int]:
    with zipfile.ZipFile(path) as z:
        members = set(z.namelist())
        bad_crc = z.testzip()
        if bad_crc:
            raise PackagingError(f"Повреждённый элемент ZIP: {bad_crc}")
        unresolved = []
        checked = 0
        for html_name in sorted(n for n in members if n.endswith('.html')):
            parser=HtmlAudit();parser.feed(z.read(html_name).decode('utf-8'))
            for link in parser.links:
                u = urlparse(link)
                if u.scheme or u.netloc or not u.path:
                    continue
                rel = unquote(u.path)
                target = posixpath.normpath(posixpath.join(posixpath.dirname(html_name), rel))
                if not target.startswith(BUNDLE + "/") or target not in members:
                    unresolved.append((link, target))
                checked += 1
        if unresolved:
            detail = "\n  ".join(f"{a} → {b}" for a, b in unresolved)
            raise PackagingError("Неразрешённые внутренние href/src в архиве:\n  " + detail)
        for member in members:
            parts = Path(member).parts
            if Path(member).name in FORBIDDEN_NAMES or any(x in parts for x in ("_deps", "run", "__pycache__")):
                raise PackagingError(f"Запрещённый элемент ZIP: {member}")
        return len(members), checked


def main() -> None:
    # Readiness gates must precede opening any live HDF5.
    original, html_stamp = ready_before_hdf5()
    html = bundled_html(original)
    home_stamp=stamp(HOME_PAGE)
    home_html=HOME_PAGE.read_text(encoding='utf-8')
    # The standalone package has its own landing page. Local-only legacy scenes
    # and a link to this ZIP itself are not meaningful inside the ZIP.
    home_html=re.sub(r'<a\b[^>]*href=["\']archive-scenes\.html["\'][^>]*>(.*?)</a>',
                     r'<span>\1 (в исходном проекте)</span>',home_html,flags=re.S)
    home_html=re.sub(r'<a\b[^>]*href=["\']park-realism-review\.zip["\'][^>]*>(.*?)</a>',
                     '<span>Отчёт и данные уже распакованы в этой папке</span>',home_html,flags=re.S)
    checks = check_pilots()
    files = collect_files()
    # The GitBook-style home embeds all figures and uses shared offline assets.
    site_files = []
    for folder in ('assets', 'sections', 'route', 'objects', 'scenes'):
        site_files.extend(p for p in (SOURCE.parent/folder).iterdir()
                          if p.is_file() and p.suffix in {'.png', '.html', '.css', '.js', '.csv', '.json', '.txt'})
    site_files.append(SOURCE.parent/'soil_modes/soil_modes_comparison.png')
    site_stamps = {p: stamp(p) for p in site_files}
    before = {p: stamp(p) for p in files}
    if before[SOURCE / "index.html"] != html_stamp:
        raise PackagingError("index.html обновлён после проверки; повторите упаковку")
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = f"""ПАРК: КОМПЛЕКТ ДЛЯ ПРОСМОТРА
Собран: {generated_at}

Откройте index.html в браузере после распаковки всей папки: это главная
страница документации проекта с инструкциями и примерами.
В комплекте: русский отчёт, рисунки, конфигурация, метрики, два завершённых
профиля по 29 трасс, эффективные частотные результаты и исходные сцены.
Оба HDF5 проверены: 29/29 complete; трассы конечны и не являются нулевыми.

Реальные данные: SFCW арматуры в бетоне (другой прибор и материал).
Авторство и происхождение: ATTRIBUTION.txt и reference_radar/sources.json.
Условия распространения SFCW: GPL-3.0.txt.
Немецкий ERT-парк исключён из отчёта и этого архива: измеренная величина
не подходит для прямого сравнения с проницаемостью наших сцен.

Не включены: рабочие run-папки, повторные промежуточные файлы, _deps,
публичный SAL pickle, его численный экспорт и изображение. Для этого
SAL-набора не установлена явная лицензия на перепубликацию. Если в HTML
остаётся ссылка на его локальный рисунок, в копии архива она заменяется
ссылкой на официальный репозиторий. Локальный отчёт при упаковке не меняется.

Это комплект просмотра и данных, а не переносимое установленное окружение.
Для полного пересчёта двух моделей нужен исходный проект georadar-start
с gprMax, Python-окружением и его инструментами. Не запускайте scene.in
в отрыве от соседних файлов геометрии и материалов. Исходные SFCW SEG-Y
сохранены внутри reference_radar/SFCW.zip; для отдельного повторного разбора
архив нужно распаковать в структуру raw/SFCW, описанную в README источника.

Проверки профилей:
{json.dumps(checks, ensure_ascii=False, indent=2)}
"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".park-realism-review-", suffix=".zip.tmp", dir=OUTPUT.parent)
    import os
    os.close(fd)
    temp = Path(temporary)
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
            z.writestr(f"{BUNDLE}/index.html",home_html)
            for p in site_files:
                z.write(p, f"{BUNDLE}/{p.relative_to(SOURCE.parent).as_posix()}")
            for p in files:
                archive_name = f"{BUNDLE}/realism/{p.relative_to(SOURCE).as_posix()}"
                if p.name == "index.html" and p.parent == SOURCE:
                    z.writestr(archive_name, html)
                else:
                    # Already compressed containers and images gain little from
                    # repeated compression. Geometry/HDF5/text use Deflate.
                    compression = zipfile.ZIP_STORED if p.suffix.lower() in {".zip", ".npz", ".png"} else zipfile.ZIP_DEFLATED
                    z.write(p, archive_name, compress_type=compression)
                if stamp(p) != before[p]:
                    raise PackagingError(f"Файл изменился при упаковке: {p.relative_to(SOURCE)}")
            z.writestr(f"{BUNDLE}/BUNDLE-MANIFEST.txt", manifest)
        # Detect changes to earlier files while later, larger files were zipped.
        if stamp(HOME_PAGE)!=home_stamp or any(stamp(p) != before[p] for p in files) or any(stamp(p) != v for p,v in site_stamps.items()):
            raise PackagingError("Исходные данные изменились во время упаковки; повторите после завершения")
        count, checked = validate_archive(temp)
        temp.replace(OUTPUT)  # Existing successful bundle is replaced only now.
        print(f"Готово: {OUTPUT}")
        print(f"Файлов: {count}; проверено локальных href/src: {checked}; размер: {OUTPUT.stat().st_size / 1024**2:.1f} МиБ")
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except PackagingError as exc:
        raise SystemExit(f"Архив не создан: {exc}")
