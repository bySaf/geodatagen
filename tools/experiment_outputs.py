"""Portable per-experiment geometry export and an offline results page."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import h5py
import numpy as np

EPSILON_0 = 8.8541878128e-12


def geometry_fields(directory):
    directory = Path(directory)
    cfg = json.loads((directory / 'config.json').read_text())
    scene = directory / 'scene'
    meta = json.loads((scene / 'soil.json').read_text())
    db = json.loads((scene / meta['materials']).read_text())['materials']
    with h5py.File(scene / meta['geometry'], 'r') as h:
        ids = h['data'][:]
        keys = [x.decode() if isinstance(x, bytes) else str(x) for x in h['material_keys'][:]]
        spacing = np.asarray(h.attrs['dx_dy_dz'], dtype=float)
    frequency = float(cfg['source']['frequency_hz'])
    omega = 2 * np.pi * frequency
    epsilon, conductivity, kinds = [], [], []
    for key in keys:
        item = db[key]
        if item.get('model') not in ('constant', 'debye'):
            raise ValueError(f"Unsupported material model for export: {item.get('model')}")
        base = item['base']
        er = float(base['relative_permittivity'])
        sigma = float(base['electric_conductivity_s_per_m'])
        for pole in item.get('poles', []):
            delta = float(pole['relative_permittivity_difference'])
            wt = omega * float(pole['relaxation_time_s'])
            er += delta / (1 + wt * wt)
            sigma += omega * EPSILON_0 * delta * wt / (1 + wt * wt)
        name = item.get('name', '')
        if name == 'pec' or np.isinf(sigma):
            er = sigma = float('nan')
            kind = 1
        elif name == 'inclusion':
            kind = 3
        elif name == 'free_space':
            kind = 2
        else:
            kind = 0
        epsilon.append(er); conductivity.append(sigma); kinds.append(kind)
    object_type = np.asarray(kinds, dtype=np.uint8)[ids]
    above = (np.arange(ids.shape[2]) + .5) * spacing[2] >= cfg['grid']['surface_z']
    object_type[:, :, above] = 0
    fields = {
        'material_id': ids.astype(np.int32),
        'epsilon_real': np.asarray(epsilon, dtype=np.float32)[ids],
        'conductivity_effective_S_m': np.asarray(conductivity, dtype=np.float32)[ids],
        'object_type': object_type,
    }
    return cfg, fields, spacing, frequency, keys


def write_vti(path, fields, spacing, frequency):
    """Write standard XML ImageData CellData; x is the fastest VTK index."""
    shape = next(iter(fields.values())).shape
    if any(a.shape != shape for a in fields.values()) or len(shape) != 3:
        raise ValueError('All VTI arrays must have the same three-dimensional shape')
    extent = f'0 {shape[0]} 0 {shape[1]} 0 {shape[2]}'
    root = ET.Element('VTKFile', type='ImageData', version='1.0', byte_order='LittleEndian', header_type='UInt64')
    image = ET.SubElement(root, 'ImageData', WholeExtent=extent, Origin='0 0 0', Spacing=' '.join(map(str, spacing)))
    field_data = ET.SubElement(image, 'FieldData')
    ET.SubElement(field_data, 'DataArray', type='Float64', Name='evaluation_frequency_hz', NumberOfTuples='1', format='ascii').text = str(frequency)
    piece = ET.SubElement(image, 'Piece', Extent=extent)
    ET.SubElement(piece, 'PointData')
    cells = ET.SubElement(piece, 'CellData', Scalars='epsilon_real')
    types = {'i4': 'Int32', 'f4': 'Float32', 'u1': 'UInt8'}
    for name, values in fields.items():
        dtype = values.dtype.newbyteorder('<')
        raw = values.astype(dtype, copy=False).ravel(order='F').tobytes()
        node = ET.SubElement(cells, 'DataArray', type=types[dtype.str[1:]], Name=name, NumberOfComponents='1', format='binary')
        node.text = base64.b64encode(struct.pack('<Q', len(raw)) + raw).decode('ascii')
    ET.indent(root)
    temp = Path(path).with_suffix('.vti.tmp')
    ET.ElementTree(root).write(temp, encoding='utf-8', xml_declaration=True)
    temp.replace(path)


def representative_y(directory, cfg):
    archive = Path(directory) / 'positions.npz'
    if archive.exists():
        with np.load(archive, allow_pickle=False) as positions:
            return float(np.median(positions['tx_m'][:, 1]))
    return float(cfg.get('profile', {}).get('y_m', cfg['grid']['domain'][1]/2))


def export_geometry(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    directory = Path(directory)
    cfg, fields, spacing, frequency, keys = geometry_fields(directory)
    path = directory / 'scene' / 'scene.vti'
    write_vti(path, fields, spacing, frequency)
    (directory / 'scene' / 'vti_fields.json').write_text(json.dumps({
        'frequency_hz': frequency, 'spacing_m': spacing.tolist(),
        'cell_order': 'x fastest, VTK CellData', 'material_keys': keys,
        'epsilon_real': 'Real relative permittivity evaluated at source frequency; NaN in PEC.',
        'conductivity_effective_S_m': 'DC conduction plus Debye dielectric loss at source frequency; NaN in PEC.',
        'object_type': {'0': 'background / soil / above-ground air', '1': 'PEC metal', '2': 'air cavity below surface', '3': 'dielectric inclusion'},
        'coordinates': 'metres; Z upward; depth = surface_z - Z',
        'surface_z_m': cfg['grid']['surface_z'],
    }, indent=2, ensure_ascii=False) + '\n')
    slice_y = representative_y(directory, cfg)
    j = max(0, min(fields['epsilon_real'].shape[1] - 1, int(slice_y / spacing[1])))
    cut = fields['epsilon_real'][:, j, :]
    g = cfg['grid']
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#161b22')
    im = ax.imshow(cut.T, origin='lower', extent=[0, g['domain'][0], 0, g['domain'][2]], aspect='equal', cmap=cmap, interpolation='nearest')
    ax.axhline(g['surface_z'], color='white', ls='--', lw=.8)
    ax.set(xlabel='X, м', ylabel='Z, м (вверх)', title=f"Грунт и объекты · Y = {(j+.5)*spacing[1]:.3f} м · {frequency/1e6:g} МГц")
    fig.colorbar(im, ax=ax, label='Действительная относительная проницаемость εr')
    fig.savefig(directory / 'soil_slice.png', dpi=150); plt.close(fig)
    instructions = f'''ОТКРЫТЬ ЭКСПЕРИМЕНТ В PARAVIEW

1. File → Open → выберите scene/scene.vti рядом с этим файлом.
2. Нажмите Apply. При необходимости Reset Camera.
3. В поле Coloring выберите epsilon_real и Rescale to Data Range.
4. Для вертикального разреза: Filters → Slice, тип Plane.
   Origin: {g['domain'][0]/2:g}, {slice_y:g}, {g['surface_z']/2:g}.
   Normal: 0, 1, 0. Нажмите Apply; скройте исходный объём значком глаза.
5. Этот разрез выбран по медиане Y всех Tx; он не повторяет повороты маршрутов.
   Меняйте Origin и Normal для других разрезов сцены.
6. Для объектов выберите Coloring → object_type:
   0 = фон, 1 = металл, 2 = воздушная полость, 3 = диэлектрик.
   Можно применить Threshold к object_type с диапазоном от 1 до 3.

X/Y/Z и размеры в метрах. Z направлена вверх.
Поверхность: Z = {g['surface_z']:g} м. Глубина = {g['surface_z']:g} − Z.
epsilon_real и conductivity_effective_S_m вычислены при {frequency/1e6:g} МГц.
В идеальном металле (PEC) эти два поля NaN: смотрите object_type.
material_id — номер материала, это не диэлектрическая проницаемость.

В ParaView открывается геометрия scene.vti. A/B-сканы находятся в ascan.png,
bscan.png и bscan.h5; общий HDF5-файл сигналов не является VTK-геометрией.
Сохраняйте свои состояния ParaView в эту папку: File → Save State.
После переноса состояния на другой компьютер при загрузке укажите новый
путь к scene/scene.vti. Сам VTI переносится без пересчёта.
'''
    (directory / 'PARAVIEW.txt').write_text(instructions)
    return path


def report(directory, status):
    directory = Path(directory)
    cfg = json.loads((directory / 'config.json').read_text())
    name = escape(directory.name)
    slice_y = representative_y(directory, cfg)
    mode = cfg.get('simulation', {}).get('mode', 'fixed')
    scene_size = ' × '.join(f'{v:g}' for v in cfg['grid']['domain'])
    route_summary = ''
    archive = directory / 'positions.npz'
    if archive.exists():
        with np.load(archive, allow_pickle=False) as positions:
            route_count = len(np.unique(positions['route_id'])) if 'route_id' in positions else 1
            route_summary = f' · маршрутов: {route_count}'
    (directory / 'status.json').write_text(json.dumps(status, indent=2, ensure_ascii=False) + '\n')
    done = status.get('completed', 0)
    total = status.get('total', cfg['profile']['n'])
    running = status.get('running', False)
    label = 'Идёт расчёт' if running else ('Готово' if done == total else 'Подготовлен' if not done else 'Рассчитан частично')
    figures = []
    for filename, caption in [('scene_layout.png', 'Маршрут Tx/Rx и размещение объектов'), ('soil_slice.png', 'Разрез фактической геометрии сцены'), ('bscan.png', 'B-скан: рассчитанные позиции; серые столбцы ещё не готовы'), ('ascan.png', 'A-скан в одной рассчитанной позиции')]:
        if (directory / filename).exists():
            figures.append(f'<figure><img src="{filename}" alt="{caption}" loading="lazy"><figcaption>{caption}</figcaption></figure>')
    for path in sorted(directory.glob('bscan_[0-9][0-9][0-9].png')):
        caption = 'B-сканы следующих маршрутов; пунктир — смена сегмента'
        figures.append(f'<figure><img src="{path.name}" alt="{caption}" loading="lazy"><figcaption>{caption}</figcaption></figure>')
    entries = [
        ('source.json', 'Исходные настройки'), ('config.json', 'Зафиксированные настройки и версии'),
        ('scene_config.json', 'Исходные настройки сцены'), ('trajectory_config.json', 'Исходное описание маршрутов'),
        ('objects.json', 'Объекты и координаты'), ('positions.npz', 'Зафиксированные положения Tx/Rx'),
        ('trajectory.csv', 'Маршруты: заданные и округлённые координаты, путь и сегменты'),
        ('scene/scene.vti', 'Геометрия ParaView'), ('scene/scene.in', 'Сцена gprMax'),
        ('scene/vti_fields.json', 'Описание полей ParaView'), ('bscan.h5', 'Численные сигналы'),
        ('effective_sfcw.npz', 'Синтетический частотный отклик'), ('status.json', 'Состояние при обновлении'),
        ('PARAVIEW.txt', 'Пошагово: ParaView'), ('logs/commands.log', 'Журнал команд'),
    ]
    links = ''.join(f'<li><a href="{file}">{title}</a> <code>{file}</code></li>' for file, title in entries if (directory / file).exists())
    commands = '\n'.join(f'python geodatagen.py {command} {directory.name}' for command in ('run', 'status', 'plot', 'export', 'docs'))
    text = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{name} · Geodatagen</title>
<style>:root{{color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;font:16px/1.65 system-ui,sans-serif;color:#1c2633;background:#fff}}header,main,footer{{max-width:1100px;margin:auto;padding:24px}}header{{border-bottom:1px solid #e5e7eb}}h1{{font-size:34px;margin:8px 0}}h2{{margin-top:40px}}a{{color:#126a58}}code,pre{{font-family:ui-monospace,monospace;font-size:14px}}pre{{background:#f4f6f7;padding:20px;overflow-x:auto;border:1px solid #e4e8ec;border-radius:8px}}figure{{margin:28px 0}}img{{display:block;width:100%;height:auto}}figcaption,footer,.muted{{color:#596674;font-size:14px}}li{{margin:9px 0}}li code{{margin-left:12px}}.status{{display:inline-block;padding:4px 12px;background:#edf6f2;border-radius:6px}}@media(max-width:600px){{header,main,footer{{padding:18px}}h1{{font-size:28px}}li code{{display:block;margin:0}}}}</style></head><body><header><p class="muted">Geodatagen / Эксперимент</p><h1>{name}</h1><p><span class="status">{label} · {done}/{total} трасс</span></p><p>Модель грунта: <code>{escape(cfg['generator'])}</code> · seed {cfg['seed']}</p><p>Сцена: {scene_size} м · расчёт: <code>{escape(mode)}</code>{route_summary}</p></header><main>
<p>Все материалы этого опыта находятся в одной папке. Отчёт обновляется командами <code>run</code>, <code>plot</code> и <code>export</code>. Во время расчёта актуальный прогресс показывает <code>status</code>.</p>
{''.join(figures)}<h2>Продолжить работу</h2><p>Выполняйте из корня скачанного репозитория, в активированном окружении:</p><pre>{escape(commands)}</pre><p>Параметры готового опыта зафиксированы. Для изменений создайте новый эксперимент с другим именем и новым JSON.</p>
<h2>Открыть сцену в ParaView</h2><ol><li>File → Open → <code>scene/scene.vti</code> → Apply.</li><li>Выберите Coloring → <code>epsilon_real</code>, затем Rescale to Data Range.</li><li>Filters → Slice: Normal = <code>0, 1, 0</code>, Origin = <code>{cfg['grid']['domain'][0]/2:g}, {slice_y:g}, {cfg['grid']['surface_z']/2:g}</code>. Apply; скройте исходный объём.</li><li>Для объектов выберите поле <code>object_type</code>: 1 — металл, 2 — полость, 3 — диэлектрик.</li></ol><p>Разрез проходит по медиане Y всех положений Tx; повороты маршрута показаны на плане. Координаты в метрах, Z вверх. Глубина = {cfg['grid']['surface_z']:g} − Z. <a href="PARAVIEW.txt">Подробная инструкция</a>.</p>
<h2>Файлы этого опыта</h2><ul>{links}</ul><p class="muted">Синтетический сигнал рассчитан для идеальных Tx/Rx. Частотная обработка не является калибровкой под конкретный реальный прибор.</p></main><footer>Обновлено {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · Отчёт открывается без сервера.</footer></body></html>'''
    (directory / 'index.html').write_text(text)
    return directory / 'index.html'
