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

# Vacuum permittivity, CODATA 2018 (F/m), DOI 10.1103/RevModPhys.93.025010.
# Kept explicit so archived exports retain their original numerical convention.
EPSILON_0 = 8.8541878128e-12

FIELD_INFO = {
    'material_id': ('1', 'Index into material_keys; solver-exported voxel palette.'),
    'epsilon_real': ('1', 'Real relative permittivity evaluated from the solver material database at frequency_hz.'),
    'epsilon_imag': ('1', 'Positive loss: epsilon*=epsilon_real-i*epsilon_imag for exp(+i omega t); includes DC conduction.'),
    'epsilon_infinite': ('1', 'High-frequency/base relative permittivity in the solver material model.'),
    'conductivity_dc_S_m': ('S/m', 'DC electric conductivity from the solver material database.'),
    'conductivity_effective_S_m': ('S/m', 'DC conductivity plus Debye polarization loss at frequency_hz.'),
    'permeability_relative': ('1', 'Relative magnetic permeability from the solver material database.'),
    'magnetic_conductivity_S_m': ('S/m', 'gprMax magnetic conductivity parameter; original database convention.'),
    'debye_pole_count': ('1', 'Number of Debye relaxation poles for this voxel material.'),
    'object_type': ('1', '0 background/air; 1 PEC; 2 buried air cavity; 3 dielectric inclusion.'),
    'object_mask': ('1', 'One for embedded target voxels, zero elsewhere.'),
    'layer_id': ('1', 'One-based layer index; 0 air; -1 target. Native flat boundaries or custom rough layer labels.'),
    'porosity': ('m3/m3', 'Continuous generated porosity before material quantization; NaN where unavailable.'),
    'saturation': ('1', 'Continuous generated water saturation of pore space; NaN where unavailable.'),
    'water_fraction': ('m3/m3', 'Continuous generated porosity*saturation; NaN where unavailable.'),
    'continuous_epsilon_real': ('1', 'Pre-quantization CRIM field; masked outside retained soil; not the FDTD palette.'),
    'continuous_conductivity_S_m': ('S/m', 'Pre-quantization Archie field; masked outside retained soil; not the FDTD palette.'),
}


def geometry_fields(directory):
    directory = Path(directory)
    cfg = json.loads((directory/'config.json').read_text())
    scene = directory/'scene'; meta = json.loads((scene/'soil.json').read_text())
    db = json.loads((scene/meta['materials']).read_text())['materials']
    with h5py.File(scene/meta['geometry']) as h:
        ids = h['data'][:]
        keys = [x.decode() if isinstance(x, bytes) else str(x) for x in h['material_keys'][:]]
        spacing = np.asarray(h.attrs['dx_dy_dz'], dtype=float)
    if ids.min() < 0 or ids.max() >= len(keys):
        raise ValueError('Geometry material indices exceed the exported palette')
    frequency = float(cfg['source']['frequency_hz'])
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError('Material evaluation frequency must be positive')
    omega = 2*np.pi*frequency  # Angular frequency from the definition omega = 2*pi*f.
    values = {k: [] for k in ('epsilon_real','epsilon_imag','epsilon_infinite','conductivity_dc_S_m',
              'conductivity_effective_S_m','permeability_relative','magnetic_conductivity_S_m','debye_pole_count')}
    kinds = []
    for key in keys:
        item = db[key]
        if item.get('model') not in ('constant', 'debye'):
            raise ValueError(f"Unsupported material model for export: {item.get('model')}")
        base = item['base']; einf = float(base['relative_permittivity'])
        dc = float(base['electric_conductivity_s_per_m']); er = einf; loss = 0.
        poles = item.get('poles', [])
        for pole in poles:
            delta = float(pole['relative_permittivity_difference'])
            wt = omega*float(pole['relaxation_time_s'])
            # Debye susceptibility delta/(1+i*omega*tau), for exp(+i*omega*t).
            er += delta/(1+wt*wt); loss += delta*wt/(1+wt*wt)
        sigma = dc+omega*EPSILON_0*loss
        name = item.get('name', '')
        kind = 1 if name == 'pec' or np.isinf(dc) else 3 if name == 'inclusion' else 2 if name == 'free_space' else 0
        quantities = [er, loss+dc/(omega*EPSILON_0), einf, dc, sigma,
                      float(base.get('relative_permeability', 1)), float(base.get('magnetic_conductivity_s_per_m', 0)), len(poles)]
        if kind == 1:
            # PEC is a boundary idealization, not a finite constitutive material.
            quantities[:5] = [float('nan')]*5
        for field, value in zip(values, quantities):values[field].append(value)
        kinds.append(kind)
    object_type = np.asarray(kinds, dtype=np.uint8)[ids]
    z = (np.arange(ids.shape[2])+.5)*spacing[2]  # FDTD geometry values are cell-centred.
    above = z >= cfg['grid']['surface_z']; object_type[:,:,above] = 0
    fields = {'material_id': ids.astype(np.int32)}
    for key, lut in values.items():
        fields[key] = np.asarray(lut, dtype=np.int32 if key == 'debye_pole_count' else np.float32)[ids]
    fields.update(object_type=object_type, object_mask=(object_type>0).astype(np.uint8))
    layers = np.zeros(ids.shape, dtype=np.int32)
    for i, layer in enumerate(meta.get('layers', []), 1):
        layers[:,:,(z >= layer['z_bottom_m']) & (z < layer['z_top_m'])] = i
    if meta.get('custom_fields'):
        with np.load(scene/meta['custom_fields'], allow_pickle=False) as raw:
            if raw['layer_id'].shape != ids.shape:
                raise ValueError('Continuous fields and solver geometry dimensions differ')
            layers = raw['layer_id'].astype(np.int32)
            retained = raw['retained_soil_mask'].astype(bool)
            for field in ('porosity', 'saturation', 'water_fraction'):
                fields[field] = np.where(retained, raw[field], np.nan).astype(np.float32)
            fields['continuous_epsilon_real'] = np.where(retained, raw['epsilon_r'], np.nan).astype(np.float32)
            fields['continuous_conductivity_S_m'] = np.where(retained, raw['sigma_s_m'], np.nan).astype(np.float32)
    layers[object_type>0] = -1; layers[:,:,above] = 0
    fields['layer_id'] = layers
    return cfg, fields, spacing, frequency, keys


def _metadata(cfg, fields, spacing, frequency, keys):
    return dict(schema_version=1, frequency_hz=frequency, shape_xyz=list(next(iter(fields.values())).shape),
        spacing_m=spacing.tolist(), origin_m=[0,0,0], numpy_axis_order='x,y,z',
        vtk_array_order='x fastest; CellData', material_keys=keys,
        surface_z_m=cfg['grid']['surface_z'], coordinates='metres; Z upward; depth=surface_z-Z; coordinate arrays contain cell centres',
        fields={k:dict(units=FIELD_INFO[k][0], meaning=FIELD_INFO[k][1]) for k in fields},
        pec='Finite epsilon and electrical conductivity exports are NaN in PEC; use object_type=1.',
        continuous_fields='Available only for custom model; native moisture is not inferred. Continuous fields precede quantization and are not solver material values.')


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
    types = {'i4': 'Int32', 'i8': 'Int64', 'f4': 'Float32', 'f8': 'Float64', 'u1': 'UInt8'}
    for name, values in fields.items():
        dtype = values.dtype.newbyteorder('<')
        raw = values.astype(dtype, copy=False).ravel(order='F').tobytes()
        node = ET.SubElement(cells, 'DataArray', type=types[dtype.str[1:]], Name=name, NumberOfComponents='1', format='binary')
        node.text = base64.b64encode(struct.pack('<Q', len(raw)) + raw).decode('ascii')
    ET.indent(root)
    temp = Path(path).with_suffix('.vti.tmp')
    ET.ElementTree(root).write(temp, encoding='utf-8', xml_declaration=True)
    temp.replace(path)


def _npz(path, arrays):
    """Atomic, pickle-free archives: strings use fixed-width Unicode arrays."""
    arrays = {k: np.asarray(v) for k,v in arrays.items()}
    if any(v.dtype.hasobject for v in arrays.values()):
        raise ValueError('Object arrays are forbidden in portable NumPy export')
    temp = Path(path).with_suffix('.npz.tmp')
    with temp.open('wb') as stream:np.savez_compressed(stream, **arrays)
    temp.replace(path)


def export_numpy(directory, geometry=None, signals_only=False):
    directory = Path(directory); out = directory/'exports'; out.mkdir(exist_ok=True)
    if signals_only and all((out/name).is_file() for name in ('geometry.npz','materials.json','metadata.json')):
        meta = json.loads((out/'metadata.json').read_text())
    else:
        cfg, fields, spacing, frequency, keys = geometry if geometry is not None else geometry_fields(directory)
        data = dict(fields, spacing_m=spacing, frequency_hz=np.asarray(frequency), material_keys=np.asarray(keys,dtype=str))
        for axis, name in enumerate(('x_m','y_m','z_m')):
            data[name] = (np.arange(next(iter(fields.values())).shape[axis])+.5)*spacing[axis]
        _npz(out/'geometry.npz', data)
        meta = _metadata(cfg, fields, spacing, frequency, keys)
        soil = json.loads((directory/'scene/soil.json').read_text())
        (out/'materials.json').write_bytes((directory/'scene'/soil['materials']).read_bytes())
        meta['materials_file'] = 'materials.json'
    meta['signals'] = None
    h5 = directory/'bscan.h5'
    if h5.exists():
        with h5py.File(h5) as h:
            done = h['complete'][:].astype(bool)
            signals = {'complete':done}
            for key in ('time_s','source_time_s','source_current_A','tx_m','rx_m','requested_tx_m','requested_rx_m',
                        'distance_m','route_id','segment_id','route_names','seconds','crop_origin_xy_m'):
                if key in h:
                    signals[key] = np.asarray(h[key].asstr()[:], dtype=str) if h[key].dtype.kind in 'SO' else h[key][:]
            if 'Ey' in h:
                signals['Ey'] = h['Ey'][:].astype(np.float32)
                signals['Ey'][~done] = np.nan
                if not np.isfinite(signals['Ey'][done]).all():
                    raise ValueError('Completed signal contains nonfinite samples')
            else:
                # Prepared/STOP-before-first-trace files have no time lattice yet.
                signals['Ey'] = np.empty((len(done),0),dtype=np.float32)
                signals['time_s'] = np.empty(0,dtype=np.float64)
            signals['signal_units'] = np.asarray('V/m')
            meta['signals'] = dict(file='signals.npz',axis_order='scan,time',units='V/m',
                time_units='s',coordinate_units='m',source_current_units='A',complete_count=int(done.sum()),
                total=len(done),missing='Ey rows with complete=False are NaN; no time axis before first computed trace')
        # Older archives lack route metadata: use the frozen positions file.
        positions = directory/'positions.npz'
        if positions.exists():
            with np.load(positions,allow_pickle=False) as pos:
                for key in pos.files:
                    if key not in signals:signals[key] = pos[key]
        _npz(out/'signals.npz',signals)
    else:
        (out/'signals.npz').unlink(missing_ok=True)
    (out/'metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
    return out/'geometry.npz'


def _vtk_array(parent, name, values, components=1):
    values = np.asarray(values)
    kinds = {'f8':'Float64','f4':'Float32','i8':'Int64','i4':'Int32','u1':'UInt8'}
    dtype = values.dtype.newbyteorder('<')
    if dtype.str[1:] not in kinds:raise ValueError(f'Unsupported VTK dtype {dtype}')
    raw = values.astype(dtype,copy=False).tobytes(order='C')
    node=ET.SubElement(parent,'DataArray',type=kinds[dtype.str[1:]],Name=name,
                       NumberOfComponents=str(components),format='binary')
    node.text=base64.b64encode(struct.pack('<Q',len(raw))+raw).decode('ascii')


def _polydata(path, points, point_data, cells, kind):
    points=np.asarray(points,dtype=np.float64).reshape(-1,3)
    root=ET.Element('VTKFile',type='PolyData',version='1.0',byte_order='LittleEndian',header_type='UInt64')
    poly=ET.SubElement(root,'PolyData')
    piece=ET.SubElement(poly,'Piece',NumberOfPoints=str(len(points)),NumberOfVerts='0',
                        NumberOfLines=str(len(cells) if kind=='Lines' else 0),NumberOfStrips='0',
                        NumberOfPolys=str(len(cells) if kind=='Polys' else 0))
    _vtk_array(ET.SubElement(piece,'Points'),'Points',points,3)
    pd=ET.SubElement(piece,'PointData')
    for name,values in point_data.items():_vtk_array(pd,name,values)
    ET.SubElement(piece,'CellData')
    conn=ET.SubElement(piece,kind)
    _vtk_array(conn,'connectivity',np.asarray([i for cell in cells for i in cell],dtype=np.int64))
    _vtk_array(conn,'offsets',np.cumsum([len(c) for c in cells],dtype=np.int64))
    ET.indent(root);ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)


def export_overlays(directory,cfg):
    directory=Path(directory);scene=directory/'scene'
    archive=directory/'positions.npz'
    if archive.exists():
        from src.experiment_solver import load_positions
        pos=load_positions(directory);n=len(pos['tx_m'])
        points=np.concatenate([pos['tx_m'],pos['rx_m']]);lines=[]
        for sensor in range(2):
            for route in np.unique(pos['route_id']):
                ix=np.flatnonzero(pos['route_id']==route)+sensor*n
                # A singleton becomes a zero-length line so it remains a valid point marker.
                lines.append(ix.tolist() if len(ix)>1 else [int(ix[0]),int(ix[0])])
        _polydata(scene/'trajectory.vtp',points,{
            'sensor_id':np.repeat(np.arange(2,dtype=np.int32),n),
            'trace_index':np.tile(np.arange(n,dtype=np.int32),2),
            'route_id':np.tile(pos['route_id'].astype(np.int32),2),
            'segment_id':np.tile(pos['segment_id'].astype(np.int32),2),
            'distance_m':np.tile(pos['distance_m'],2)},lines,'Lines')
    # Display mesh of the requested spheres, not a replacement for voxel geometry.
    # 16 azimuth and 8 polar subdivisions are visual tessellation choices only.
    vertices=[];triangles=[];ids=[];kinds=[];radii=[]
    for i,obj in enumerate(cfg.get('experiment_objects',[])):
        center=np.asarray(obj['center_m']);r=obj['radius_m'];offset=len(vertices)
        local=[center+[0,0,r]]
        for latitude in range(1,8):
            theta=np.pi*latitude/8
            for longitude in range(16):
                phi=2*np.pi*longitude/16
                local.append(center+r*np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]))
        local.append(center+[0,0,-r]);vertices.extend(local)
        for j in range(16):
            triangles.append([offset,offset+1+j,offset+1+(j+1)%16])
            for band in range(6):
                a=offset+1+band*16+j;b=offset+1+band*16+(j+1)%16;c=a+16;d=b+16
                triangles.extend([[a,c,b],[b,c,d]])
            triangles.append([offset+len(local)-1,offset+1+6*16+(j+1)%16,offset+1+6*16+j])
        ids.extend([i]*len(local));kinds.extend([{'pec':1,'free_space':2,'inclusion':3}[obj['material']]]*len(local));radii.extend([r]*len(local))
    _polydata(scene/'objects.vtp',vertices,dict(object_id=np.asarray(ids,dtype=np.int32),
        object_type=np.asarray(kinds,dtype=np.int32),radius_m=np.asarray(radii,dtype=np.float64)),triangles,'Polys')


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
    (directory/'scene/vti_fields.json').write_text(json.dumps(_metadata(cfg,fields,spacing,frequency,keys),indent=2,ensure_ascii=False)+'\n')
    export_numpy(directory, geometry=(cfg,fields,spacing,frequency,keys))
    export_overlays(directory,cfg)
    slice_y = representative_y(directory, cfg)
    j = max(0, min(fields['epsilon_real'].shape[1] - 1, int(slice_y / spacing[1])))
    cut = fields['epsilon_real'][:, j, :]
    g = cfg['grid']
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    cmap = plt.get_cmap('viridis').with_extremes(bad='#161b22')
    im = ax.imshow(cut.T, origin='lower', extent=[0, g['domain'][0], 0, g['domain'][2]], aspect='equal', cmap=cmap, interpolation='nearest')
    ax.axhline(g['surface_z'], color='white', ls='--', lw=.8)
    ax.set(xlabel='X, м', ylabel='Z, м (вверх)', title=f"Грунт и объекты · Y = {(j+.5)*spacing[1]:.3f} м · {frequency/1e6:g} МГц")
    fig.colorbar(im, ax=ax, label='Действительная относительная проницаемость εr')
    fig.savefig(directory / 'soil_slice.png', dpi=150); plt.close(fig)
    # Orthogonal cuts are sampled at cell centres, with identical colour limits.
    i = min(fields['epsilon_real'].shape[0]-1, int(g['domain'][0]/2/spacing[0]))
    k = min(fields['epsilon_real'].shape[2]-1, max(0,int(g['surface_z']/2/spacing[2])))
    er = fields['epsilon_real']; finite = er[np.isfinite(er)]
    vmin, vmax = float(finite.min()), float(finite.max())
    fig, axes = plt.subplot_mosaic([['xy','xz'],['xy','yz']],figsize=(13,7),layout='constrained')
    pieces = [('xy',er[:,:,k].T,[0,g['domain'][0],0,g['domain'][1]],'X, м','Y, м',f'План Z = {(k+.5)*spacing[2]:.3f} м'),
              ('xz',er[:,j,:].T,[0,g['domain'][0],0,g['domain'][2]],'X, м','Z, м',f'Разрез Y = {(j+.5)*spacing[1]:.3f} м'),
              ('yz',er[i,:,:].T,[0,g['domain'][1],0,g['domain'][2]],'Y, м','Z, м',f'Разрез X = {(i+.5)*spacing[0]:.3f} м')]
    for key,values,extent,xlabel,ylabel,title in pieces:
        ax=axes[key];im=ax.imshow(values,origin='lower',extent=extent,cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest',aspect='equal')
        ax.set(xlabel=xlabel,ylabel=ylabel,title=title)
        if key!='xy':
            top=g['surface_z'];ax.axhline(top,color='white',ls='--',lw=.8)
            for layer in cfg.get('preset',{}).get('layers',[]):
                top-=layer['thickness'];ax.axhline(top,color='white',ls=':',lw=.6)
    fig.colorbar(im,ax=list(axes.values()),label=f'εr при {frequency/1e6:g} МГц',shrink=.75)
    fig.suptitle('Фактическая геометрия: три разреза; чёрный — PEC, пунктир — номинальные границы')
    fig.savefig(directory/'soil_sections.png',dpi=150,bbox_inches='tight');plt.close(fig)
    instructions = f'''ОТКРЫТЬ ЭКСПЕРИМЕНТ В PARAVIEW

1. File → Open → выберите scene/scene.vti рядом с этим файлом.
2. Нажмите Apply. При необходимости Reset Camera.
3. В поле Coloring выберите epsilon_real и Rescale to Data Range.
4. Для вертикального разреза: Filters → Slice, тип Plane.
   Origin: {g['domain'][0]/2:g}, {slice_y:g}, {g['surface_z']/2:g}.
   Normal: 0, 1, 0. Нажмите Apply; скройте исходный объём значком глаза.
5. Этот разрез выбран по медиане Y всех Tx; он не повторяет повороты маршрутов.
   Меняйте Origin и Normal для других разрезов сцены.
6. Дополнительно откройте scene/trajectory.vtp (Tx/Rx) и scene/objects.vtp (заданные сферы), Apply.
   sensor_id: 0 = Tx, 1 = Rx; object_id соответствует порядку objects.json.
   Поверхности spheres — отображение заданной геометрии; точная воксельная геометрия в scene.vti.
7. Для объектов выберите Coloring → object_type:
   0 = фон, 1 = металл, 2 = воздушная полость, 3 = диэлектрик.
   Можно применить Threshold к object_type с диапазоном от 1 до 3.

X/Y/Z и размеры в метрах. Z направлена вверх.
Поверхность: Z = {g['surface_z']:g} м. Глубина = {g['surface_z']:g} − Z.
epsilon_real, epsilon_imag и conductivity_effective_S_m вычислены при {frequency/1e6:g} МГц.
epsilon_imag — положительные полные потери, включая DC-проводимость.
conductivity_dc_S_m, permeability_relative, layer_id и поля влажности перечислены в vti_fields.json.
В идеальном металле (PEC) электрические параметры NaN: смотрите object_type.
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
    for filename, caption in [('scene_layout.png', 'Маршрут Tx/Rx и размещение объектов'), ('soil_sections.png', 'План и два вертикальных разреза фактической геометрии; общий масштаб цвета'), ('bscan.png', 'B-скан: рассчитанные позиции; серые столбцы ещё не готовы'), ('ascan.png', 'A-скан в одной рассчитанной позиции')]:
        if (directory / filename).exists():
            figures.append(f'<figure><img src="{filename}" alt="{caption}" loading="lazy"><figcaption>{caption}</figcaption></figure>')
    for path in sorted(directory.glob('bscan_[0-9][0-9][0-9].png')):
        caption = 'B-сканы следующих маршрутов; пунктир — смена сегмента'
        figures.append(f'<figure><img src="{path.name}" alt="{caption}" loading="lazy"><figcaption>{caption}</figcaption></figure>')
    entries = [
        ('source.json', 'Исходные настройки'), ('config.json', 'Зафиксированные настройки и версии'),
        ('scene_config.json', 'Исходные настройки сцены'), ('trajectory_config.json', 'Сохранённое описание маршрутов'),
        ('objects.json', 'Объекты и координаты'), ('positions.npz', 'Зафиксированные положения Tx/Rx'),
        ('trajectory.csv', 'Маршруты: заданные и округлённые координаты, путь и сегменты'),
        ('scene/scene.vti', 'Геометрия ParaView'), ('scene/scene.in', 'Сцена gprMax'),
        ('scene/vti_fields.json', 'Описание полей ParaView'), ('scene/trajectory.vtp', 'Маршруты Tx/Rx для ParaView'),
        ('scene/objects.vtp', 'Поверхности заданных объектов для ParaView'), ('exports/geometry.npz', 'NumPy: трёхмерные материалы и координаты'),
        ('exports/signals.npz', 'NumPy: сигналы, время, позиции и маска готовности'), ('exports/metadata.json', 'Единицы и определения экспортированных полей'),
        ('exports/materials.json', 'Материалы и все полюса Дебая'), ('bscan.h5', 'Численные сигналы'),
        ('effective_sfcw.npz', 'Синтетический частотный отклик'), ('status.json', 'Состояние при обновлении'),
        ('PARAVIEW.txt', 'Пошагово: ParaView'), ('logs/commands.log', 'Журнал команд'),
    ]
    links = ''.join(f'<li><a href="{file}">{title}</a> <code>{file}</code></li>' for file, title in entries if (directory / file).exists())
    commands = '\n'.join(f'python geodatagen.py {command} {directory.name}' for command in ('run', 'status', 'plot', 'export', 'docs'))
    objects = cfg.get('experiment_objects', [])
    object_rows = ''.join('<tr><td>'+str(i)+'</td><td>'+escape(str(o['type']))+'</td><td>'+', '.join(f'{v:g}' for v in o['center_m'])+'</td><td>'+f"{o['radius_m']:g}"+'</td><td>'+f"{cfg['grid']['surface_z']-o['center_m'][2]:g}"+'</td></tr>' for i,o in enumerate(objects))
    object_table = '<p>Объекты не заданы.</p>' if not objects else '<div class="table"><table><thead><tr><th>ID</th><th>Тип</th><th>Центр X, Y, Z · м</th><th>Радиус · м</th><th>Глубина центра · м</th></tr></thead><tbody>'+object_rows+'</tbody></table></div>'
    layer_rows=[];depth=0.
    for i,layer in enumerate(cfg.get('preset',{}).get('layers',[]),1):
        bottom=depth+layer['thickness']
        description = (f"песок {layer['sand']:g}; глина {layer['clay']:g}; плотность {layer['bulk_density']:g} г/см³; влажность {layer['water_min']:g}–{layer['water_max']:g} м³/м³"
                       if 'sand' in layer else f"εr = {layer['er']:g}; σ = {layer.get('sigma',0):g} См/м")
        layer_rows.append(f'<tr><td>{i}</td><td>{depth:g}–{bottom:g}</td><td>{escape(description)}</td></tr>');depth=bottom
    layer_table='<div class="table"><table><thead><tr><th>Слой</th><th>Глубина · м</th><th>Заданные свойства</th></tr></thead><tbody>'+''.join(layer_rows)+'</tbody></table></div>'
    route_rows=[]
    if archive.exists():
        from src.experiment_solver import load_positions
        pos=load_positions(directory)
        for rid in np.unique(pos['route_id']):
            ix=np.flatnonzero(pos['route_id']==rid)
            route_rows.append(f"<tr><td>{escape(str(pos['route_names'][int(rid)]))}</td><td>{len(ix)}</td><td>{pos['distance_m'][ix[-1]]:.3f}</td></tr>")
    route_table='<div class="table"><table><thead><tr><th>Маршрут</th><th>Позиции</th><th>Длина · м</th></tr></thead><tbody>'+''.join(route_rows)+'</tbody></table></div>'
    pending = '<p class="muted">Это предпросмотр подготовленной сцены. Сигналы ещё не рассчитаны.</p>' if not done else ''
    text = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{name} · Geodatagen</title>
<style>:root{{color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;font:16px/1.65 system-ui,sans-serif;color:#1c2633;background:#fff}}header,main,footer{{max-width:1100px;margin:auto;padding:24px}}header{{border-bottom:1px solid #e5e7eb}}h1{{font-size:34px;margin:8px 0}}h2{{margin-top:40px}}a{{color:#126a58}}code,pre{{font-family:ui-monospace,monospace;font-size:14px}}pre{{background:#f4f6f7;padding:20px;overflow-x:auto;border:1px solid #e4e8ec;border-radius:8px}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #e5e7eb}}.table{{overflow-x:auto}}figure{{margin:28px 0}}img{{display:block;width:100%;height:auto}}figcaption,footer,.muted{{color:#596674;font-size:14px}}li{{margin:9px 0}}li code{{margin-left:12px}}.status{{display:inline-block;padding:4px 12px;background:#edf6f2;border-radius:6px}}@media(max-width:600px){{header,main,footer{{padding:18px}}h1{{font-size:28px}}li code{{display:block;margin:0}}}}</style></head><body><header><p class="muted">Geodatagen / Эксперимент</p><h1>{name}</h1><p><span class="status">{label} · {done}/{total} трасс</span></p><p>Модель грунта: <code>{escape(cfg['generator'])}</code> · seed {cfg['seed']}</p><p>Сцена: {scene_size} м · расчёт: <code>{escape(mode)}</code>{route_summary}</p></header><main>
<p>Все материалы этого опыта находятся в одной папке. Отчёт обновляется командами <code>run</code>, <code>plot</code> и <code>export</code>. Во время расчёта актуальный прогресс показывает <code>status</code>.</p>
{pending}{''.join(figures)}<h2>Маршруты</h2>{route_table}<h2>Объекты</h2>{object_table}<h2>Слои грунта</h2>{layer_table}<p class="muted">В таблице номинальные границы и входные параметры. У модели полей внутренние границы могут быть неровными; экспортированные поля показывают фактическое распределение.</p><h2>Продолжить работу</h2><p>Выполняйте из корня скачанного репозитория, в активированном окружении:</p><pre>{escape(commands)}</pre><p>Параметры готового опыта зафиксированы. Для изменений создайте новый эксперимент с другим именем и новым JSON.</p>
<h2>Открыть сцену в ParaView</h2><ol><li>File → Open → <code>scene/scene.vti</code> → Apply.</li><li>Выберите Coloring → <code>epsilon_real</code>, затем Rescale to Data Range.</li><li>Filters → Slice: Normal = <code>0, 1, 0</code>, Origin = <code>{cfg['grid']['domain'][0]/2:g}, {slice_y:g}, {cfg['grid']['surface_z']/2:g}</code>. Apply; скройте исходный объём.</li><li>Откройте дополнительно <code>scene/trajectory.vtp</code> и <code>scene/objects.vtp</code>, затем Apply: маршруты и заданные поверхности объектов будут в той же системе координат.</li><li>Для объектов выберите поле <code>object_type</code>: 1 — металл, 2 — полость, 3 — диэлектрик.</li></ol><p>Разрез проходит по медиане Y всех положений Tx; повороты маршрута показаны на плане. Координаты в метрах, Z вверх. Глубина = {cfg['grid']['surface_z']:g} − Z. <a href="PARAVIEW.txt">Подробная инструкция</a>.</p>
<h2>Файлы этого опыта</h2><ul>{links}</ul><p class="muted">Синтетический сигнал рассчитан для идеальных Tx/Rx. Частотная обработка не является калибровкой под конкретный реальный прибор.</p></main><footer>Обновлено {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · Отчёт открывается без сервера.</footer></body></html>'''
    (directory / 'index.html').write_text(text)
    return directory / 'index.html'
