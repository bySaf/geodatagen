"""Export comparable physical-property sections from the saved solver geometry."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch

DEST = ROOT / 'user_review/sections'
DEST.mkdir(parents=True, exist_ok=True)
SCENES = [('park', 'Парк'), ('field', 'Поле — контроль без объектов'), ('road', 'Дорога')]
FREQUENCY = 400e6
CMAP = plt.get_cmap('viridis').copy()
CMAP.set_bad('#242424')
VMIN, VMAX = 1, 18


def read_scene(scene):
    base = ROOT / 'data/final' / scene
    meta = json.loads((base / 'soil.json').read_text())
    database = json.loads((base / meta['materials']).read_text())['materials']
    with h5py.File(base / meta['geometry']) as f:
        voxels = f['data'][:]
        keys = [v.decode() for v in f['material_keys'][:]]
    epsilon = []
    for key in keys:
        m = database[key]
        e = float(m['base']['relative_permittivity'])
        for pole in m.get('poles', []):
            e += pole['relative_permittivity_difference'] / (1 + (2*np.pi*FREQUENCY*pole['relaxation_time_s'])**2)
        epsilon.append(np.nan if m['name'] == 'pec' else e)
    objects = json.loads((base / 'objects.json').read_text())
    return meta, voxels, np.asarray(epsilon), objects


def plot_xy(ax, item, title, draw_objects=True):
    meta, voxels, eps, objects = item
    dx = meta['dx']; domain = meta['domain']; surface = meta['surface_z']
    k = int(np.floor((surface - .45) / dx + 1e-8))
    z = (k + .5)*dx
    im = ax.imshow(eps[voxels[:, :, k]].T, origin='lower', extent=(0,domain[0],0,domain[1]),
                   cmap=CMAP, vmin=VMIN, vmax=VMAX, interpolation='nearest', aspect='equal')
    if draw_objects:
        for obj in objects:
            x,y,zc = obj['center_m']; r=obj['radius_m']
            intersects = abs(z-zc) < r
            # Dashed outlines are projections only; the voxel field itself is unchanged.
            ax.add_patch(Circle((x,y),r,fill=False,ec='white',lw=1,ls='--'))
            if intersects:
                ax.add_patch(Circle((x,y),np.sqrt(r*r-(z-zc)**2),fill=False,ec='white',lw=1.2))
            label=obj['id'].replace('object_', '#')
            ax.annotate(f"{label}\nd={surface-zc:.2f} м",(x,y),xytext=(5,9),textcoords='offset points',
                        fontsize=7,color='white',bbox=dict(fc='#25313c',alpha=.8,ec='none',pad=1.5))
    ax.axhline(1.03,color='white',lw=.8,ls=':',alpha=.9)
    ax.set(title=f'{title}\nГоризонтальный срез: глубина {surface-z:.2f} м',xlabel='x, м',ylabel='y, м')
    return im


def plot_xz(ax, item, title):
    meta, voxels, eps, objects = item
    dx=meta['dx']; surface=meta['surface_z']; domain=meta['domain']
    j=int(np.floor(1.03/dx)); y=(j+.5)*dx
    xedges=np.arange(voxels.shape[0]+1)*dx
    depths=surface-np.arange(voxels.shape[2]+1)*dx
    im=ax.pcolormesh(xedges, depths, eps[voxels[:,j,:]].T, cmap=CMAP,vmin=VMIN,vmax=VMAX,shading='flat')
    for layer in meta['layers']:
        depth=surface-layer['z_top_m']
        ax.axhline(depth,color='white',ls='--',lw=.9)
    for obj in objects:
        x,oy,z=obj['center_m']; r=obj['radius_m']
        if abs(oy-y)<r:
            ax.annotate(obj['id'].replace('object_','#'),(x,surface-z),xytext=(8,14),textcoords='offset points',
                        color='white',fontsize=8,arrowprops=dict(arrowstyle='-',color='white'),
                        bbox=dict(fc='#25313c',alpha=.8,ec='none',pad=1))
    ax.set(xlim=(0,domain[0]),ylim=(surface,-(domain[2]-surface)),xlabel='x, м',ylabel='Глубина, м',
           title=f'{title} — вертикальный срез y={y:.2f} м')
    ax.set_aspect('equal')
    ax.text(.08,-.16,'ВОЗДУХ',color='white',fontsize=8)
    return im


def main():
    data={scene:read_scene(scene) for scene,_ in SCENES}
    note='Единая шкала Re(εr) при 400 МГц. Чёрный — идеальный металл. Данные: сохранённые воксели, шаг 2 см.'
    fig,axs=plt.subplots(1,3,figsize=(16,5.2),layout='constrained')
    for ax,(scene,label) in zip(axs,SCENES):im=plot_xy(ax,data[scene],label)
    fig.colorbar(im,ax=axs,label='Re(εr) при 400 МГц',shrink=.8)
    fig.suptitle('Три сцены в одинаковом масштабе\n'+note+'\nПунктирные окружности — проекции всех целей; белая точечная линия — положение вертикального среза.',fontsize=10)
    fig.savefig(DEST/'three_scenes_horizontal.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(13,8.2),layout='constrained')
    for ax,(scene,label) in zip(axs,SCENES):im=plot_xz(ax,data[scene],label)
    fig.colorbar(im,ax=axs,label='Re(εr) при 400 МГц',shrink=.85)
    fig.suptitle('Вертикальные разрезы: одинаковые координаты и физический масштаб\n'+note+'\nПоказаны только цели, которые пересекает эта плоскость; отсутствие кружка не означает отсутствие цели в сцене.',fontsize=10)
    fig.savefig(DEST/'three_scenes_vertical.png',dpi=170);plt.close(fig)
    inventory=[]
    for scene,label in SCENES:
        item=data[scene]
        fig=plt.figure(figsize=(12,10),layout='constrained')
        gs=fig.add_gridspec(2,1,height_ratios=[3.8,1.5])
        a=fig.add_subplot(gs[0]); b=fig.add_subplot(gs[1])
        im=plot_xy(a,item,label);plot_xz(b,item,label)
        fig.colorbar(im,ax=[a,b],label='Re(εr) при 400 МГц',shrink=.8)
        fig.suptitle(note+'\nПунктирные окружности сверху — проекции; подписи d — глубины центров целей.',fontsize=10)
        fig.savefig(DEST/(scene+'_sections.png'),dpi=160);plt.close(fig)
        meta,voxels,eps,objects=item
        inventory.append(dict(scene=scene,objects=objects,domain_m=meta['domain'],surface_z_m=meta['surface_z'],
                              nominal_layer_boundaries_depth_m=[meta['surface_z']-v['z_bottom_m'] for v in meta['layers']],
                              finite_epsilon_min=float(np.nanmin(eps[voxels])),finite_epsilon_max=float(np.nanmax(eps[voxels]))))
    (DEST/'section_metadata.json').write_text(json.dumps(inventory,indent=2,ensure_ascii=False))
    (DEST/'HOW-TO-READ.txt').write_text('''РАЗРЕЗЫ СОХРАНЁННЫХ СЦЕН

Это карты материалов, не B-сканы. Все картинки построены напрямую из global.h5, то есть геометрии, которую использовали расчёты. Цвета показывают вещественную часть относительной диэлектрической проницаемости при 400 МГц. Одна шкала применяется ко всем трём сценам; номера материалов между сценами для сравнения не подходят. Идеальный металл показан чёрным: присваивать ему обычную скалярную εr некорректно.

Горизонтальный разрез проходит на глубине 45 см. Пунктирные окружности — проекции всех объектов на план, даже когда их центр выше или ниже разреза; d — глубина центра. Заполненная карта показывает только настоящие ячейки выбранного разреза. Вертикальный разрез проходит при y=1.03 м, в нём показаны только реально пересечённые объекты. Глубина 0 — поверхность, положительная глубина направлена вниз; отрицательная — воздух. В файлах координата z, наоборот, растёт вверх, поверхность z=0.9 м.

Парк: минеральный верх 20 см и более плотный нижний слой. Поле: 26 см верхнего слоя, ниже более плотный грунт; целей нет — контроль. Дорога: 8 см асфальта, 20 см основания, ниже грунт. Парк и дорога содержат соответственно 5 и 3 сферы. Центры всех целей задавались на глубинах 40–52 см. Все границы слоёв в этих сохранённых сценах плоские.

Это упрощённые учебные пресеты. Такая компоновка внутренне согласована, но не воспроизводит измеренный реальный участок: нет корней, кабелей/труб, естественного разнообразия форм и размеров, неровных границ или подтверждённой геологии. Расположение сфер специально ограничено полосами стандартных маршрутов. Второй генератор и новые координаты создают новые сцены; существующие 4500 A-сканов остаются результатами старых сцен.
''')
    print('Saved',DEST,flush=True)

if __name__=='__main__':main()
