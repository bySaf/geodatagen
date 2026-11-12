"""Read-only review of an existing raster acquisition; no simulation is run."""
from pathlib import Path
import json
import html
import hashlib
import re
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'data/final/park'
OUT = ROOT/'user_review/route'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.grid': False})
config = json.loads((ROOT/'data/final/config.json').read_text())
objects = json.loads((SRC/'objects.json').read_text())
with h5py.File(SRC/'bscan_0.h5') as f:
    a = f['Ey'][:].astype(float)
    t = f['time_s'][:]*1e9
    tx, rx = f['tx_m'][:], f['rx_m'][:]
    complete = f['complete'][:]
    dt = float(f.attrs['dt'])
assert a.shape == (500, 832) and complete.all() and np.isfinite(a).all()
coords = np.load(SRC/'path_0.npz')
assert np.array_equal(tx, coords['tx']) and np.array_equal(rx, coords['rx'])
steps = np.linalg.norm(np.diff(tx, axis=0), axis=1)
assert np.allclose(steps, .1)
rows = [np.arange(i*50, (i+1)*50) for i in range(10)]
for ids in rows:
    assert np.allclose(tx[ids, 1], tx[ids[0], 1])
    assert np.allclose(abs(np.diff(tx[ids, 0])), .1)
ids = rows[2]  # Eastbound third line, y=1.0 m, near the metal sphere.
selected = [120, 135, 139, 143]
colors = ['#5851aa', '#c17900', '#dc362e', '#008583']
late = t >= 6
residual = a[ids]-a[ids].mean(axis=0)
row_residuals = [a[r]-a[r].mean(axis=0) for r in rows]
scale = lambda data: max(float(np.percentile(abs(data), 99)), np.finfo(float).eps)

db = json.loads((SRC/'global_materials.json').read_text())['materials']
with h5py.File(SRC/'global.h5') as f:
    vox = f['data'][:]
    keys = [s.decode() if isinstance(s, bytes) else str(s) for s in f['material_keys'][:]]
er = []
for k in keys:
    m = db[k]
    eps = float(m['base']['relative_permittivity'])
    for pole in m.get('poles', []):
        eps += pole['relative_permittivity_difference']/(1+(2*np.pi*400e6*pole['relaxation_time_s'])**2)
    er.append(eps)
er = np.asarray(er)
soil_material = np.array([db[k]['model']=='debye' for k in keys])
layer_eps = []
for lo, hi in [(35,45),(0,35)]:
    cells = vox[:,:,lo:hi]
    values = er[cells][soil_material[cells]]
    layer_eps.append(dict(mean=float(values.mean()), min=float(values.min()), max=float(values.max()),
                          velocity_from_mean_m_per_ns=float(.299792458/np.sqrt(values.mean()))))

def save(fig, name):
    fig.savefig(OUT/name, dpi=170, bbox_inches='tight')
    plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios':[1, 1.15]})
for ax in axes:
    ax.add_patch(Rectangle((0,0),7,4.8,facecolor='#f3f5f7',edgecolor='#a4abb5'))
    for n, rr in enumerate(rows):
        ax.plot(tx[rr,0],tx[rr,1],color='#8b9bad',lw=1)
        i,j=rr[22],rr[27]
        ax.annotate('',xy=tx[j,:2],xytext=tx[i,:2],arrowprops=dict(arrowstyle='->',color='#61768b',lw=1.6))
    ax.plot(tx[:,0],tx[:,1],color='#8b9bad',lw=.7,zorder=2)
    ax.plot(tx[ids,0],tx[ids,1],color='#174fb0',lw=2.5,label='Разбираемый проход: y = 1.00 м')
    for o in objects:
        c={'metal_sphere':'#cf2e25','air_sphere':'#13a6a6','dielectric_sphere':'#d29900'}[o['type']]
        ax.add_patch(Circle(o['center_m'][:2],o['radius_m'],facecolor=c,edgecolor='white',lw=.8,zorder=5,alpha=.85))
        ax.annotate(o['id'].replace('object_','#'),o['center_m'][:2],xytext=(7,7),textcoords='offset points',fontsize=9,color=c,zorder=8)
    ax.set(xlabel='x, м',ylabel='y, м',aspect='equal')
axes[0].set(xlim=(0,7),ylim=(0,4.8),title='Вся сцена и маршрут overlap_a')
axes[0].scatter(*tx[0,:2],marker='>',color='#174fb0',s=60,zorder=6)
axes[0].annotate('Старт: ID 0',tx[0,:2],xytext=(-15,-24),textcoords='offset points',fontsize=9)
axes[0].scatter(*tx[-1,:2],marker='s',color='#174fb0',s=35,zorder=6)
axes[0].annotate('Финиш: ID 499',tx[-1,:2],xytext=(-15,13),textcoords='offset points',fontsize=9)
axes[0].legend(loc='upper right',fontsize=9)
for i,c in zip(selected, colors):
    axes[1].scatter(*tx[i,:2],color=c,s=45,edgecolors='white',zorder=7)
    axes[1].annotate(f'ID {i}',tx[i,:2],xytext=(0,-26),textcoords='offset points',ha='center',color=c,fontweight='bold',fontsize=9)
axes[1].set(xlim=(.6,6),ylim=(.65,1.9),title='Нижняя полоса: 10 прямых проходов по 50 точек')
axes[1].text(.02,-.33,'Кружки — проекция центров и радиусов объектов на землю.\nКрасный: металл; бирюзовый: воздух; жёлтый: диэлектрик.\nПриёмник всегда на 4 см правее Tx по x, даже на обратном ходе.',transform=axes[1].transAxes,fontsize=10,va='top')
fig.suptitle('500 A-сканов = змейка 49.9 м внутри полосы 4.9 × 0.9 м',fontsize=16,y=1.02)
fig.tight_layout()
save(fig,'01_route_map.png')

fig, axes = plt.subplots(3,1,figsize=(13,10))
for i,c in zip(selected,colors):
    label=f'ID {i}: Tx x={tx[i,0]:.2f} м'
    axes[0].plot(t,a[i],color=c,lw=1.1,label=label)
    axes[1].plot(t[late],a[i,late],color=c,lw=1.2)
    axes[2].plot(t[late],(a[i]-a[ids].mean(0))[late],color=c,lw=1.2)
for ax in axes:
    ax.set(xlabel='Время с начала расчёта, нс',ylabel='Ey, В/м')
    ax.grid(alpha=.18)
axes[0].set(xlim=(0,16),title='Полные A-сканы: ранний общий сигнал подавляет поздние различия')
axes[0].legend(ncol=2,fontsize=9)
axes[1].set(xlim=(6,16),title='Те же сырые сигналы после 6 нс: только увеличен масштаб')
axes[2].set(xlim=(6,16),title='После вычитания средней трассы одного прохода: это отображение, не новый расчёт')
fig.suptitle('Парк, проход y = 1.00 м: далеко → до цели → над целью → после цели',fontsize=15)
fig.tight_layout(rect=(0,0,1,.97))
save(fig,'02_ascans.png')

fig,axes=plt.subplots(2,2,figsize=(14,9))
for col,(data,label) in enumerate([(a[ids],'Сырые Ey'),(residual,'Ey минус средняя трасса прохода')]):
    for row, mask in enumerate([np.ones(len(t),dtype=bool),late]):
        vals=data[:,mask]
        lim=scale(vals)
        im=axes[row,col].imshow(vals.T,aspect='auto',interpolation='nearest',cmap='gray',vmin=-lim,vmax=lim,
                    extent=(.75,5.75,t[mask][-1]+dt*1e9/2,t[mask][0]-dt*1e9/2))
        axes[row,col].set(xlabel='Положение Tx по x, м',ylabel='Время, нс',title=label+(' · 0–16 нс' if row==0 else ' · 6–16 нс'))
        for o in objects[:2]:
            x=o['center_m'][0]
            axes[row,col].axvline(x,color='#225d19',lw=.8,ls='--',alpha=.85)
        fig.colorbar(im,ax=axes[row,col],label='В/м · предел ±99-й процентиль',shrink=.85)
fig.suptitle('Корректный B-скан одного прямого прохода: 50 трасс, ID 100–149',fontsize=16)
fig.text(.5,.005,'Пунктиры — x центров металла #000 и воздушной полости #001. Это координаты целей, не доказанное выделение их отклика.\nВсе четыре шкалы амплитуды разные; предел подписан справа. Нет усиления по времени и нормировки отдельных трасс.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.055,1,.965))
save(fig,'03_straight_bscan.png')

fig,axes=plt.subplots(2,1,figsize=(15,8))
for ax,mask,label in [(axes[0],np.ones(len(t),dtype=bool),'Все 500 трасс: сырые данные 0–16 нс'),(axes[1],late,'Те же 500 трасс: сырые данные после 6 нс')]:
    vals=a[:,mask];lim=scale(vals)
    im=ax.imshow(vals.T,aspect='auto',interpolation='nearest',cmap='gray',vmin=-lim,vmax=lim,
                 extent=(-.5,499.5,t[mask][-1],t[mask][0]))
    for n in range(1,10): ax.axvline(n*50-.5,color='black',ls='--',lw=.8,alpha=.65)
    ax.axvspan(99.5,149.5,facecolor='none',edgecolor='#38a036',lw=2)
    ax.set(xlabel='ID трассы в порядке движения (не x и не один прямой профиль)',ylabel='Время, нс',title=label)
    fig.colorbar(im,ax=ax,label='Ey, В/м · ±99-й процентиль')
fig.suptitle('Почему склейка змейки вводит в заблуждение: каждые 50 трасс — смена направления',fontsize=15)
fig.tight_layout(rect=(0,0,1,.96))
save(fig,'04_whole_route_bscan.png')

fig,axes=plt.subplots(5,2,figsize=(14,14),sharex=True,sharey=True)
lim=scale(np.concatenate([r[:,late] for r in row_residuals]))
for n,ax in enumerate(axes.flat):
    rr=rows[n];order=np.argsort(tx[rr,0]);vals=row_residuals[n][order][:,late]
    im=ax.imshow(vals.T,aspect='auto',interpolation='nearest',cmap='gray',vmin=-lim,vmax=lim,extent=(.75,5.75,t[-1],t[late][0]))
    ax.set(title=f'Проход {n+1}: y={tx[rr[0],1]:.1f} м · ID {rr[0]}–{rr[-1]}'+(' ←' if n%2 else ' →'),ylabel='Время, нс',xlabel='x, м')
fig.suptitle('10 отдельных проходов: одинаковые оси и одна шкала амплитуд',fontsize=16)
fig.text(.5,.01,'Вычтена средняя трасса КАЖДОГО прохода; показано t ≥ 6 нс. Обратные проходы развёрнуты по x для сравнения.\nЭто не чистый сигнал объектов: здесь остаются неоднородность грунта, цели и возможные численные артефакты.',ha='center',fontsize=10)
fig.subplots_adjust(left=.07,right=.9,top=.95,bottom=.065,hspace=.45)
cax=fig.add_axes([.92,.18,.016,.66]);fig.colorbar(im,cax=cax,label='Остаток Ey, В/м · общий ±99-й процентиль')
save(fig,'05_ten_straight_passes.png')

v=layer_eps[1]['velocity_from_mean_m_per_ns']
metal=objects[0];depth=metal['depth_center_m'];radius=metal['radius_m']
dx=np.linspace(-1.2,1.2,500)
delta_t=2*(np.sqrt(depth**2+dx**2)-depth)/v
fig,ax=plt.subplots(figsize=(11,5))
ax.plot(dx,delta_t,color='#174fb0',lw=2,label='Иллюстративная гипербола в однородной среде')
ax.axvspan(-.4+radius,.4-radius,color='#4faa65',alpha=.18,label='Сфера целиком до бокового PML')
ax.axvspan(-.6-radius,-.4+radius,color='#dca54b',alpha=.12)
ax.axvspan(.4-radius,.6+radius,color='#dca54b',alpha=.12,label='Частичное попадание в PML / обрезка')
for edge in [-.4,.4]:ax.axvline(edge,color='#cc851e',ls='--',lw=1)
for edge in [-.6,.6]:ax.axvline(edge,color='#c24537',ls=':',lw=1.5)
ax.invert_yaxis();ax.grid(alpha=.2)
ax.set(xlabel='Смещение антенны от центра цели по x, м',ylabel='Дополнительное время относительно вершины, нс',
       title=f'Окно расчёта оставляет лишь центральную часть кривой: глубина {depth:.2f} м, v ≈ {v:.3f} м/нс')
ax.legend(fontsize=9,loc='lower left')
fig.text(.5,.015,'Схема, НЕ подгонка к данным: без слоёв, воздушного зазора, преломления, размера цели и разнесения Tx/Rx.\nФормула Δt = 2·(√(d²+Δx²) − d)/v. Границы относятся к движущемуся окну 1.2 м и PML 0.2 м.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.10,1,1))
save(fig,'06_window_and_hyperbola.png')

mean=a.mean(0)
metrics={
 'scene':'park','route':'overlap_a / path_0','trace_count':500,'complete_count':int(complete.sum()),
 'samples_per_trace':832,'dt_ns':dt*1e9,'last_sample_ns':float(t[-1]),
 'route_length_m':float(steps.sum()),'step_min_m':float(steps.min()),'step_max_m':float(steps.max()),
 'raster_rows':10,'traces_per_row':50,'straight_leg_span_m':4.9,'row_pitch_m':.1,
 'selected_row':{'zero_based_index':2,'ids':[100,149],'y_m':1.,'tx_x_range_m':[.8,5.7]},
 'selected_ascans':[{'id':i,'tx_m':tx[i].tolist(),'rx_m':rx[i].tolist()} for i in selected],
 'peak_abs_raw_V_per_m':float(abs(a).max()),
 'energy_after_6ns_percent':float(100*np.sum(a[:,late]**2)/np.sum(a*a)),
 'energy_after_mean_trace_subtraction_percent':float(100*np.sum((a-mean)**2)/np.sum(a*a)),
 'mean_trace_peak_time_ns':float(t[np.argmax(abs(mean))]),
 'late_p99_raw_V_per_m':scale(a[:,late]),'selected_row_late_p99_residual_V_per_m':scale(residual[:,late]),
 'soil_permittivity_at_400MHz':layer_eps,
 'wave_speed_note':'c/sqrt(mean real permittivity), illustrative low-loss local estimate, not fitted group velocity',
 'source_frequency_MHz':400,'nominal_period_ns':2.5,'tx_rx_offset_m':.04,'antenna_height_m':.06,
 'window_xy_m':[1.2,1.2],'pml_thickness_m':.2,'clear_side_halfwidth_m':.4,
 'geometry_dx_m':.02,'field_solver_dx_m':.01,
 'object0_full_sphere_before_x_pml_halfspan_m':.4-radius,
 'object0_illustrative_delta_t_at_04m_ns':float(2*(np.sqrt(depth**2+.4**2)-depth)/v),
 'source_h5_sha256':hashlib.sha256((SRC/'bscan_0.h5').read_bytes()).hexdigest(),
 'processing':'All figures read existing Ey; no FDTD. Residuals subtract mean within each straight leg. Per-image global p99 clipping only, no time gain or per-trace normalisation.',
 'interpretation':'Integrity and geometry inspection, not real-radar calibration or unique identification of target echoes.'}
(OUT/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
np.savetxt(OUT/'selected_ascans.csv',np.c_[t,*[a[i] for i in selected]],delimiter=',',header='time_ns,'+','.join(f'Ey_ID_{i}_V_per_m' for i in selected),comments='')
np.savetxt(OUT/'straight_pass_coordinates.csv',np.c_[ids,tx[ids],rx[ids]],delimiter=',',header='scan_id,tx_x_m,tx_y_m,tx_z_m,rx_x_m,rx_y_m,rx_z_m',comments='')

sources=[('Документация gprMax: B-скан металлического цилиндра','https://docs.gprmax.com/en/latest/examples_simple_2D.html'),
 ('Документация gprMax: готовые антенные модели','https://docs.gprmax.com/en/latest/inc_GPRAntennaModels.html'),
 ('Sensors & Software: физика гипербол и определение глубины','https://www.sensoft.ca/georadar/lmx200-online-manual/'),
 ('USGS: оценка скорости по отражениям и гиперболам','https://pubs.usgs.gov/of/1990/0414/report.pdf')]

report=f'''РАЗБОР МАРШРУТА: ПАРК / OVERLAP_A
Проверено 12 ноября 2026. Используются уже рассчитанные данные; новых FDTD запусков нет.

1. ЧТО ИМЕННО МЫ РАЗБИРАЕМ
Источник: data/final/park/bscan_0.h5. Все 500 трасс завершены, значения конечны,
координаты HDF5 совпадают с path_0.npz. В каждой трассе 832 отсчёта; шаг времени
{dt*1e12:.3f} пс; последний отсчёт {t[-1]:.4f} нс. Регистрируется компонента Ey в В/м.
Это поле в идеальной точке, а не напряжение реального приёмного тракта.

Маршрут: 10 прямых проходов по 50 точек. x=0.8…5.7 м, y=0.8…1.7 м;
между точками и строками 10 см. Суммарный путь 49.9 м = 10×4.9 + 9×0.1.
Это не один прямой 50-метровый профиль. Слева направо, затем справа налево,
и так змейкой. В точке разворота ориентация диполя и разнос Rx не поворачиваются:
диполь направлен по y, Rx всегда смещён на +4 см по глобальной оси x.
Для физического прибора нужно отдельно определить, разворачивается ли он целиком.

2. КАРТА И ВЫБРАННЫЕ СИГНАЛЫ
01_route_map.png: точный путь и проекции всех пяти объектов на землю.
Выбран третий прямой проход y=1.00 м, ID100–149. В нём 50 трасс за 4.9 м.
Металлическая сфера #000: центр ({metal['center_m'][0]:.3f}, {metal['center_m'][1]:.3f}, {metal['center_m'][2]:.3f}) м,
радиус {radius*100:.2f} см, глубина центра {depth*100:.2f} см. Воздушная #001 расположена
рядом сбоку, поэтому отражения нельзя без проверки приписать одному металлу.
На карте кружки означают проекции объёмных сфер, а не их сечения в одной плоскости.

02_ascans.png: ID120 (x=2.8), 135 (4.3), 139 (4.7), 143 (5.1 м).
Верх — полные сырые сигналы; середина — те же сигналы после6нс с другой шкалой;
низ — поздняя часть после вычитания средней трассы выбранного прохода.
CSV содержит исходные значения этих четырёх A-сканов. Никакой нормировки по трассам нет.

3. ЧТО ВИДНО В ДАННЫХ
Максимум |Ey| по маршруту = {abs(a).max():.3f} В/м.
После 6 нс остаётся {metrics['energy_after_6ns_percent']:.8f}% суммы квадратов отсчётов.
После вычитания средней трассы всего маршрута остаётся
{metrics['energy_after_mean_trace_subtraction_percent']:.8f}% этой же энергии.
Это доли численной энергии записи, НЕ SNR, НЕ энергия объектов и НЕ процент сходства с реальностью.
Доминирует общий ранний сигнал. На полных A-сканах четыре кривые почти совпадают.
Увеличение позднего участка показывает различия; вычитание среднего подавляет общий фон,
но вместе с ним убирает стационарные отражения от слоёв. Остаток не является чистым сигналом цели.
Близкие идеальные Tx/Rx без экрана — правдоподобная причина большой прямой связи;
её вклад ещё не отделён контролируемым сравнением разных антенн.

4. КАК ПРАВИЛЬНО ЧИТАТЬ B-СКАНЫ
03_straight_bscan.png — один физически прямой профиль. Горизонтальная ось: x в метрах;
вертикальная: время в нс с начала расчёта. Это НЕ глубина. Верхняя строка0–16нс,
нижняя6–16нс; слева сырой сигнал, справа удалена средняя трасса этого прохода.
У каждого изображения своя симметричная шкала ±99-й процентиль; это явно подписано.
Пунктиры отмечают x известных объектов и не доказывают, что видимый пик создан именно ими.
04_whole_route_bscan.png — все500трасс в порядке движения, с границами проходов.
05_ten_straight_passes.png — десять отдельных B-сканов с едиными осями и шкалой.
На обратных проходах столбцы переставлены по возрастающему x только для отображения.
Так можно сопоставлять одни и те же объекты между соседними проходами.

5. ПОЧЕМУ ГИПЕРБОЛЫ МОГУТ ВЫГЛЯДЕТЬ ВЫТЯНУТЫМИ
А) Масштаб рисунка и склейка змейки. Растяжение окна картинки меняет вид дуги без
изменения физики. Сборка500трасс в одну полоску смешивает10проходов и9разворотов.
Б) Время зависит от длины пути Tx→цель→Rx и скорости в грунте. Для совпадающих Tx/Rx,
точечной цели в однородном грунте: t(x)=2·sqrt(d²+(x−x0)²)/v. Глубина, скорость,
боковое смещение относительно цели и соотношение масштабов осей меняют кривизну.
Антенна меняет видимость и форму импульса, но не отменяет эту геометрию.
В) В расчёте вырезается окно всего1.2×1.2м, а0.2м с каждого края занимает PML.
Tx находится около центра; до начала бокового PML около0.4м, до края0.6м.
Для сферы радиусом{radius:.3f}м полный объект без попадания в PML доступен только при
|Δx|≲{.4-radius:.3f}м (при подходящем y). Дальние хвосты физической гиперболы
не воспроизводятся как в большой сцене. Могут оставаться пологие вершины и возникать
артефакты при входе/выходе объекта. Это ограничение постановки, а не доказательство,
что каждая видимая дуга испорчена именно им.
Г) Источник Ricker400МГц: характерный период2.5нс, не точная длительность импульса.
Средние Re(εr) в двух слоях: {layer_eps[0]['mean']:.2f} и {layer_eps[1]['mean']:.2f};
оценки c/sqrt(εr): {layer_eps[0]['velocity_from_mean_m_per_ns']:.3f} и {v:.3f}м/нс.
В нижнем слое длина волны около{v/.4:.2f}м. Шаг трасс10см редкий для аккуратного
изображения небольшой цели, а импульс размазывает отражение по времени.
Д) Реальные радарограммы часто уже имеют удаление фона, усиление, фильтры и
калибровку нулевого времени. Сырые Ey нельзя честно сравнивать с такой картинкой
по форме и контрасту без согласования обработки.
06_window_and_hyperbola.png показывает ограничение окна на аналитической схеме.
Это не подгонка к нашим данным: схема не учитывает слои, преломление, воздушный зазор,
размер цели и разнос Tx/Rx. При d={depth:.3f}м смещение±0.4м даёт лишь
{metrics['object0_illustrative_delta_t_at_04m_ns']:.2f}нс дополнительного хода в этой схеме.

6. МОЖНО ЛИ ПОСТАВИТЬ ДРУГИЕ Tx/Rx
Да. Сейчас это идеальный y-диполь и точечный приёмник Ey, разнос4см, высота6см.
Реалистичная геометрия антенны, экран, питание, нагрузка и выходной тракт могут
сильно изменить прямую связь, звон, диаграмму направленности и зарегистрированный A-скан.
В официальной библиотеке есть модель, подобная GSSI400МГц, с заявленной сеткой2мм.
Это лучше согласуется с исходной частотой, чем произвольная замена на1.5ГГц,
но не гарантирует сходства с неизвестным реальным прибором.
Текущий расчёт поля идёт на1см; просто подключить такую модель и оставить1см нельзя.
При одинаковом объёме переход1см→2мм даёт примерно125раз больше ячеек и5раз больше
временных шагов (идеализированная оценка вычислительной работы~625раз, не замер времени).
Геометрия грунта2см при этом тоже требует отдельного осмысленного решения.

Практичный следующий эксперимент: один прямой проход над известной целью,
20–50позиций на небольшом участке, шаг2–5см, одинаковый грунт/seed; сравнить
идеальную пару и выбранную антенну, отдельно проверить более широкое окно.
Для чистого отклика цели нужна парная сцена с ней и без неё. Только после этого
масштабировать на тысячи трасс. Изменять разнос/частоту ради красивой гиперболы
без привязки к прибору — не калибровка реального георадара.

7. ОГРАНИЧЕНИЯ ВЫВОДОВ
Этот разбор подтверждает структуру маршрута и объясняет ограничения просмотра.
Он не подтверждает реализм прибора, не даёт процента сходства с IRL и не заменяет
проверку сходимости/границ/парных object–no-object расчётов. Сферы малые,5целей
идеализированы, грунт имеет плоские слои. Данные и исходные результаты не изменены.

ИСТОЧНИКИ
'''+ '\n'.join(f'{title}: {url}' for title,url in sources)+'\n'
report=re.sub(r'(?<=\d)(?=[А-Яа-яЁё])|(?<=[А-Яа-яЁё])(?=\d)', ' ', report)
report=re.sub(r'\b(ID|GSSI|Ricker)(?=\d)', r'\1 ', report)
(OUT/'REPORT.txt').write_text(report,encoding='utf-8')

obj_rows=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [o['id'],o['type'],', '.join(f'{c:.3f}' for c in o['center_m']),f"{o['radius_m']*100:.2f}",f"{o['depth_center_m']*100:.2f}"] )+'</tr>' for o in objects)
body='''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Разбор маршрута — парк</title>
<style>body{max-width:1150px;margin:40px auto;padding:0 24px;font:17px/1.6 system-ui,sans-serif;color:#203044;background:#f8fafc}h1{font-size:34px;line-height:1.2}h2{margin-top:44px}img{width:100%;background:white;border:1px solid #d9e0e7;border-radius:8px}figure{margin:28px 0}figcaption{font-size:14px;color:#4c6177;margin-top:8px}.note{background:#e8eff9;padding:18px 24px;border-left:5px solid #295daf}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:9px;border-bottom:1px solid #d4dce4}pre{white-space:pre-wrap;font:15px/1.65 system-ui,sans-serif;background:white;padding:24px}a{color:#174fb0}</style>
<h1>Один маршрут: как ехал георадар и что записал</h1><p>Парк · overlap_a · 500 A-сканов · существующий расчёт · проверка 12 ноября 2026</p>
<div class="note"><b>Ключевое:</b> это змейка из 10 прямых проходов, а не один профиль 50 м. Ниже есть корректные B-сканы каждого прохода. Прямая связь сильно доминирует; реалистичность конкретного прибора пока не доказана.</div>'''
captions=[('01_route_map.png','Маршрут сверху','Геометрически точные координаты. Объекты показаны проекциями на землю; стрелки — направление движения.'),('02_ascans.png','Четыре A-скана','Полные сигналы, увеличенный поздний участок и отдельно отображение после удаления среднего фона.'),('03_straight_bscan.png','Один прямой B-скан','Третий проход: y=1.00м, ID100–149. Разные шкалы амплитуды подписаны; вертикальная ось — время, не глубина.'),('04_whole_route_bscan.png','Все500трасс вместе','Пунктиры разделяют10проходов. Зелёная рамка выделяет подробно разобранный проход.'),('05_ten_straight_passes.png','Сравнение10прямых проходов','Общая шкала амплитуды; отдельно удалён средний фон каждого прохода. Обратные проходы развёрнуты поx.'),('06_window_and_hyperbola.png','Что делает маленькое окно расчёта','Схема объясняет, почему длинные хвосты могут отсутствовать. Это аналитическая иллюстрация, не подгонка к измерениям.')]
for filename,title,caption in captions:
    title=re.sub(r'(?<=\d)(?=[А-Яа-яЁё])|(?<=[А-Яа-яЁё])(?=\d)', ' ', title)
    caption=re.sub(r'(?<=\d)(?=[А-Яа-яЁё])|(?<=[А-Яа-яЁё])(?=\d)', ' ', caption)
    caption=re.sub(r'\bID(?=\d)', 'ID ', caption).replace('поx', 'по x')
    body+=f'<h2>{title}</h2><figure><a href="{filename}"><img src="{filename}" alt="{html.escape(title)}"></a><figcaption>{caption}</figcaption></figure>'
body+='<h2>Координаты всех объектов парка</h2><table><tr><th>ID</th><th>Тип</th><th>Центр x,y,z, м</th><th>Радиус, см</th><th>Глубина центра, см</th></tr>'+obj_rows+'</table>'
body+='<h2>Подробные выводы и методика</h2><pre>'+html.escape(report)+'</pre><p><a href="REPORT.txt">Текст отчёта</a> · <a href="metrics.json">Все числа (JSON)</a> · <a href="selected_ascans.csv">Четыре A-скана (CSV)</a> · <a href="straight_pass_coordinates.csv">Координаты прямого прохода (CSV)</a></p>'
body+='<h2>Первичные источники</h2><ul>'+''.join(f'<li><a href="{url}">{title}</a></li>' for title,url in sources)+'</ul></html>'
(OUT/'index.html').write_text(body,encoding='utf-8')
print(json.dumps({'output':str(OUT),'metrics':metrics},ensure_ascii=False,indent=2))
