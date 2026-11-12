"""Readable, explicitly comparable review plots; stored FDTD data stay unchanged."""
from pathlib import Path
import json
import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.report import soil_statistics
DEST=ROOT/'user_review/realism'
MODES=[('native_peplinski','Peplinski'),('custom_crim_archie','Модель полей')]


def sections():
    cfg=json.loads((DEST/'park-pilot.json').read_text())
    edges=np.r_[0,np.cumsum([l['thickness'] for l in cfg['preset']['layers']])]
    colors=['#c5dfbc','#e4cf91','#bcb19c','#97afb5']
    catmap=ListedColormap(colors);catmap.set_bad('#292929')
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#292929')
    fig,axs=plt.subplots(2,2,figsize=(13,6.2),layout='constrained')
    profiles=[];records=[]
    for col,(mode,title) in enumerate(MODES):
        folder=DEST/'pilot'/mode/'scene'
        meta,ids,epsilon,stats=soil_statistics(folder,600e6)
        dx=meta['dx'];depth=.8-(np.arange(ids.shape[2])+.5)*dx
        db=json.loads((folder/meta['materials']).read_text())['materials']
        with h5py.File(folder/meta['geometry']) as h:keys=[s.decode() for s in h['material_keys'][:]]
        e=epsilon[ids].astype(float)
        for i,k in enumerate(keys):
            if db[k]['name'] in ('pec','free_space'):e[ids==i]=np.nan
        e[:,:,depth<0]=np.nan
        if mode=='custom_crim_archie':
            with np.load(folder/'custom_fields.npz',allow_pickle=False) as h:layers=h['layer_id'].astype(float)
        else:layers=np.broadcast_to(np.searchsorted(edges[1:],depth,side='right')+1,ids.shape).astype(float).copy()
        layers[~np.isfinite(e)]=np.nan
        y=int(.6/dx);sl=e[:,y,:].T;ls=layers[:,y,:].T
        im=axs[0,col].imshow(ls,origin='lower',extent=[0,2.4,.8,-.4],vmin=.5,vmax=4.5,cmap=catmap,aspect='equal',interpolation='nearest')
        im=axs[1,col].imshow(sl,origin='lower',extent=[0,2.4,.8,-.4],vmin=5,vmax=17,cmap=cmap,aspect='equal',interpolation='nearest')
        x=(np.arange(ids.shape[0])+.5)*dx
        # Boundary lines follow actual labels, rather than inventing visible dielectric jumps.
        for k in (1,2,3):
            boundary=[]
            for row in layers[:,y,:]:
                idx=np.flatnonzero(row==k+1)
                boundary.append(float(depth[idx].min()-dx/2) if len(idx) else np.nan)
            for ax in axs[:,col]:ax.plot(x,boundary,color='white',lw=.9,ls='--')
        for k in range(4):
            axs[0,col].text(.08,(edges[k]+edges[k+1])/2,str(k+1),weight='bold',va='center',fontsize=11)
        axs[0,col].set_title(title+' — расположение четырёх слоёв')
        axs[1,col].set_title(title+' — проницаемость ε′ при 600 МГц')
        for ax in axs[:,col]:ax.set(xlim=(0,2.4),ylim=(.8,0),xlabel='Расстояние X, м',ylabel='Глубина, м')
        mask=depth>=0
        values=e[:,:,mask]
        profiles.append((title,depth[mask],np.nanmean(values,axis=(0,1)),np.nanstd(values,axis=(0,1))))
        records.append(dict(mode=mode,layers=stats,section_y_m=(y+.5)*dx))
    fig.colorbar(im,ax=axs[1,:],label='ε′ при 600 МГц',shrink=.85)
    fig.suptitle('Слои есть. Их номера и электрические свойства — разные вещи',fontsize=15)
    fig.supxlabel('Верх: цвет — номер слоя. Низ: одна шкала проницаемости. Чёрный — металлическая цель.\nПунктир показывает границы слоёв, а не силу отражения. Воздух исключён из вида; свойства грунта не изменены.',fontsize=9)
    for name in ('soil_models.png','soil_three_way.png'):fig.savefig(DEST/name,dpi=165)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for (title,d,mean,sd),color in zip(profiles,['#285f8b','#9e5a20']):
        ax.plot(mean,d,label=title,color=color,lw=2)
        ax.fill_betweenx(d,mean-sd,mean+sd,color=color,alpha=.16)
    for v in edges[1:-1]:ax.axhline(v,ls='--',lw=.8,color='#999')
    ax.set(ylim=(.8,0),xlabel='Проницаемость ε′ при 600 МГц',ylabel='Глубина, м',title='Как свойства реально меняются с глубиной')
    ax.grid(alpha=.15);ax.legend()
    fig.supxlabel('Линия — среднее по горизонтальному срезу; полоса — ±1 стандартное отклонение неоднородностей.\nЭто разброс материала, а не ошибка измерения. Воздух и цель исключены.',fontsize=10)
    fig.savefig(DEST/'soil_depth_profile.png',dpi=165);plt.close(fig)
    (DEST/'soil_comparison_metrics.json').write_text(json.dumps(records,indent=2)+'\n')


def remove_dc(a):return a-a.mean(axis=1,keepdims=True)
def remove_background(a):return a-a.mean(axis=0,keepdims=True)
def p99(a):return float(np.percentile(abs(a),99)) or 1


def draw(ax,a,x,t,title,scale=None):
    scale=p99(a) if scale is None else scale
    dx=float(np.median(np.diff(x)));dt=float(np.median(np.diff(t)))
    im=ax.imshow((a/scale).T,cmap='gray',vmin=-1,vmax=1,interpolation='nearest',aspect='auto',
                 extent=[x[0]-dx/2,x[-1]+dx/2,(t[-1]+dt/2)*1e9,(t[0]-dt/2)*1e9])
    ax.set(xlim=(0,1.4),ylim=(16,0),xlabel='Путь, м',ylabel='Время, нс',title=title)
    return im


def bscans():
    with np.load(DEST/'standardized_comparison.npz',allow_pickle=False) as h:
        x=h['x_m'];t=h['time_s']
        arrays=[h[k].astype(float) for k in ('native_raw','fields_raw','real_raw')]
    names=['Peplinski · грунт','Модель полей · грунт','GP8800 · бетон, образец II']
    fig,axs=plt.subplots(2,3,figsize=(15,8.6),layout='constrained')
    for col,(name,a) in enumerate(zip(names,arrays)):
        a=remove_dc(a);b=remove_background(a)
        draw(axs[0,col],a,x,t,name+'\nДо удаления общей трассы')
        im=draw(axs[1,col],b,x,t,name+'\nПосле удаления общей трассы')
    fig.colorbar(im,ax=axs,label='Нормированная амплитуда: чёрный −, серый 0, белый +',shrink=.7)
    fig.suptitle('Один формат: 1,4 м × 16 нс; сетка 5 см × 50 пс',fontsize=15)
    fig.supxlabel('Везде одинаковая обработка. Контраст каждой панели настроен отдельно по P99. Частотная полоса и начало времени приборов не выравнивались.\nРеальный профиль сглажен перед уменьшением числа отсчётов. Это единый формат просмотра, а не одинаковые условия опыта.',fontsize=10)
    fig.savefig(DEST/'bscan_three_way.png',dpi=165);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,a,title in zip(axs,arrays[:2],names[:2]):
        draw(ax,remove_background(remove_dc(a)),x,t,title)
    fig.suptitle('Отклик одной сферы в двух моделях грунта',fontsize=15)
    fig.supxlabel('Шаг 5 см; одинаковые источник, маршрут и цель. Шкала яркости каждой панели настроена отдельно.',fontsize=10)
    fig.savefig(DEST/'synthetic_hyperbolas.png',dpi=165);plt.close(fig)
    a=remove_dc(arrays[1]);b=remove_background(a);gain=p99(a)/p99(b)
    fig,axs=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    titles=['1. Исходный фон доминирует','2. Фон вычтен; контраст прежний',f'3. Фон вычтен; контраст ×{gain:.1f}']
    for ax,v,title,scale in zip(axs,[a,b,b],titles,[p99(a),p99(a),p99(b)]):draw(ax,v,x,t,title,scale)
    fig.suptitle('Откуда берётся «классная контрастность» — на одних и тех же данных',fontsize=15)
    fig.supxlabel('Вычитание общей трассы и усиление контраста — два отдельных действия. Полоса и время здесь не меняются.\nОбщий сигнал = среднее по всем позициям отдельно для каждого момента времени. Вместе с фоном могут исчезнуть отражения от плоских слоёв.',fontsize=10)
    fig.savefig(DEST/'processing_effect.png',dpi=165);plt.close(fig)
    # Preserve dense measured export so coarse-grid unification cannot hide its resolution.
    with np.load(DEST/'reference_radar/concrete_specimen_II.npz',allow_pickle=False) as h:
        real=h['amplitude'];rt=h['time_s'];rx=h['x_m']
    keep=rx<=1.4+1e-9;real=real[keep];rx=rx[keep]
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    draw(axs[0],remove_background(remove_dc(real)),rx,rt,'Реальное измерение: исходный шаг 5 мм')
    draw(axs[1],remove_background(remove_dc(arrays[2])),x,t,'То же измерение: общий шаг 5 см')
    fig.suptitle('Почему синтетика выглядит крупными пикселями',fontsize=15)
    fig.supxlabel('В симуляции соседние позиции расположены в 10 раз реже. Для общей сетки реальное измерение приходится сглаживать.\nУвеличение числа пикселей картинки не создаёт новых рассчитанных трасс.',fontsize=10)
    fig.savefig(DEST/'sampling_comparison.png',dpi=165);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,(mode,title) in zip(axs,MODES):
        with np.load(DEST/'pilot'/mode/'effective_sfcw.npz',allow_pickle=False) as h:
            mask=h['time_s']<16e-9;sa=h['response'][:,mask];st=h['time_s'][mask];sx=h['x_m']-h['x_m'][0]
        draw(ax,remove_background(remove_dc(sa)),sx,st,title+' · эффективные 400–900 МГц')
    fig.suptitle('Дополнительный вид: отклик модели в выбранной полосе',fontsize=15)
    fig.supxlabel('Здесь удалён спектр введённого импульса, выделена полоса и применено окно Hann. Это другая обработка.\nЭтот рисунок не используется как одинаково обработанная пара с приборным экспортом.',fontsize=10)
    fig.savefig(DEST/'effective_sweep_gray.png',dpi=165);plt.close(fig)
