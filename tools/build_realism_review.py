"""Compare measured references with computed scenes; never invent matched truth."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

DEST = ROOT/'user_review/realism'
MODES = [('native_peplinski','Peplinski + fractal_box'),('custom_crim_archie','Модель полей: CRIM/Archie')]


def route():
    cfg=json.loads((DEST/'park-pilot.json').read_text())
    g,p,s,target=(cfg[k] for k in ('grid','profile','source','target'))
    x=np.arange(p['x_start_m'],p['x_end_m']+p['step_m']/2,p['step_m'])
    cx,cy,cz=target['center_m'];depth=g['surface_z']-cz
    fig,axs=plt.subplots(2,1,figsize=(12,7),layout='constrained')
    ax=axs[0];pad=g['pml_thickness'];lx,ly,lz=g['domain']
    ax.add_patch(Rectangle((0,0),lx,ly,facecolor='#f4e8d3',edgecolor='#555'))
    ax.add_patch(Rectangle((pad,pad),lx-2*pad,ly-2*pad,facecolor='#e7f0e1',edgecolor='#999',ls='--'))
    ax.add_patch(Circle((cx,cy),target['radius_m'],color='#333',label='Проекция сферы'))
    ax.scatter(x,np.full_like(x,p['y_m']),s=28,color='#196cba',label='Tx — 29 точек')
    ax.scatter(x+s['tx_rx_offset_m'],np.full_like(x,p['y_m']),s=52,facecolors='none',edgecolors='#b34d32',label='Rx — сдвиг +10 см',zorder=3)
    ax.annotate('',xy=(1.9,.87),xytext=(.5,.87),arrowprops=dict(arrowstyle='->',color='#196cba',lw=2))
    ax.text(1.2,.94,'Один прямой проход: 1,4 м; шаг 5 см',ha='center')
    ax.set(xlim=(0,lx),ylim=(0,ly),aspect='equal',xlabel='X, м',ylabel='Y, м',title='Вид сверху: обе модели используют одинаковые координаты')
    ax.legend(loc='lower center',ncol=3,fontsize=9)
    ax=axs[1]
    colors=['#c4b394','#afa889','#93997c','#7f896f'];d=0
    for layer,color in zip(cfg['preset']['layers'],colors):
        thick=layer['thickness'];ax.axhspan(d,d+thick,color=color)
        ax.text(.08,d+thick/2,f'{d:.2f}–{d+thick:.2f} м',va='center',fontsize=9);d+=thick
    ax.axhline(0,color='#333');ax.scatter(x,np.full_like(x,-s['height_m']),s=18,color='#196cba')
    ax.add_patch(Circle((cx,depth),target['radius_m'],color='#333'))
    ax.annotate(f'Металлическая сфера Ø{2*target["radius_m"]*100:.0f} см\nцентр на глубине {depth*100:.0f} см',xy=(cx,depth),xytext=(1.55,.16),arrowprops=dict(arrowstyle='->'),fontsize=10)
    ax.set(xlim=(0,lx),ylim=(.8,-.12),aspect='equal',xlabel='X, м',ylabel='Глубина от поверхности, м',title='Схема по глубине: показаны номинальные границы слоёв')
    fig.supxlabel('Расчётная область фиксирована: 2,4 × 1,2 × 1,2 м. Бежевые края сверху — поглощающая область 12 см.\nРеальные воксели и неровности слоёв показаны в отдельном цветном разрезе. Сфера — проверочная цель, не типичный объект парка.',fontsize=9)
    fig.savefig(DEST/'pilot_route.png',dpi=160);plt.close(fig)


from review_v2_figures import sections, bscans


def controls():
    fig,axs=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for ax,(mode,title) in zip(axs,MODES):
        folder=DEST/'pilot'/mode
        qc=json.loads((folder/'center_checks.json').read_text())
        i=qc['center_trace_index']
        with h5py.File(folder/'bscan.h5') as h:
            a=h['Ey'][i].astype(float);t=h['time_s'][:]*1e9
        with h5py.File(folder/'background_center/bscan.h5') as h:b=h['Ey'][0].astype(float)
        with h5py.File(folder/'pml_center/bscan.h5') as h:p=h['Ey'][0].astype(float)
        ax.plot(t,a-b,label='С целью − без цели',color='#196cba')
        ax.plot(t,p-a,label='PML 20 см − PML 12 см',color='#a44535',ls='--')
        ax.axvspan(*qc['echo_window_ns'],color='#d8e4d2',alpha=.4)
        error=100*qc['pml_difference_relative_to_target_delta_l2_echo_window']
        ax.set(xlim=(4,18),xlabel='Время сырой записи, нс',ylabel='Разность Ey, В/м',title=title+f'\nИзменение границы / отклик цели: {error:.3f}% в окне эха')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.supxlabel('Одна проверочная позиция: Tx X=1,20 м; Rx X=1,30 м. Грунт вне сферы совпадает.\nЗелёное окно выбрано по пику разности «с целью − без цели». Проценты характеризуют только чувствительность к границе, не точность радара.',fontsize=9)
    fig.savefig(DEST/'target_checks.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sections-only',action='store_true');args=p.parse_args()
    sections();route()
    if not args.sections_only:
        bscans()
        if all((DEST/'pilot'/mode/'center_checks.json').exists() for mode,_ in MODES):controls()
    print('READY',DEST,flush=True)
