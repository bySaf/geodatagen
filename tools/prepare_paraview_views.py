"""Run with ParaView pvpython. Build ready-to-open scenes without GUI dialogs."""
from pathlib import Path
import json
import sys
from paraview.simple import *

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/final/paraview'
DEST.mkdir(exist_ok=True)
scene_name=sys.argv[1] if len(sys.argv)>1 else 'park'
if scene_name not in ('park','field','road') or len(sys.argv)>2:
    raise SystemExit('Usage: pvpython prepare_paraview_views.py [park|field|road]')
results=[]
# One scene per process avoids macOS OpenGL texture exhaustion across sessions.
for scene in (scene_name,):
    ResetSession()
    base=ROOT/'data/final'/scene
    reader=OpenDataFile(str(base/'scene.vtkhdf'))
    RenameSource(scene+' - full frozen geometry',reader)
    reader.UpdatePipeline()
    view=GetActiveViewOrCreate('RenderView')
    # Conservative rendering for this Mac: keep the verified FXAA-free GUI path.
    # This changes edge smoothing only, not geometry, materials or GPR signals.
    view.UseFXAA=0
    view.ViewSize=[1400,900]
    view.UseColorPaletteForBackground=0
    view.Background=[.10,.12,.16]
    view.CameraParallelProjection=1
    outline=Show(reader,view);outline.Representation='Outline';outline.DiffuseColor=[.6,.65,.7]
    horizontal=Slice(registrationName='Horizontal soil section - depth 45 cm',Input=reader)
    horizontal.SliceType='Plane';horizontal.SliceType.Origin=[3.5,2.4,.45];horizontal.SliceType.Normal=[0,0,1]
    horizontal.UpdatePipeline()
    d=Show(horizontal,view);d.Representation='Surface';ColorBy(d,('CELLS','Material'))
    d.RescaleTransferFunctionToDataRange(True,False);d.SetScalarBarVisibility(view,True)
    lut=GetColorTransferFunction('Material');lut.ApplyPreset('Cool to Warm',True)
    bar=GetScalarBar(lut,view);bar.Title='Material ID';bar.ComponentTitle='';bar.TitleFontSize=15;bar.LabelFontSize=12
    vertical=Slice(registrationName='Vertical soil section',Input=reader)
    vertical.SliceType='Plane';vertical.SliceType.Origin=[3.5,4.2,.6];vertical.SliceType.Normal=[0,1,0]
    dv=Show(vertical,view);dv.Representation='Surface';ColorBy(dv,('CELLS','Material'))
    clip=Clip(registrationName='Ground only (exclude air above surface)',Input=reader)
    clip.ClipType='Plane';clip.ClipType.Origin=[0,0,.88];clip.ClipType.Normal=[0,0,1];clip.Invert=1
    clip.UpdatePipeline()
    if scene!='field':
        for label,mid,color in [('Metal',0,[1,.25,.1]),('Air void',1,[.2,.85,1]),('Dielectric',2 if scene=='park' else 4,[1,.75,.1])]:
            th=Threshold(registrationName=label+' - actual voxel cells',Input=clip)
            th.Scalars=['CELLS','Material'];th.LowerThreshold=mid;th.UpperThreshold=mid
            th.UpdatePipeline();dd=Show(th,view);dd.Representation='Surface';dd.ColorArrayName=['POINTS',''];dd.DiffuseColor=color;dd.AmbientColor=color
    for filename,color in [('overlap_a',[1,.2,.35]),('overlap_b',[1,.8,.15]),('separate_curve',[.15,.85,1])]:
        csv=CSVReader(registrationName=filename+' coordinates',FileName=[str(base/(filename+'.csv'))])
        points=TableToPoints(registrationName=filename+' Tx positions',Input=csv)
        points.XColumn='tx_x_m';points.YColumn='tx_y_m';points.ZColumn='tx_z_m'
        dp=Show(points,view);dp.Representation='Points';dp.PointSize=3;dp.ColorArrayName=['POINTS',''];dp.DiffuseColor=color;dp.AmbientColor=color
    txt=Text(registrationName='View guide')
    txt.Text=f'{scene.upper()} | frozen gprMax geometry\nSoil slices + actual object voxels + Tx positions\nColor scale: material IDs, not permittivity\nHorizontal slice: 45 cm below surface'
    td=Show(txt,view);td.FontSize=16;td.Color=[.95,.95,.95];td.WindowLocation='Upper Left Corner'
    view.CameraPosition=[10,-12,13];view.CameraFocalPoint=[3.5,2.4,.45];view.CameraViewUp=[0,0,1]
    view.ResetCamera();view.CameraParallelScale*=1.12
    SetActiveSource(horizontal)
    Render(view)
    SaveState(str(DEST/(scene+'.pvsm')))
    SaveScreenshot(str(DEST/(scene+'_preview.png')),view,ImageResolution=[1400,900])
    results.append(dict(scene=scene,cells=reader.GetDataInformation().GetNumberOfCells(),bounds=list(reader.GetDataInformation().GetBounds()),state=str(DEST/(scene+'.pvsm')),rendered=True))
    print('READY',scene,flush=True)
(DEST/(scene_name+'_view_checks.json')).write_text(json.dumps(results,indent=2))
