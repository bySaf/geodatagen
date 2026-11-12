#!/usr/bin/env python3
"""Package the completed local datasets for an optional GitHub release.

No simulation runs here. Unzip at the root of a cloned repository to restore
the data paths used by the analysis tools. Working solver scratch is omitted.
"""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    import h5py
    import numpy as np
    files = set()
    checks = []
    final = ROOT/'data/final'
    files.update(p for p in final.iterdir() if p.suffix in {'.json', '.txt'} and p.is_file())
    for scene in ('park', 'field', 'road'):
        folder = final/scene
        files.update(p for p in folder.iterdir() if p.is_file() and p.suffix in
                     {'.h5', '.json', '.csv', '.npz', '.in', '.vtkhdf'})
        for n in range(3):
            path = folder/f'bscan_{n}.h5'
            with h5py.File(path) as h:
                assert h['complete'][:].shape == (500,) and np.all(h['complete'][:])
                assert np.isfinite(h['Ey'][:]).all()
                checks.append({'file': str(path.relative_to(ROOT)), 'traces': 500})
    pilot = ROOT/'user_review/realism/pilot'
    files.update(p for p in pilot.iterdir() if p.is_file() and p.suffix in {'.json', '.txt'})
    for mode in ('native_peplinski', 'custom_crim_archie'):
        folder = pilot/mode
        for p in folder.rglob('*'):
            if p.is_file() and not any(v in p.relative_to(folder).parts for v in ('run', '.mpl', '__pycache__')):
                if p.suffix in {'.h5', '.json', '.npz', '.in', '.vtkhdf', '.txt'}:
                    files.add(p)
        for rel, expected in [('bscan.h5', 29), ('background_center/bscan.h5', 1), ('pml_center/bscan.h5', 1)]:
            path = folder/rel
            with h5py.File(path) as h:
                assert h['complete'][:].shape == (expected,) and np.all(h['complete'][:])
                assert np.isfinite(h['Ey'][:]).all()
                checks.append({'file': str(path.relative_to(ROOT)), 'traces': expected})
    dest = ROOT/'dist'
    dest.mkdir(exist_ok=True)
    target = dest/'geodatagen-datasets-v1.zip'
    tmp = target.with_suffix('.zip.tmp')
    manifest = {'description': 'Completed synthetic datasets; extract into repository root.',
                'total_traces': sum(v['traces'] for v in checks), 'signal_checks': checks, 'files': []}
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(files):
            if p.is_symlink():
                raise ValueError(f'Refusing symlink: {p}')
            name = p.relative_to(ROOT).as_posix()
            before = p.stat().st_mtime_ns
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            z.write(p, name)
            if p.stat().st_mtime_ns != before:
                raise RuntimeError(f'File changed while packaging: {name}')
            manifest['files'].append({'path': name, 'bytes': p.stat().st_size, 'sha256': digest})
        z.writestr('DATASET-MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
        z.writestr('DATASET-README.txt',
            'Распакуйте архив в корень клона geodatagen.\n'
            'data/final: 4500 трасс (парк, поле, дорога).\n'
            'user_review/realism/pilot: 58 трасс двух моделей грунта и 4 контрольных расчёта.\n'
            'Это сохранённые результаты предыдущей версии: их конфигурации и хеши оставлены неизменными.\n'
            'Новые опыты создавайте через geodatagen.py в experiments/.\n'
            'В ParaView открывайте scene.vtkhdf, а не bscan.h5. Новые опыты экспортируют scene/scene.vti.\n'
            'Исходные данные реального GP8800 отдельно: user_review/realism/reference_radar/SFCW.zip в репозитории.\n'
            'Эти синтетические данные не являются калибровкой конкретного георадара.\n')
    with zipfile.ZipFile(tmp) as z:
        if z.testzip() is not None:
            raise RuntimeError('ZIP integrity check failed')
    tmp.replace(target)
    (dest/'dataset-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(f'{target.name}: {len(files)} files, {manifest["total_traces"]} complete traces, {target.stat().st_size/1024**2:.1f} MiB')


if __name__ == '__main__':
    main()
