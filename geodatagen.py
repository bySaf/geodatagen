#!/usr/bin/env python3
"""One experiment = one folder. Run --help for the short workflow."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import webbrowser

ROOT = Path(__file__).resolve().parent
MODES = ('native_peplinski', 'custom_crim_archie')


def experiment_path(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name):
        raise ValueError('Имя: 1–80 латинских букв, цифр, дефисов или подчёркиваний; без пути.')
    experiment_root = ROOT / 'experiments'
    if experiment_root.is_symlink():
        raise ValueError('experiments/ должна быть обычной папкой, а не ссылкой.')
    base = experiment_root.resolve()
    candidate = base / name
    if candidate.is_symlink():
        raise ValueError('Папка эксперимента не может быть символической ссылкой.')
    path = candidate.resolve()
    if path.parent != base:
        raise ValueError('Папка эксперимента должна находиться внутри experiments/.')
    return path


def append_log(directory, message):
    logs = directory / 'logs'; logs.mkdir(exist_ok=True)
    with (logs / 'commands.log').open('a') as stream:
        stream.write(f'\n[{datetime.now(timezone.utc).isoformat()}]\n{message}\n')


def invoke(directory, command, options=()):
    args = [sys.executable, str(ROOT / 'tools' / 'experiment.py'), command, '--data', str(directory), *options]
    # Preparation rejects a nonempty destination; write its command log only
    # after the subprocess has created its immutable config.
    if command == 'prepare':
        run = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(run.stdout, end='')
        if directory.exists():append_log(directory, ' '.join(args) + '\n' + run.stdout)
        if run.returncode:raise RuntimeError(f'Подготовка завершилась с ошибкой ({run.returncode}); проверьте журнал.')
        return
    append_log(directory, ' '.join(args))
    with (directory / 'logs' / 'commands.log').open('a') as log:
        process = subprocess.Popen(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
            code = process.wait()
        except KeyboardInterrupt:
            (directory / 'STOP').touch()
            print('\nЗапрошена остановка. Для продолжения: run с флагом --resume.', file=sys.stderr)
            process.wait()
            raise
        if code:raise RuntimeError(f'Команда завершилась с ошибкой ({code}); смотрите logs/commands.log и run/last.log.')


def refresh(directory, export=False, plots=False):
    sys.path.insert(0, str(ROOT / 'tools'))
    import experiment
    import experiment_outputs
    state = experiment.status(directory)
    if state.get('running'):
        return state
    with experiment.locked(directory):
        if export or not (directory / 'scene' / 'scene.vti').exists():
            experiment_outputs.export_geometry(directory)
        if plots and state.get('completed', 0):experiment.plot(directory)
        sweep_metadata = directory / 'effective_sfcw.json'
        if sweep_metadata.is_file():
            meta = json.loads(sweep_metadata.read_text())
            duration_ns = 1e9 / meta['finite_record_frequency_resolution_hz']
            step_mhz = meta['frequency_step_hz'] / 1e6
            meta['limitations'][1] = (
                f'Frequency sampling: {step_mhz:g} MHz; finite record: {duration_ns:.6g} ns. '
                'Frequency interpolation does not create independently resolved frequencies.')
            sweep_metadata.write_text(json.dumps(meta, indent=2) + '\n')
        experiment_outputs.report(directory, state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description='Geodatagen: отдельная папка на каждый эксперимент.')
    subs = parser.add_subparsers(dest='command', required=True)
    init = subs.add_parser('init', help='Подготовить сцену, ParaView и страницу эксперимента')
    init.add_argument('name'); init.add_argument('--mode', choices=MODES, default='custom_crim_archie')
    init.add_argument('--config', type=Path, default=ROOT / 'examples' / 'park-objects.json')
    run = subs.add_parser('run', help='Рассчитать или продолжить маршрут; графики обновятся автоматически')
    run.add_argument('name'); run.add_argument('--limit', type=int); run.add_argument('--resume', action='store_true', help='Удалить локальный STOP и продолжить')
    for command, help_text in [('status', 'Показать прогресс'), ('plot', 'Обновить графики и отчёт'), ('export', 'Обновить VTI для ParaView и отчёт'), ('stop', 'Остановиться после текущей трассы')]:
        sub = subs.add_parser(command, help=help_text); sub.add_argument('name')
    docs = subs.add_parser('docs', help='Открыть документацию или отчёт эксперимента')
    docs.add_argument('name', nargs='?'); docs.add_argument('--no-open', action='store_true', help='Только вывести путь')
    args = parser.parse_args(argv)
    if args.command == 'docs':
        path = experiment_path(args.name) / 'index.html' if args.name else ROOT / 'user_review' / 'index.html'
        if not path.is_file():raise ValueError('Страница ещё не создана; сначала выполните init.')
        print(path)
        if not args.no_open:webbrowser.open(path.as_uri())
        return 0
    directory = experiment_path(args.name)
    if args.command == 'init':
        source = args.config.resolve()
        if not source.is_file():raise ValueError(f'Не найден JSON: {source}')
        if directory.exists():raise ValueError('Это имя уже занято. Для нового опыта выберите новое имя; run продолжает прежний.')
        # Capture the input before invoking preparation; a subsequent edit must
        # not replace the exact input which created this experiment.
        original = source.read_text(); json.loads(original)
        invoke(directory, 'prepare', ['--config', str(source), '--mode', args.mode])
        (directory / 'source.json').write_text(original)
        refresh(directory, export=True)
    else:
        if not (directory / 'config.json').is_file():raise ValueError('Эксперимент не подготовлен. Сначала выполните init.')
        if args.command == 'stop':
            (directory / 'STOP').touch();print('Остановка запрошена: после текущей трассы. Продолжить: run --resume.');return 0
        if args.command == 'status':
            sys.path.insert(0, str(ROOT / 'tools'))
            import experiment
            print(json.dumps(experiment.status(directory), indent=2, ensure_ascii=False));return 0
        if args.command == 'run':
            if args.limit is not None and args.limit < 1:raise ValueError('--limit должен быть положительным.')
            if args.resume:(directory / 'STOP').unlink(missing_ok=True)
            if (directory / 'STOP').exists():raise ValueError('Есть файл STOP. Для продолжения добавьте --resume.')
            invoke(directory, 'run', ['--limit', str(args.limit)] if args.limit else [])
            refresh(directory, plots=True)
        elif args.command == 'plot':refresh(directory, plots=True)
        elif args.command == 'export':refresh(directory, export=True)
    print(f'Отчёт: {directory / "index.html"}')
    print(f'ParaView: {directory / "scene" / "scene.vti"}')
    return 0


if __name__ == '__main__':
    try:raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, ModuleNotFoundError) as exc:
        raise SystemExit(f'Ошибка: {exc}')
