# Установка и запуск

Нужны Git и Python 3.12 или 3.13. Поддерживаются macOS и Linux; для Windows используется WSL2. ParaView устанавливается отдельно.

```bash
git clone https://github.com/bySaf/geodatagen.git
cd geodatagen
bash scripts/setup.sh
source .venv/bin/activate
```

Проверка окружения:

```bash
bash scripts/setup.sh --check
```

В PyCharm откройте корень `geodatagen`, затем Settings → Python Interpreter → Add Local Interpreter → Existing environment. Выберите `.venv/bin/python` внутри проекта. Если установщик сообщает `Requires-Python >=3.12`, проверьте `python --version`: системный Python 3.9 не подходит.

## Создать эксперимент

```bash
python geodatagen.py init park-demo --scene examples/scenes/park-pilot.json --trajectory examples/trajectories/pilot-line.json
python geodatagen.py docs park-demo
```

`init` строит геометрию и HTML-предпросмотр. Уже можно проверить маршрут, объекты, разрезы грунта и экспорт материалов. Волновые сигналы на этом этапе ещё не рассчитаны.

## Рассчитать сигналы

Сначала одна новая трасса:

```bash
python geodatagen.py run park-demo --limit 1
python geodatagen.py status park-demo
```

Затем оставшиеся:

```bash
python geodatagen.py run park-demo
python geodatagen.py docs park-demo
```

Графики и отчёт обновляются после расчёта. Повторный `run` продолжает опыт с незавершённых позиций. `--limit` считает новые трассы суммарно по всем маршрутам.

## Остановить и продолжить

```bash
python geodatagen.py stop park-demo
python geodatagen.py run park-demo --resume
```

Остановка происходит после текущей трассы. Для изменения сцены или маршрута создайте новый эксперимент с другим именем: настройки уже подготовленного опыта зафиксированы.

## Файлы эксперимента

```text
experiments/park-demo/
├── index.html              предпросмотр и результаты
├── scene_config.json       исходная сцена
├── trajectory_config.json  сохранённый маршрут
├── config.json             разрешённые параметры расчёта
├── objects.json            фактические объекты
├── positions.npz           положения Tx/Rx
├── trajectory.csv          координаты, путь и сегменты
├── scene/                  геометрия и материалы ParaView/gprMax
├── exports/                NumPy и описание массивов
├── bscan.h5                рассчитанные сигналы
└── logs/                   журнал команд
```

Папку можно передавать целиком. Для просмотра `index.html` сервер не нужен. Незавершённые трассы обозначены маской `complete`; в численном экспорте их сигнал заполнен `NaN`.
