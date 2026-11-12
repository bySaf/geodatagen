# GeoDataGen

Генератор трёхмерных грунтовых сцен и синтетических A- и B-сканов в gprMax. Сцена и траектория задаются отдельными JSON: грунт Peplinski или CRIM/Archie, сферические цели, источник и произвольный маршрут.

[Руководство с примерами и рисунками](https://bysaf.github.io/geodatagen/) · [Локальная документация](user_review/index.html) · [Лицензии и атрибуция](THIRD_PARTY_NOTICES.md)

## Установка

Нужен Python **3.12 или 3.13**. Поддерживаются macOS и Linux; на Windows используйте WSL2.

```bash
git clone https://github.com/bySaf/geodatagen.git
cd geodatagen
bash scripts/setup.sh
source .venv/bin/activate
python geodatagen.py docs
```

Документация также открывается прямо из `user_review/index.html`, без Python и интернета. В PyCharm выберите существующий интерпретатор `<папка проекта>/.venv/bin/python`.

## Первый опыт

```bash
python geodatagen.py init park-demo --scene examples/scenes/park-pilot.json --trajectory examples/trajectories/pilot-line.json
python geodatagen.py run park-demo --limit 1
python geodatagen.py plot park-demo
python geodatagen.py docs park-demo
```

Результаты находятся в `experiments/park-demo/`: сохранённые настройки сцены и пути, координаты объектов и Tx/Rx, схема сцены, геометрия для ParaView и сигналы. После одной позиции доступен A-скан; для B-скана рассчитайте остальные позиции:

```bash
python geodatagen.py run park-demo
python geodatagen.py plot park-demo
```

## Полная сцена и свой маршрут

```bash
python geodatagen.py init park-full --scene examples/scenes/park.json --trajectory examples/trajectories/snake.json
python geodatagen.py docs park-full
python geodatagen.py run park-full --limit 1
```

В `examples/scenes/` есть парк, поле и дорога размером 7 × 4.8 × 1.2 м. Скопируйте подходящую сцену и траекторию, отредактируйте их и создайте опыт с новым именем. Доступны прямая, змейка, ломаная, список точек, CSV и несколько маршрутов над одной сценой. Генератор выбран полем `generator`; `--mode native_peplinski` или `--mode custom_crim_archie` переопределяет его при подготовке.

Полные сцены рассчитываются локальным движущимся окном 1.2 × 1.2 м. Дальние отражения за его границами могут отсутствовать. Используются идеальные Tx/Rx с фиксированной ориентацией; точность конкретного прибора не подтверждена. Форматы JSON, координаты, ограничения и научные источники приведены в [руководстве](user_review/index.html).

## Геометрия и проверка установки

```bash
python geodatagen.py export park-demo
bash scripts/setup.sh --check
```

В [ParaView](https://www.paraview.org/download/) откройте `experiments/park-demo/scene/scene.vti` и нажмите **Apply**. Геометрия, B-скан и входной файл gprMax — разные виды данных; пошаговая инструкция находится в [разделе ParaView](user_review/index.html#paraview).

Исторические численные данные доступны отдельно в [релизе v1.0.0](https://github.com/bySaf/geodatagen/releases/tag/v1.0.0); новым экспериментам они не нужны. Исходники находятся в `src/` и `tools/`, проверки — в `tests/`.
