# GeoDataGen

Трёхмерные грунтовые сцены и синтетические A- и B-сканы в gprMax. Сцена и траектория задаются отдельно: Peplinski или модель полей CRIM/Archie, сферические цели, фиксированные и случайные маршруты. Каждый эксперимент хранится в своей папке вместе с предпросмотром, сигналами, NumPy и ParaView.

[Документация](https://bysaf.github.io/geodatagen/) · [Исходники руководства](docs/README.md) · [Лицензии](THIRD_PARTY_NOTICES.md)

## Начало работы

Нужен Python **3.12 или 3.13**. macOS и Linux поддерживаются напрямую; Windows — через WSL2.

```bash
git clone https://github.com/bySaf/geodatagen.git
cd geodatagen
bash scripts/setup.sh
source .venv/bin/activate
python geodatagen.py init park-demo --scene examples/scenes/park-pilot.json --trajectory examples/trajectories/pilot-line.json
python geodatagen.py docs park-demo
python geodatagen.py run park-demo --limit 1
```

Результат — `experiments/park-demo/`. `init` сразу создаёт HTML-предпросмотр и экспорт материалов. `run` рассчитывает сигналы; повторный запуск без `--limit` завершает оставшиеся позиции. В PyCharm выберите интерпретатор `.venv/bin/python` из папки проекта.

Руководство доступно локально в `user_review/index.html`: [сцена](docs/usage/scene.md), [траектория](docs/usage/trajectory.md), [NumPy и ParaView](docs/usage/exports.md), [CPU/GPU](docs/usage/performance.md), [научная основа](docs/theory/model.md).

Используются идеальные Tx/Rx; соответствие конкретному реальному прибору не заявляется. Полные сцены по умолчанию рассчитываются движущимся окном, которое ограничивает дальние отражения. В документации указаны постановка, единицы и ограничения каждого примера.
