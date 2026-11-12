# NumPy и ParaView

```bash
python geodatagen.py export park-demo
```

Экспорт сохраняется в папке эксперимента. Геометрия доступна сразу после `init`; массивы сигналов появляются после начала расчёта.

## NumPy

- `exports/geometry.npz` — геометрия и поля материалов;
- `exports/signals.npz` — сигналы, координаты, маршруты и маска готовности;
- `exports/metadata.json` — единицы, оси и смысл полей;
- `exports/materials.json` — таблица материалов и параметры дисперсии.

Все массивы читаются без pickle:

```python
import numpy as np

with np.load("experiments/park-demo/exports/geometry.npz", allow_pickle=False) as g:
    epsilon = g["epsilon_real"]       # [x, y, z]
    x, y, z = g["x_m"], g["y_m"], g["z_m"]

with np.load("experiments/park-demo/exports/signals.npz", allow_pickle=False) as s:
    ready = s["complete"]
    traces = s["Ey"][ready]           # [scan, time], В/м
    time_s = s["time_s"]
    positions = s["tx_m"][ready]
```

`x_m`, `y_m`, `z_m` — центры ячеек. Нерассчитанные сигналы заполнены `NaN`; не используйте их как нулевые измерения. Нормализация графика не меняет экспортированный сырой `Ey`.

## Поля материалов

| Поле | Смысл |
| --- | --- |
| `material_id`, `material_keys` | Номер ячейки в таблице материалов и соответствующие ключи |
| `epsilon_real` | Действительная относительная проницаемость на указанной частоте |
| `epsilon_imag` | Положительная величина потерь, включая вклад проводимости |
| `epsilon_infinite` | Высокочастотная часть относительной проницаемости |
| `conductivity_dc_S_m` | Статическая проводимость, См/м |
| `conductivity_effective_S_m` | Проводимость вместе с эквивалентными потерями Debye, См/м |
| `permeability_relative` | Относительная магнитная проницаемость |
| `magnetic_conductivity_S_m` | Магнитная проводимость в обозначениях gprMax |
| `debye_pole_count` | Число полюсов Debye |
| `object_type`, `object_mask` | Тип и маска включений |
| `layer_id` | Слой: 0 — воздух, −1 — цели, положительные номера — грунт сверху вниз |

Соглашение о комплексной величине: `epsilon* = epsilon_real - i·epsilon_imag` при временной зависимости `exp(+iωt)`. Частота оценки хранится в `frequency_hz`. В PEC обычные скалярные электрические свойства не определены; используйте маску объекта.

Для модели полей дополнительно доступны `porosity`, `saturation`, `water_fraction`, `continuous_epsilon_real` и `continuous_conductivity_S_m`. Это исходные непрерывные поля до разбиения на палитру материалов. Фактический расчёт использует квантованные поля из экспортированной геометрии. Для штатного Peplinski не подставляются вымышленные значения пористости или непрерывной влажности.

## Открыть в ParaView

1. File → Open → `scene/scene.vti` → Apply.
2. В Color By выбрать `epsilon_real`, затем Rescale to Data Range.
3. Filters → Slice → Plane. Для вертикального среза X–Z задать Normal `[0, 1, 0]`, а Origin Y — нужную координату.
4. Нажать Apply, скрыть исходный объём и выключить Show Plane.
5. Направить камеру вдоль Y. Для горизонтального среза использовать Normal `[0, 0, 1]`.

В `scene/trajectory.vtp` находятся линии Tx/Rx, в `scene/objects.vtp` — поверхности заданных сфер. Откройте их вместе с VTI для совмещённого просмотра. Поверхности сфер показывают заданную форму, VTI — её растеризацию на сетку.

Чтобы оставить только цели, примените Threshold к `object_type` в диапазоне 1–3. Коды: 1 — PEC, 2 — воздушная полость, 3 — диэлектрик. `material_id` — служебный индекс, а не величина проницаемости.

Все координаты — в метрах; Z вверх. Глубина равна `surface_z − Z`. Конкретные координаты срезов и ссылки на файлы доступны в HTML-предпросмотре опыта.
