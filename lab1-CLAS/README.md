# Lab 1 — DVM Color Classification

Классификация цвета автомобиля по фронтальному фото из датасета [DVM](https://deepvisualmarketing.github.io/) (top-6 цветов: `black, grey, white, blue, silver, red`). Цель — `F1_macro > 0.8` на test.

Сравниваются три модели на одинаковом budget'е в 12 эпох:

- `custom_resnet18` — ResNet-18, написанный руками (from-scratch, без ImageNet);
- `resnet18` — fine-tune `torchvision.models.resnet18` (ImageNet), param groups: head LR=1e-3, backbone LR=1e-4;
- `mobilenet_v3_small` — fine-tune, та же стратегия. Лёгкая модель из другого семейства.

## Структура

```
lab1-CLAS/
├── lab1.ipynb              — главный ноутбук, запускается в Colab
├── env_config.py           — Colab/local detect, device, save_dir
├── config.py               — пути, dataset spec, model registry, normalization
├── data.py                 — Dataset, splits 70/15/15, WeightedRandomSampler
├── models.py               — CustomResNet18 + pretrained-обёртки + param_groups
├── engine.py               — train_one_epoch / validate / fit / evaluate / load_checkpoint
├── build_index.py          — парсит имена файлов и собирает index.csv
├── requirements.txt        — зависимости лабы
├── task.md                 — условие задания
└── outputs/                — checkpoints/ + logs/ + figures/ (генерируется обучением)
```

## Запуск в Colab

Ноутбук рассчитан на запуск из папки `lab1-CLAS/`. Первая ячейка (`§1 Path setup`) сама детектит папку и делает `cd` + `sys.path`.

### Сценарий A — разработка (с GitHub)

1. В Colab один раз клонируй репо в `/content/`:
   ```python
   import os
   from google.colab import userdata
   TOKEN = userdata.get("GITHUB_TOKEN")
   !git clone https://{TOKEN}@github.com/Ma-XD/ITMO-CV.git /content/ITMO-CV
   ```
2. `Runtime → Change runtime type → T4 GPU`.
3. Открой `lab1-CLAS/lab1.ipynb` через `File → Open notebook → GitHub` или напрямую.
4. Перед каждым новым прогоном — повтори `git pull` в `/content/ITMO-CV`, в Colab UI у ноутбука `File → Revert`.

### Сценарий B — без GitHub (для проверяющего)

1. Распакуй zip папки `lab1-CLAS/` в `/content/` через `Files panel` Colab.
2. Открой `lab1.ipynb` из распакованной папки.
3. Положи zip датасета в `/content/dvm_confirmed_fronts.zip` (см. ниже).
4. `Runtime → Change runtime type → T4 GPU`.

В обоих сценариях ноутбук разбит на две части:

- **Часть 1 — Подготовка** (§1–§6): path setup → Drive mount → `pip install` → verify env → распаковка датасета → `build_index.py`.
- **Часть 2 — Лаба** (§7–§13): характеристики моделей → 3 тренировки → инференс и оценка → сравнение → вывод.

## Датасет

DVM, [сайт проекта](https://deepvisualmarketing.github.io/). Используются только confirmed фронтальные виды.

Архив `dvm_confirmed_fronts.zip` (~765 МБ) ноутбук ищет в трёх местах (по порядку):

1. `/content/drive/MyDrive/ITMO-CV/lab1-CLAS/data/dvm_confirmed_fronts.zip` — для dev-флоу с Drive.
2. `/content/dvm_confirmed_fronts.zip` — если загрузил напрямую в Colab.
3. `./dvm_confirmed_fronts.zip` — рядом с ноутбуком.

После распаковки — `/content/data/dvm/confirmed_fronts/*.jpg` (61 827 файлов). Имена парсятся по `$$` в `build_index.py`, off-target цвета фильтруются → `index.csv` с колонками `path,maker,model,year,color,label`.

## Обучение

Тренируются три модели последовательно, ячейки §8 / §9 / §10. Каждая ячейка независима: своими импортами и `device = get_device()`, своим оптимизатором, своим scheduler'ом.

Гипер-параметры **inline в ячейках** (видны при просмотре):

```python
EPOCHS = 12
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4   # для pretrained, не используется для custom
WEIGHT_DECAY = 1e-4
```

Что происходит при запуске `fit(...)`:

- 12 эпох. Каждая эпоха: train pass → val pass → `scheduler.step()`.
- В val считаются loss / accuracy / `f1_macro` (sklearn).
- После последней эпохи сохраняется:
  - `outputs/checkpoints/{model_name}_last.pth` — `state_dict + history + epoch`;
  - `outputs/logs/{model_name}_history.json` — `train_loss/val_loss/val_acc/val_f1_macro/lr_*` по эпохам (для построения кривых).

Время на T4 GPU (приблизительно):

- `custom_resnet18`: ~30–40 мин;
- `resnet18` fine-tune: ~25–30 мин;
- `mobilenet_v3_small` fine-tune: ~12–15 мин.

Если Colab дисконнектнулся — повторно запусти только нужную train-ячейку. Чекпоинт затирается, остальные модели не пострадают.

## Инференс и оценка

Секция §11 — независимая от обучения: грузит чекпоинты и считает метрики на test.

```python
from engine import evaluate, load_checkpoint
model = build_model(name).to(device)
load_checkpoint(model, name, device=device)
res = evaluate(model, loaders["test"], device=device, class_names=TARGET_COLORS)
# res: {accuracy, f1_macro, f1_per_class, confusion_matrix, y_true, y_pred}
```

Ячейки этой секции:

1. Eval всех трёх моделей → словарь `results`.
2. Confusion matrices (heatmap) сохраняются в `outputs/figures/confusion_matrices.png`.
3. Сетка 3×4 случайных тестовых картинок с GT vs prediction (используется `resnet18`) → `outputs/figures/inference_examples.png`.

Секция §12 — **сравнение моделей**: сводная `pandas.DataFrame` (params, accuracy, F1_macro, per-class F1) + кривые `train/val loss` и `val_f1_macro` по эпохам с пунктирной линией target=0.8.

## Видео-демо

При записи видео:

- **Не запускай** ячейки §8 / §9 / §10 (`fit`) — это часы. Просто покажи и проговори: «Adam, CosineAnnealingLR, 12 эпох, гиперы видны прямо в ячейке». Чекпоинты уже сохранены в `outputs/checkpoints/`.
- **Запускай live**: §7 (характеристики моделей — секунды), §11 (инференс с чекпоинтов — единицы минут на test), §12 (сравнение — секунды).
- Confusion matrices и сетка предсказаний — кульминация: рассказываешь про типичные ошибки (`silver ↔ grey ↔ white`).

## Сдача

Папку `lab1-CLAS/` zip-нуть **без `outputs/`** (он в `.gitignore`, но если решишь приложить — добавляй только `figures/`, `logs/`, не `*.pth`).

Минимальный пакет для проверяющего:

- `lab1-CLAS.zip` — самодостаточная папка лабы;
- видео-демо (через форму);
- если требуется визуальный артефакт обучения — добавь `outputs/figures/*.png`.

Чекпоинты `*.pth` обычно не отправляют — они большие, а проверяющий их сам не сгенерирует без датасета.
