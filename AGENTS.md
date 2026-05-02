# AGENTS.md — инструкция для AI-агента по репозиторию ITMO-CV

Этот файл — «паспорт» репозитория для агента. Прочитай перед правками.

## О чём репозиторий

Курс **«Компьютерное зрение»** в ИТМО. Здесь хранятся лабораторные работы.
Сценарий:

- **Разработка** — локально на macOS через AI-агентов.
- **Запуск обучения** — в Google Colab (GPU).
- **Хранение результатов** — Google Drive (в Colab) или `outputs/` (локально).

## Структура репозитория

```
.
├── examples/                     — материалы с практик (не трогать)
├── lab<N>-<NAME>/                — самодостаточная лаба
│   ├── lab<N>.ipynb              — ноутбук, плоско в корне лабы
│   ├── env_config.py             — копия (Colab/local detect, device, save_dir)
│   ├── config.py                 — пути и dataset/model spec лабы
│   ├── data.py / models.py / engine.py — код лабы
│   ├── build_index.py            — препроцессинг датасета (если нужен)
│   ├── requirements.txt          — зависимости лабы
│   ├── task.md                   — задание (не редактируй без просьбы)
│   └── outputs/                  — checkpoints/, logs/, figures/ (gitignored)
└── .cursor/rules/                — правила для агента
```

**Каждая лаба самодостаточна.** Все её зависимости (`env_config.py`, `requirements.txt`, скрипты) лежат внутри `lab<N>-<NAME>/`.

## Лабораторные

- Каждая лаба — отдельная папка `lab<N>-<NAME>/` с **плоской** структурой (ноутбук и скрипты лежат в корне лабы, без подпапок `notebooks/`/`scripts/`).
- `task.md` — условие задания (не редактируй без просьбы).
- `env_config.py` живёт внутри лабы. Экспортирует: `is_colab`, `LAB_DIR`, `PROJECT_NAME`, `DEVICE`, `get_device()`, `get_save_dir()`, `print_env()`.
- `config.py` в каждой лабе:
  - sys.path.insert свой `LAB_DIR` (только `parent`, не `parent.parent`);
  - импортирует `from env_config import LAB_DIR, get_save_dir, is_colab`;
  - определяет пути сохранений: `SAVE_DIR = get_save_dir()`, далее `CHECKPOINT_DIR`, `LOG_DIR`, `FIGURE_DIR`.
- Результаты сохраняй **только через пути из `config.py`**, не хардкодь.
- **Тренировочные гипер-параметры** (`EPOCHS`, `LR`, `WEIGHT_DECAY`) inline в train-ячейках ноутбука — **не в config.py**.

Пример чекпоинта:

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'f1_macro': f1,
}, CHECKPOINT_DIR / f"{model_name}_best.pth")
```

## Примеры с практик

`examples/` — ноутбуки с практик, формат имени **`XX_YY_name.ipynb`**:
- `XX` — номер практики (`01`, `02`, …).
- `YY` — номер файла внутри практики.
- Используй их как справку/исходники решений; **не модифицируй**.

## Локальный запуск

- Python — всегда из **проектного venv**: `.venv/bin/python` (или активированный `source .venv/bin/activate`).
- Homebrew-Python заблокирован PEP 668; глобально `pip install` не ставить.
- Установка зависимостей: `./.venv/bin/pip install -r lab<N>-<NAME>/requirements.txt` (deps лежат внутри лабы).
- Предпочтительный интерпретатор в Cursor — `.venv/bin/python` (Command Palette → *Python: Select Interpreter*).

## Запуск в Colab

1. Открой `lab<N>.ipynb` с GitHub: `https://colab.research.google.com/github/Ma-XD/ITMO-CV/blob/main/lab<N>-<NAME>/lab<N>.ipynb` (или File → Open notebook → GitHub).
2. Запусти **§1 Dev — git pull/clone** — ячейка возьмёт `GITHUB_TOKEN` из Colab Secrets и склонирует/обновит репо в `/content/ITMO-CV/`. Без токена ячейка молча скипнется.
3. Дальше §2+ ноутбук сам делает path setup → drive.mount → `pip install -r requirements.txt` → `print_env()`. Никакого внешнего `colab_setup.py` нет.

## Git / рабочий процесс

- **Никаких `git commit` / `git push` без явного согласия пользователя** в текущем диалоге (см. `.cursor/rules/git-ask-before-commit.mdc`).
- `data/`, `outputs/`, `checkpoints/`, `.venv/`, `.cursor/`, `*.pth`, `*.pt`, `*.onnx`, `.DS_Store` — **не коммитятся** (см. `.gitignore`).
- Для пуша: `git push` (upstream `origin/main` уже настроен).

### Сообщения коммитов

- **Язык — английский.** Conventional Commits prefix (`feat`/`fix`/`docs`/`refactor`/`chore`) + scope, например `feat(lab1)`, `fix(env_config)`.
- **Одно предложение**, отвечающее на вопрос **«зачем этот коммит»** (мотивация, не пересказ диффа). Без тела, без блока с co-author. Пример: `fix(lab1): keep notebook cells runnable standalone so the video demo can replay any cell`.
- **Маленькие коммиты.** Один логически цельный тематический шаг = один коммит. Не лепи `engine.py` + правки `data.py` + изменения в AGENTS.md в один коммит — разнеси.

## Что обсудить с пользователем, прежде чем делать

- Крупные рефакторинги структуры лаб (плоская vs вложенная, контракт `env_config.py`).
- Изменения в `infra/` (затронут все будущие лабы).
- Добавление тяжёлых зависимостей.
- Любые изменения в `examples/`.
- Создание новых лабораторных (структура/название).

## Комментарии в коде

Цель: код читается сам; комментарии — только там, где без них непонятно **почему** так сделано.

- **Не пиши** модульные docstring-шапки («что в этом файле и зачем»).
- **Не пиши** декоративные разделители (`# ----`, `# === Раздел ===`).
- **Не дублируй** код словами (`# импортируем`, `# создаём датасет`).
- **Не раздувай** docstring у очевидных функций; если имя и типы всё объясняют — docstring не нужен.
- **Можно**: короткий docstring у публичного API при неочевидном поведении; inline-комментарий про нетривиальный выбор или ограничение; `argparse` help — это часть интерфейса.
