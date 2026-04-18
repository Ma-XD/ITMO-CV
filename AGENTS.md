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
├── env_config.py                 — автодетект Colab/Local, device, пути
├── colab_setup.py                — настройка среды в Colab (GPU, Drive, pip, импорт-тест)
├── requirements.txt              — общие зависимости (torch, sklearn, …)
├── notebooks/
│   └── colab_template.ipynb      — шаблон ноутбука для Colab
├── scripts/
│   └── sync_results.py           — локальная утилита для выгрузки результатов с Drive
├── examples/                     — материалы с практик (не трогать)
├── lab<N>-<NAME>/                — лабораторные (сейчас есть lab1-CLAS)
│   ├── task.md                   — задание
│   ├── config.py                 — пути для сохранений (обязателен, см. ниже)
│   └── train.py                  — точка входа обучения
└── .cursor/rules/                — правила для агента
```

## Лабораторные

- Каждая лаба — отдельная папка `lab<N>-<NAME>/`.
- `task.md` — условие задания (не редактируй без просьбы).
- `config.py` в каждой лабе:
  - импортирует `env_config` из корня;
  - определяет пути через `env_config.get_save_dir(<project_name>)`:
    - `SAVE_DIR`, `CHECKPOINT_DIR` (для `.pth`), `LOG_DIR` (для `.json`), `FIGURE_DIR` (для `.png`).
- `train.py` **запускается из корня репо**: `python lab1-CLAS/train.py`.
- Результаты сохраняй **только через пути из `config.py`**, не хардкодь.

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
- Установка зависимостей: `./.venv/bin/pip install -r requirements.txt`.
- Предпочтительный интерпретатор в Cursor — `.venv/bin/python` (Command Palette → *Python: Select Interpreter*).

## Запуск в Colab

- Открой `notebooks/colab_template.ipynb` (или копию), порядок ячеек:
  1. Clone / Pull (нужен Colab-secret `GITHUB_TOKEN`).
  2. `drive.mount('/content/drive')` — **в ячейке ноутбука**, а не из `!python`.
  3. `!python colab_setup.py` — проверки GPU/Drive, установка пакетов.
  4. `from env_config import print_env; print_env()`.

## Git / рабочий процесс

- **Никаких `git commit` / `git push` без явного согласия пользователя** в текущем диалоге (см. `.cursor/rules/git-ask-before-commit.mdc`).
- `data/`, `outputs/`, `checkpoints/`, `.venv/`, `.cursor/`, `*.pth`, `*.pt`, `*.onnx`, `.DS_Store` — **не коммитятся** (см. `.gitignore`).
- Для пуша: `git push` (upstream `origin/main` уже настроен).

## Что обсудить с пользователем, прежде чем делать

- Крупные рефакторинги общей инфраструктуры (`env_config.py`, `colab_setup.py`, шаблон ноутбука).
- Добавление тяжёлых зависимостей.
- Любые изменения в `examples/`.
- Создание новых лабораторных (структура/название).

## Связанные документы

- `.cursor/rules/git-ask-before-commit.mdc` — правило про согласование коммитов.
- `.cursor/rules/repo-guide.mdc` — краткий указатель на этот файл.
