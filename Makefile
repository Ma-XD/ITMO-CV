# Интерпретатор из виртуального окружения (torch и остальное — там)
PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help venv install print-env

help:
	@echo "Цели:"
	@echo "  make venv       — создать .venv (если ещё нет)"
	@echo "  make install    — pip install -r requirements.txt в .venv"
	@echo "  make print-env  — env_config.print_env() через .venv"

venv:
	@test -d .venv || python3 -m venv .venv
	@echo "✅ .venv готов"

install: venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt
	@echo "✅ Зависимости установлены"

print-env: venv
	@test -x $(PYTHON) || { echo "❌ Нет $(PYTHON). Запусти: make install"; exit 1; }
	@$(PYTHON) -c "from env_config import print_env; print_env()"
