#!/usr/bin/env bash
# Запуск Python из .venv репозитория: ./py -c "..." или ./py script.py
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "❌ Нет ${PY}" >&2
  echo "   Создай окружение и поставь зависимости:" >&2
  echo "   make install" >&2
  echo "   (или: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt)" >&2
  exit 1
fi
exec "$PY" "$@"
