#!/usr/bin/env bash
set -euo pipefail
app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$app_dir"
if [[ ! -x .venv/bin/python ]]; then
    echo "Set up the app first:"
    echo "  python3 -m venv .venv"
    echo "  .venv/bin/python -m pip install -r requirements.txt"
    exit 1
fi
exec .venv/bin/python -m streamlit run app.py "$@"
