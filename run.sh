#!/usr/bin/env bash
set -euo pipefail

if [[ -x ".venv/bin/python" ]]; then
    python_bin=".venv/bin/python"
    streamlit_bin=".venv/bin/streamlit"
else
    python_bin="python"
    streamlit_bin="streamlit"
fi

"$python_bin" -m pip install -r requirements.txt
exec env PYTHONPATH=".:${PYTHONPATH:-}" "$streamlit_bin" run ui/app.py --server.headless=false --browser.gatherUsageStats=false
