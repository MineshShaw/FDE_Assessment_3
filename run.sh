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

api_pid=""
cleanup() {
    if [[ -n "$api_pid" ]] && kill -0 "$api_pid" 2>/dev/null; then
        kill "$api_pid" 2>/dev/null || true
        wait "$api_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

"$python_bin" -m uvicorn mock_api.app:app --host 127.0.0.1 --port 8001 &
api_pid=$!

for _ in {1..20}; do
    if curl -fsS http://127.0.0.1:8001/health >/dev/null; then
        break
    fi
    sleep 0.25
done

if ! curl -fsS http://127.0.0.1:8001/health >/dev/null; then
    echo "Vendor-risk mock API failed to start." >&2
    exit 1
fi

env PYTHONPATH=".:${PYTHONPATH:-}" "$streamlit_bin" run ui/app.py \
    --server.headless=false \
    --browser.gatherUsageStats=false
