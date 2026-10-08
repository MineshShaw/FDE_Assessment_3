#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
    python3 -m venv .venv
fi

python_bin="$ROOT/.venv/bin/python"
streamlit_bin="$ROOT/.venv/bin/streamlit"
requirements_hash="$(sha256sum requirements.txt | awk '{print $1}')"
requirements_marker="$ROOT/.venv/.requirements-${requirements_hash}"

if [[ ! -f "$requirements_marker" ]]; then
    "$python_bin" -m pip install -r requirements.txt
    find "$ROOT/.venv" -maxdepth 1 -type f -name '.requirements-*' -delete
    touch "$requirements_marker"
fi

api_pid=""
streamlit_pid=""

cleanup() {
    if [[ -n "$streamlit_pid" ]] && kill -0 "$streamlit_pid" 2>/dev/null; then
        kill "$streamlit_pid" 2>/dev/null || true
        wait "$streamlit_pid" 2>/dev/null || true
    fi
    if [[ -n "$api_pid" ]] && kill -0 "$api_pid" 2>/dev/null; then
        kill "$api_pid" 2>/dev/null || true
        wait "$api_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

api_url="${VENDOR_RISK_BASE_URL:-http://127.0.0.1:8001}"
api_url="${api_url%/}"
if curl -fsS "$api_url/health" >/dev/null 2>&1; then
    echo "Using existing vendor-risk API at $api_url."
else
    "$python_bin" -m uvicorn mock_api.app:app --host 127.0.0.1 --port 8001 &
    api_pid=$!
    ready=0
    for _ in {1..60}; do
        if curl -fsS "$api_url/health" >/dev/null 2>&1; then
            ready=1
            break
        fi
        if ! kill -0 "$api_pid" 2>/dev/null; then
            echo "Vendor-risk API exited during startup; port 8001 may be busy." >&2
            exit 1
        fi
        sleep 1
    done
    if [[ "$ready" -ne 1 ]]; then
        echo "Vendor-risk API did not become ready at $api_url/health within 60 seconds." >&2
        exit 1
    fi
fi

echo "Starting Streamlit dashboard..."
env PYTHONPATH=".:${PYTHONPATH:-}" "$streamlit_bin" run ui/app.py \
    --server.headless=false \
    --browser.gatherUsageStats=false &
streamlit_pid=$!
wait "$streamlit_pid"
