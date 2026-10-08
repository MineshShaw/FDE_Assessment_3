#!/usr/bin/env bash
set -euo pipefail

log_file="$(mktemp)"
streamlit_pid=""

cleanup() {
    if [[ -n "$streamlit_pid" ]] && kill -0 "$streamlit_pid" 2>/dev/null; then
        kill "$streamlit_pid" 2>/dev/null || true
        wait "$streamlit_pid" 2>/dev/null || true
    fi
    rm -f "$log_file"
}
trap cleanup EXIT

./run.sh >"$log_file" 2>&1 &
streamlit_pid=$!

for _ in {1..60}; do
    sleep 1
    dashboard_status="$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501/_stcore/health || true)"
    api_status="$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8001/health || true)"
    if [[ "$dashboard_status" == "200" && "$api_status" == "200" ]]; then
        echo "Dashboard healthcheck passed (HTTP 200)."
        exit 0
    fi
done

echo "Dashboard or vendor-risk API healthcheck failed." >&2
cat "$log_file" >&2
exit 1
