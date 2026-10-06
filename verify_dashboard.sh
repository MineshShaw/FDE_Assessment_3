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

for _ in {1..7}; do
    sleep 1
    if [[ "$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501/_stcore/health)" == "200" ]]; then
        echo "Dashboard healthcheck passed (HTTP 200)."
        exit 0
    fi
done

echo "Dashboard healthcheck failed." >&2
cat "$log_file" >&2
exit 1
