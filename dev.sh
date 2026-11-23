#!/usr/bin/env bash
# Start the Pricer API and frontend together. Press Ctrl+C to stop both.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Project virtualenv not found at $PYTHON. Create it and install the backend dependencies first." >&2
  exit 1
fi
node_version_is_supported() {
  local version major minor
  command -v node >/dev/null 2>&1 || return 1
  version="$(node -p 'process.versions.node' 2>/dev/null)" || return 1
  IFS=. read -r major minor _ <<< "$version"
  [[ "$major" =~ ^[0-9]+$ && "$minor" =~ ^[0-9]+$ ]] || return 1
  (( major >= 23 || (major == 22 && minor >= 12) || (major == 20 && minor >= 19) ))
}

if ! node_version_is_supported; then
  CURRENT_NODE="$(node --version 2>/dev/null || echo missing)"
  echo "Node.js 20.19+ or 22.12+ is required; found $CURRENT_NODE. Install a supported LTS release and retry." >&2
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "npm was not found. Reinstall Node.js with npm included, then retry." >&2
  exit 1
fi
if [[ ! -x "$ROOT_DIR/frontend/node_modules/.bin/vite" ]]; then
  echo "Frontend dependencies are missing; installing them with npm ci..."
  (cd "$ROOT_DIR/frontend" && npm ci)
fi

if ! command -v setsid >/dev/null 2>&1; then
  echo "setsid is required to stop both app processes cleanly (usually provided by util-linux)." >&2
  exit 1
fi

backend_pid=""
frontend_pid=""
stop_services() {
  trap - EXIT INT TERM
  for pid in "$frontend_pid" "$backend_pid"; do
    if [[ -n "$pid" ]]; then
      kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "$frontend_pid" "$backend_pid"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap stop_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting Pricer API at http://127.0.0.1:8000"
setsid "$PYTHON" "$ROOT_DIR/run.py" &
backend_pid=$!

echo "Starting Pricer frontend at http://127.0.0.1:5173"
setsid bash -c 'cd "$1/frontend" && npm run dev -- --host 127.0.0.1' _ "$ROOT_DIR" &
frontend_pid=$!

wait -n "$backend_pid" "$frontend_pid"
