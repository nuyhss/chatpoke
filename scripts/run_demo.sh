#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$PROJECT_DIR/.run"
PID_FILE="$RUN_DIR/demo_server.pid"
LOG_FILE="$RUN_DIR/demo_server.log"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
RELOAD="${RELOAD:-true}"

UVICORN_BIN="$PROJECT_DIR/.venv/bin/uvicorn"
PY_BIN="$PROJECT_DIR/.venv/bin/python"

usage() {
  cat <<USAGE
Usage:
  ./scripts/run_demo.sh start
  ./scripts/run_demo.sh stop
  ./scripts/run_demo.sh restart
  ./scripts/run_demo.sh status
  ./scripts/run_demo.sh logs

Env overrides:
  PORT=8001 HOST=0.0.0.0 RELOAD=true ./scripts/run_demo.sh start
USAGE
}

is_running_pid() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

port_in_use() {
  local port="$1"
  ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${port}$"
}

health_check() {
  curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1
}

ensure_ready() {
  mkdir -p "$RUN_DIR"

  if [[ ! -x "$UVICORN_BIN" || ! -x "$PY_BIN" ]]; then
    echo "[ERROR] .venv not found."
    echo "Run:"
    echo "  cd $PROJECT_DIR"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
  fi

  if [[ ! -f "$PROJECT_DIR/.env" && -f "$PROJECT_DIR/.env.example" ]]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo "[INFO] .env created from .env.example"
  fi
}

start_server() {
  ensure_ready

  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if is_running_pid "$pid"; then
      echo "[INFO] Server already running (PID: $pid)"
      echo "      UI:   http://127.0.0.1:${PORT}/ui"
      echo "      Docs: http://127.0.0.1:${PORT}/docs"
      exit 0
    else
      rm -f "$PID_FILE"
    fi
  fi

  if port_in_use "$PORT"; then
    echo "[ERROR] Port ${PORT} is already in use."
    echo "Try another port: PORT=8002 ./scripts/run_demo.sh start"
    exit 1
  fi

  local reload_flag=""
  if [[ "$RELOAD" == "true" ]]; then
    reload_flag="--reload"
  fi

  echo "[INFO] Starting demo server..."
  (
    cd "$PROJECT_DIR"
    nohup "$UVICORN_BIN" main:app \
      --app-dir "$PROJECT_DIR" \
      --host "$HOST" \
      --port "$PORT" \
      $reload_flag \
      >>"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
  )

  sleep 2

  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if ! is_running_pid "$pid"; then
    echo "[ERROR] Failed to start server."
    echo "Last logs:"
    tail -n 80 "$LOG_FILE" || true
    exit 1
  fi

  if health_check; then
    echo "[OK] Demo server started (PID: $pid)"
    echo "     UI:   http://127.0.0.1:${PORT}/ui"
    echo "     Docs: http://127.0.0.1:${PORT}/docs"
  else
    echo "[WARN] Server process is running (PID: $pid) but /health is not ready yet."
    echo "       Check logs: ./scripts/run_demo.sh logs"
  fi
}

stop_server() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo "[INFO] No PID file. Server may already be stopped."
    exit 0
  fi

  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"

  if ! is_running_pid "$pid"; then
    rm -f "$PID_FILE"
    echo "[INFO] Stale PID file removed."
    exit 0
  fi

  echo "[INFO] Stopping server (PID: $pid)..."
  kill "$pid" 2>/dev/null || true

  for _ in {1..20}; do
    if ! is_running_pid "$pid"; then
      rm -f "$PID_FILE"
      echo "[OK] Server stopped."
      exit 0
    fi
    sleep 0.25
  done

  echo "[WARN] Graceful stop timed out. Force killing..."
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "[OK] Server force-stopped."
}

status_server() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if is_running_pid "$pid"; then
      echo "[INFO] Server running (PID: $pid)"
      if health_check; then
        echo "[INFO] Health: OK"
      else
        echo "[WARN] Health: NOT READY"
      fi
      echo "      UI:   http://127.0.0.1:${PORT}/ui"
      echo "      Docs: http://127.0.0.1:${PORT}/docs"
      exit 0
    fi
  fi

  if port_in_use "$PORT"; then
    echo "[WARN] Port ${PORT} is in use, but not by this script-managed PID."
  else
    echo "[INFO] Server is not running."
  fi
}

show_logs() {
  if [[ ! -f "$LOG_FILE" ]]; then
    echo "[INFO] No log file yet: $LOG_FILE"
    exit 0
  fi
  tail -n 120 "$LOG_FILE"
}

CMD="${1:-start}"
case "$CMD" in
  start) start_server ;;
  stop) stop_server ;;
  restart) stop_server || true; start_server ;;
  status) status_server ;;
  logs) show_logs ;;
  *) usage; exit 1 ;;
esac
