#!/bin/bash
# Start Ollama, ModelWatch and Sanad together, wait until all three answer,
# and print the URLs. Safe to re-run: it frees the ports first, which also
# clears the stale-data-directory state that makes uploads 500 after the
# working tree has been cleaned underneath a running server.
#
#   ./run.sh          start everything
#   ./run.sh --stop   stop everything
set -u
cd "$(dirname "$0")"

# Local overrides (auth, session secret, etc.) persisted across shells --
# see .env's own comment. Gitignored; safe to be absent.
[ -f .env ] && source .env

SANAD_PORT=${SANAD_API_PORT:-8100}
MW_PORT=${MODELWATCH_API_PORT:-8000}

stop_all() {
  lsof -ti:"$SANAD_PORT" 2>/dev/null | xargs kill -9 2>/dev/null
  lsof -ti:"$MW_PORT" 2>/dev/null | xargs kill -9 2>/dev/null
  pkill -f "modelwatch.examples.telemetry_reporter" 2>/dev/null
}

if [ "${1:-}" = "--stop" ]; then
  stop_all; echo "stopped Sanad and ModelWatch (Ollama left running)"; exit 0
fi

if [ ! -d .venv ]; then
  echo "no .venv found. Create it first:"
  echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -r sanad/requirements.txt -r modelwatch/requirements.txt"
  exit 1
fi
source .venv/bin/activate

# Ollama: start only if it isn't already answering.
if ! curl -s -o /dev/null --max-time 3 http://localhost:11434/api/tags 2>/dev/null; then
  echo "starting ollama..."
  nohup ollama serve > /tmp/ollama.log 2>&1 &
  for _ in $(seq 1 30); do
    curl -s -o /dev/null --max-time 2 http://localhost:11434/api/tags 2>/dev/null && break
    sleep 1
  done
fi
curl -s -o /dev/null --max-time 3 http://localhost:11434/api/tags 2>/dev/null \
  || { echo "ollama did not start -- is it installed? (brew install ollama)"; exit 1; }

stop_all; sleep 1
echo "starting modelwatch on :$MW_PORT ..."
nohup uvicorn modelwatch.api.app:app --host 0.0.0.0 --port "$MW_PORT" > /tmp/modelwatch_api.log 2>&1 &
echo "starting sanad on :$SANAD_PORT ..."
nohup uvicorn sanad.api.app:app --host 0.0.0.0 --port "$SANAD_PORT" > /tmp/sanad_api.log 2>&1 &

for _ in $(seq 1 60); do
  curl -s -o /dev/null "http://localhost:$MW_PORT/models" 2>/dev/null \
    && curl -s -o /dev/null "http://localhost:$SANAD_PORT/api/documents" 2>/dev/null && break
  sleep 1
done

# Forward live usage to ModelWatch. Without this nothing polls Sanad's
# telemetry buffer, so asking questions in the app leaves the dashboard
# unchanged -- which reads as the monitor being broken.
nohup python -u -m modelwatch.examples.telemetry_reporter --interval 15 --min-batch 5 \
  > /tmp/telemetry_reporter.log 2>&1 &

AUTH=$(curl -s --max-time 5 "http://localhost:$SANAD_PORT/api/auth/session" 2>/dev/null | grep -c '"auth_enabled":true')
if [ "${AUTH:-0}" = "1" ]; then
  # /api/admin/model is behind require_user, so it 401s with no session
  # cookie -- read the configured default straight from Sanad's own
  # config instead of hitting a protected endpoint.
  MODEL=$(python -c "from sanad.config import config; print(config.ollama_model)" 2>/dev/null)
else
  MODEL=$(curl -s --max-time 5 "http://localhost:$SANAD_PORT/api/admin/model" 2>/dev/null | sed 's/.*"model":"\([^"]*\)".*/\1/')
fi
HAVE=$(ollama list 2>/dev/null | awk 'NR>1{print $1}' | tr '\n' ' ')

echo
echo "  Sanad      http://localhost:$SANAD_PORT/"
echo "  ModelWatch http://localhost:$MW_PORT/dashboard/"
echo
if [ "${AUTH:-0}" = "1" ]; then
  echo "  auth:         ON (sign in required)"
else
  echo "  auth:         off  --  enable with: python -m sanad.create_user <name>"
fi
echo "  active model: ${MODEL:-unknown}"
echo "  installed:    ${HAVE:-none}"
case " $HAVE " in
  *" ${MODEL} "*) ;;
  *) echo "  WARNING: '$MODEL' is not installed -- run: ollama pull $MODEL" ;;
esac
echo
echo "  telemetry:    reporting live usage every 20s -> sanad-live"
echo
echo "  logs: /tmp/sanad_api.log  /tmp/modelwatch_api.log  /tmp/telemetry_reporter.log"
echo "  stop: ./run.sh --stop"
