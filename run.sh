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

SANAD_PORT=${SANAD_API_PORT:-8100}
MW_PORT=${MODELWATCH_API_PORT:-8000}

stop_all() {
  lsof -ti:"$SANAD_PORT" 2>/dev/null | xargs kill -9 2>/dev/null
  lsof -ti:"$MW_PORT" 2>/dev/null | xargs kill -9 2>/dev/null
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

MODEL=$(curl -s --max-time 5 "http://localhost:$SANAD_PORT/api/admin/model" 2>/dev/null | sed 's/.*"model":"\([^"]*\)".*/\1/')
HAVE=$(ollama list 2>/dev/null | awk 'NR>1{print $1}' | tr '\n' ' ')

echo
echo "  Sanad      http://localhost:$SANAD_PORT/"
echo "  ModelWatch http://localhost:$MW_PORT/dashboard/"
echo
echo "  active model: ${MODEL:-unknown}"
echo "  installed:    ${HAVE:-none}"
case " $HAVE " in
  *" ${MODEL} "*) ;;
  *) echo "  WARNING: '$MODEL' is not installed -- run: ollama pull $MODEL" ;;
esac
echo
echo "  logs: /tmp/sanad_api.log  /tmp/modelwatch_api.log"
echo "  stop: ./run.sh --stop"
