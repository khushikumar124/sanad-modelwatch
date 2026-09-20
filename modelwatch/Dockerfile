# Build from the repository root: docker build -f modelwatch/Dockerfile .
# (or `docker compose up` from the root -- see ../docker-compose.yml).
# Needs the root as build context so `modelwatch.*` and `shared.*`
# resolve as top-level packages, same reasoning as sanad/Dockerfile.
#
# Deliberately does NOT copy sanad/ -- modelwatch/api/app.py used to
# import JobManager from sanad.jobs, which meant this image would have
# needed the whole Sanad package (and its much heavier dependency list:
# torch-adjacent sentence-transformers, chromadb, PyMuPDF) just to start
# its own API. That import now points at shared/jobs.py instead (a
# stdlib-only module both apps depend on, so neither depends on the
# other), so this image only needs modelwatch/ and shared/.
FROM python:3.12-slim

# curl is only for the container healthcheck below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY modelwatch/requirements.txt modelwatch/requirements.txt
RUN pip install --no-cache-dir -r modelwatch/requirements.txt

COPY shared/ shared/
COPY modelwatch/ modelwatch/

# modelwatch.db (models/baselines/runs/alerts/experiments/traces) lives
# under /app/data by default -- docker-compose.yml mounts a named volume
# there so history survives a restart.
VOLUME ["/app/data"]

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD curl -f http://localhost:8000/dashboard/ || exit 1

CMD ["uvicorn", "modelwatch.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
