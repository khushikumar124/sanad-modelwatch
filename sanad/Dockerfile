# Build from the repository root: docker build -f sanad/Dockerfile .
# (or, simpler, `docker compose up` from the root, which sets this
# context automatically -- see ../docker-compose.yml). The build context
# has to be the root, not sanad/, so `sanad.*` and `shared.*` resolve as
# top-level packages exactly the way they do when running
# `uvicorn sanad.api.app:app` locally from the repo root (see the root
# README's quickstart).
FROM python:3.12-slim

# Tesseract is the OCR fallback (sanad/ingestion/extraction.py shells out
# to it for pages with no text layer) -- everything else (PyMuPDF,
# sentence-transformers, chromadb) ships its own bundled binaries as
# Python wheels. curl is only for the container healthcheck below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY sanad/requirements.txt sanad/requirements.txt
RUN pip install --no-cache-dir -r sanad/requirements.txt

# Bakes the embedding model into the image at build time rather than
# downloading it from HuggingFace on first use -- caught by an actual
# smoke test (docker compose up + a real upload), which showed every
# fresh container re-fetching all-MiniLM-L6-v2 over the network on its
# first request, and losing that cache again on every restart since
# nothing was persisting ~/.cache/huggingface. A container meant to be
# handed to a client/dev team shouldn't need to reach the internet just
# to embed a chunk, and shouldn't redo a fetch it already did.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Baking the model in isn't enough on its own -- sentence-transformers
# still does a HEAD request per file to check the Hub for a newer etag
# on every load, caught by the same smoke test (34 huggingface.co HEAD
# requests logged on the very next upload, right after the "offline"
# fix above). HF_HUB_OFFLINE skips that check entirely and uses the
# baked-in cache directly, since this image was built with a specific,
# known-good model version -- there's nothing to check against at
# runtime that build time didn't already settle.
ENV HF_HUB_OFFLINE=1

COPY shared/ shared/
COPY sanad/ sanad/

# Uploaded files, the SQLite document registry, and the Chroma vector
# store all default to paths under /app/data -- docker-compose.yml mounts
# a named volume there so they survive a container restart instead of an
# image rebuild wiping every document a demo uploaded.
VOLUME ["/app/data"]

EXPOSE 8100

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=5 \
    CMD curl -f http://localhost:8100/ || exit 1

# --host 0.0.0.0 so other containers (and the host, via the published
# port) can reach it -- config.api_host's 127.0.0.1 default is a
# same-machine-only safeguard for the non-Docker ./run.sh path, and is
# irrelevant here since uvicorn's own --host flag is what actually binds.
CMD ["uvicorn", "sanad.api.app:app", "--host", "0.0.0.0", "--port", "8100"]
