# Sanad

An AI-powered contract comprehension platform for Indian legal agreements
(rental, employment, freelance/service). Upload a contract as a PDF or
scanned image and get:

- **Summarizer** — a structured plain-language extraction: parties, key
  obligations, important dates/deadlines, notice period, penalty clauses,
  termination conditions.
- **Chatbot** — grounded Q&A over the uploaded document via RAG. Answers
  are restricted to the retrieved excerpts, cite which excerpt(s) they
  used, and explicitly refuse rather than guess when the document doesn't
  address the question.

Both features share one ingestion pipeline (extraction → chunking →
embedding → vector store) — see `rag/pipeline.py`.

Sanad's chatbot is monitored by [ModelWatch](../modelwatch) (a separate,
reusable framework) via its `LLMAdapter` — see
`../modelwatch/examples/sanad_golden_set_runner.py`.

## Architecture

```
 sample_docs/{rental,employment,freelance}/*.pdf
                    │
                    ▼
        ┌───────────────────────┐
        │ ingestion/              │
        │  extraction.py           │  PyMuPDF (native text) with
        │  chunking.py              │  Tesseract OCR fallback per page;
        └───────────┬───────────┘  clause/section-aware chunking
                    │
                    ▼
        ┌───────────────────────┐
        │ rag/                    │
        │  embeddings.py            │  sentence-transformers
        │  vector_store.py           │  (all-MiniLM-L6-v2)
        │  llm_client.py               │  ChromaDB, doc_id-scoped
        │  pipeline.py (ingest_document)│  LLMClient → OllamaClient
        └───────────┬───────────┘
                    │
       ┌────────────┴─────────────┐
┌──────▼──────────┐      ┌─────────▼─────────┐
│ features/          │      │ features/            │
│ summarizer.py        │      │ chatbot.py             │
│ (structured JSON       │      │ (retrieve → answer →     │
│  extraction prompt)      │      │  cite, or refuse)          │
└──────┬──────────┘      └─────────┬─────────┘
       └────────────┬─────────────┘
                    ▼
        ┌───────────────────────┐
        │ api/app.py (FastAPI)     │  /api/documents (upload/list/get/delete)
        │                            │  /api/documents/{id}/summarize
        │                            │  /api/documents/{id}/chat
        │                            │  /api/admin/model (demo model swap)
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │ frontend/index.html      │  upload → summary → chat, vanilla JS
        └───────────────────────┘
```

## Quickstart

**1. System dependencies (manual, one-time):**

```bash
brew install tesseract   # OCR fallback for scanned pages
```

**2. Ollama (manual, one-time) — not installed by this project:**

Install Ollama from [ollama.com](https://ollama.com) or `brew install
ollama`, then pull the default model and start the server:

```bash
ollama pull llama3.2:3b
ollama serve   # or just open the Ollama app
```

`llama3.2:3b` is the configured default — chosen for a 16GB-RAM laptop
with no discrete GPU (see `rag/llm_client.py` for the tradeoff notes). Set
`SANAD_OLLAMA_MODEL` to use a different one.

**3. Python environment:**

```bash
cd sanad
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest              # run the test suite
uvicorn sanad.api.app:app --port 8100
```

> Commands above assume you run them from the repository root (so
> `sanad.api.app` resolves as a package).

Open `http://localhost:8100/` — upload a contract from `sample_docs/`,
then generate a summary or ask it a question.

## Sample documents

Drop PDF/image files into `sample_docs/{rental,employment,freelance}/`.
Currently populated:

- `rental/` — 2 real Indian rental/lease agreement templates
- `freelance/` — 5 real Indian freelance/consultancy/service agreement templates
- `employment/` — **empty.** The only file initially dropped here turned
  out to be a compilation of real people's real signed offer letters
  (names, home addresses, salaries) rather than a blank template, and was
  excluded from the project entirely rather than processed. Add a proper
  blank employment offer-letter template here to fill this gap.

## Configuration

All via environment variables (see `config.py`):

| Variable                     | Default            | Meaning |
|-------------------------------|---------------------|---------|
| `SANAD_CHUNK_MAX_CHARS` / `SANAD_CHUNK_MIN_CHARS` | `1500` / `200` | Clause-aware chunker size bounds |
| `SANAD_OCR_LANGUAGE`          | `eng`               | Tesseract language pack |
| `SANAD_OCR_DPI`                | `300`               | Rasterization DPI before OCR |
| `SANAD_EMBEDDING_MODEL`        | `all-MiniLM-L6-v2`  | sentence-transformers model |
| `SANAD_CHROMA_DB_PATH`          | `sanad_chroma_db`   | ChromaDB persistence directory |
| `SANAD_RETRIEVAL_TOP_K`          | `4`                 | Chunks retrieved per chat question |
| `SANAD_API_HOST` / `SANAD_API_PORT` | `0.0.0.0` / `8100` | API bind address |
| `SANAD_UPLOAD_DIR`                | `sanad_uploads`     | Where uploaded files are saved |
| `SANAD_OLLAMA_BASE_URL`            | `http://localhost:11434` | Ollama server |
| `SANAD_OLLAMA_MODEL`                | `llama3.2:3b`       | Model Ollama is asked to run |

## Testing

```bash
pytest sanad/tests/ -v
```

The summarizer, chatbot, and OllamaClient error-handling are tested
against a `FakeLLMClient`/the real connection-refused path rather than a
live model (Ollama isn't assumed to be running in CI). The extraction,
chunking, embedding, and retrieval pipeline is tested against real sample
contracts, including a synthetic OCR test (a native-text page rasterized
to a flat image, to exercise the Tesseract fallback without needing an
actual scanned document).

## Known limitations

- **Local LLM quality.** `llama3.2:3b` is small and won't always reliably
  stay inside the JSON schema the summarizer/chatbot prompts require. Both
  features detect this (`parse_error=True`) and fail safe rather than
  return garbage, but that means you can occasionally get "couldn't
  produce a valid answer" even when the document does contain the answer
  — a model-following-instructions failure, not a grounding failure. A
  larger model (e.g. Mistral 7B) reduces this at the cost of speed/memory.
- **No multi-turn conversation.** Each chat question is answered
  independently; there's no conversation history, so a follow-up like
  "what about the deposit?" after a previous question has no context of
  what "the deposit" refers to.
- **Chunking is regex-heuristic, not a real document parser.** It handles
  clean numbered-clause structure well, but documents that mix numbering
  conventions (e.g. a clause that starts a new line with "1." but later
  text also references "Clause 3." inline) can produce an extra spurious
  chunk boundary mid-paragraph.
- **OCR quality depends on scan quality.** Tesseract (English pack only)
  can misread stamps, seals, handwriting, or the ₹ symbol on lower-quality
  scans; there's no confidence scoring or manual-correction flow.
- **In-memory document registry.** Uploaded document *metadata* (filename,
  contract type, full text used for summarization) lives in the API
  process's memory, not on disk. Restarting the server loses it — the
  underlying ChromaDB vectors persist to disk, but the API will 404 on the
  old `doc_id` until you re-upload.
- **No authentication, no multi-tenancy.** One shared ChromaDB collection
  filtered by `doc_id`; fine for a single-user local demo, not for
  multiple untrusted users sharing one deployment.
- **No response streaming.** Summarize/chat calls block until the local
  model finishes generating the full response — there's no token-by-token
  streaming to the frontend, so a slow model feels slow with no
  intermediate feedback.
- **PDF and common image formats only** — no `.docx`/`.txt` support; those
  need to be converted to PDF first.
- **ModelWatch's visibility into this chatbot is a lexical proxy.**
  `LLMAdapter`'s TF-IDF similarity (see ModelWatch's own README) measures
  vocabulary overlap with the golden set's expected answers, not semantic
  correctness — a correct paraphrase can score as more "drifted" than a
  wrong answer that happens to reuse the same words.
