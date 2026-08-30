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
- **Clause risk scan** — flags clauses that are unusual or work against the
  weaker party, each quoting the clause it came from. This is the
  proactive counterpart to the chatbot: it surfaces what you didn't know
  to ask about, which is the actual problem a non-lawyer has with a
  contract. Detection is rule-based, not LLM-judged — see
  `features/risk_flagger.py` for why.

Built on top of those, a **contract intelligence** layer (`features/`):

- **Obligation extraction** (`obligations.py`) — who owes what to whom,
  with a deadline where the document states one. Every extracted
  obligation is grounded back to the chunk it came from (exact match,
  falling back to content-word overlap for paraphrase) — an obligation
  the grounding check can't locate in the document is dropped rather
  than shown as fact.
- **Coverage scan** (`coverage.py`) — checks the document against 9
  standard clause categories (termination, notice period,
  confidentiality, IP ownership, dispute resolution, liability, payment,
  renewal, governing law) and reports which ones this rule-based scan
  didn't find. `not_found` never means "missing" — only that nothing
  matched these patterns.
- **Contradiction detection** (`contradictions.py`) — flags conflicting
  duration statements (e.g. two different notice periods), scoped to the
  categories where a numeric mismatch is unambiguous rather than
  guessed.
- **Review synthesis** (`review.py`) — combines risk findings, coverage
  gaps, and contradictions into one report, with a suggested negotiation
  question per flagged risk. Pure synthesis of already-computed results,
  no additional LLM call.
- **Risk heatmap + click-to-source** (`frontend/index.html`) — every
  clause in the document is addressable by index (`/api/documents/{id}/clauses`),
  so a risk/coverage/obligation finding can jump straight to the exact
  clause it's about, and the heatmap colors the whole document by
  per-clause severity.
- **Document comparison** (`comparison.py`) — diffs the risk profile of
  two uploaded documents (e.g. two drafts of the same contract).

All features share one ingestion pipeline (extraction → chunking →
embedding → vector store) — see `rag/pipeline.py` — and one document
registry (`db.py`, SQLite by default, Postgres via `SANAD_DATABASE_URL`)
and object store (`storage.py`, local disk by default, S3-compatible via
`SANAD_STORAGE_BACKEND=s3`) for the underlying data.

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
ollama pull phi3:3.8b
ollama serve   # or just open the Ollama app
```

`phi3:3.8b` is the configured default — sized for a 16GB-RAM laptop with
no discrete GPU, and measurably better than `llama3.2:3b` on this task
(0.536 vs 0.458 average golden-set answer similarity; see known
limitations). Set `SANAD_OLLAMA_MODEL` to use a different one.

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

## Authentication

Off by default, so the tests and a local demo need no credentials. To turn
it on:

```bash
python -m sanad.create_user khushi
```

That prompts for a password and prints three variables to export
(`SANAD_AUTH_ENABLED`, `SANAD_SESSION_SECRET`, `SANAD_USERS`). Export them,
restart, and the app shows a sign-in screen.

Design notes, since "we added login" is not on its own a security claim:

- Passwords are stored as PBKDF2-HMAC-SHA256 (240k rounds) with a random
  per-user salt. Never in the repository — credentials live in the
  environment.
- Sessions are HMAC-signed cookies, `HttpOnly` (so a script cannot read
  them) and `SameSite=Lax` (so they are not sent on cross-site POSTs),
  expiring after 12 hours.
- Login failures return one message for both an unknown username and a
  wrong password, and hash anyway when the user does not exist, so neither
  the message nor the timing reveals which usernames are real.
- With auth on and no session secret set, startup fails loudly instead of
  signing sessions with a throwaway key that would log everyone out on
  each restart.
- **Both servers bind `127.0.0.1` by default.** They previously bound
  `0.0.0.0`, which exposed uploaded contracts to everyone on the local
  network with no authentication at all — a worse hole than the missing
  login screen.

There is no OAuth. It would put an internet dependency and a third party
in front of an app whose entire premise is that the document never leaves
your machine. `sanad/api/auth.py` is structured so a provider can be added
later: a session records *who* you are, not *how* you proved it, so an
OAuth callback can mint the same session without touching route protection.

## Sample documents

Drop PDF/image files into `sample_docs/{rental,employment,freelance}/`.
Currently populated:

- `rental/` — 2 real Indian rental/lease agreement templates
- `freelance/` — 5 real Indian freelance/consultancy/service agreement templates
- `employment/` — an anonymised compilation of 16 Indian offer letters.
  The file originally dropped here held real candidates' names, home
  addresses and salaries, so it was rebuilt with dummy personal details
  before being used: company names, job titles, dates and salary figures
  are kept, since those aren't personal to an individual. Verified before
  inclusion — no real name, street address, phone, Aadhaar or PAN
  survives. Because it holds many letters rather than one contract,
  golden-set questions for it are scoped by employer.

## Configuration

All via environment variables (see `config.py`):

| Variable                     | Default            | Meaning |
|-------------------------------|---------------------|---------|
| `SANAD_CHUNK_MAX_CHARS` / `SANAD_CHUNK_MIN_CHARS` | `1500` / `200` | Clause-aware chunker size bounds |
| `SANAD_OCR_LANGUAGE`          | `eng`               | Tesseract language pack |
| `SANAD_OCR_DPI`                | `300`               | Rasterization DPI before OCR |
| `SANAD_EMBEDDING_MODEL`        | `all-MiniLM-L6-v2`  | sentence-transformers model |
| `SANAD_CHROMA_DB_PATH`          | `sanad_chroma_db`   | ChromaDB persistence directory |
| `SANAD_RETRIEVAL_TOP_K`          | `6`                 | Chunks retrieved per chat question (4 was measurably too few — see limitations) |
| `SANAD_API_HOST` / `SANAD_API_PORT` | `127.0.0.1` / `8100` | Bind address; loopback so uploads aren't network-reachable |
| `SANAD_UPLOAD_DIR`                | `sanad_uploads`     | Where uploaded files are saved |
| `SANAD_OLLAMA_BASE_URL`            | `http://localhost:11434` | Ollama server |
| `SANAD_OLLAMA_MODEL`                | `phi3:3.8b`         | Model Ollama is asked to run |
| `SANAD_AUTH_ENABLED`                 | `false`             | Require sign-in on every API route |
| `SANAD_SESSION_SECRET`                | *(unset)*           | HMAC key for session cookies; required when auth is on |
| `SANAD_USERS`                          | *(unset)*           | `name:hash` pairs, comma-separated (see `create_user.py`) |
| `SANAD_SESSION_TTL_SECONDS`             | `43200`             | Session lifetime (12 hours) |
| `SANAD_SESSION_COOKIE_SECURE`            | `false`             | Set true when serving over HTTPS |

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

- **The risk scan is pattern matching, not legal analysis.** It fires on
  explicit contract language drawn from patterns common in Indian rental,
  employment and freelance agreements. It cannot catch a harmful term
  phrased in wording no rule anticipates, it has no notion of context (a
  flagged clause may be entirely reasonable), and it is not legal advice.
  It is deliberately conservative for that reason, and every finding
  quotes the clause so a reader can judge it themselves. Being
  deterministic, it is also testable against known inputs — which an
  LLM-judged version would not be.

- **The chatbot still over-refuses**, though less than it used to, and
  the causes turned out to be mixed. On a 7-question probe where every
  answer is provably in the document:

  | | `retrieval_top_k=4` | `top_k=6` (current) |
  |---|---|---|
  | `llama3.2:3b` | 4/7 | 4/7 |
  | `phi3:3.8b` | 4/7 | **5/7** |

  Two distinct failure causes were found by checking retrieval directly
  rather than assuming:

  1. **Retrieval misses.** "What is the term of this lease?" failed on
     both models because the clause containing "11 Months" ranked *6th*
     by cosine distance, so a top-4 window never showed it to the model.
     The model refused correctly — it was never given the answer. Raising
     `SANAD_RETRIEVAL_TOP_K` to 6 fixed this specific question. The root
     cause is that the clause sits inside a large merged chunk covering
     several topics, which dilutes its embedding; better chunk sizing
     would be the deeper fix.
  2. **Model misses.** "Does this agreement create an employer-employee
     relationship?" fails even though retrieval puts the answering clause
     ("8. Nature of Relationship") at rank 1. That one is the model not
     using what it was given.

  An earlier version of this section claimed retrieval had been verified
  correct for all these failures. That was wrong: it had been checked on
  one question and generalised. Case 1 above is the counter-example.

  The design errs deliberately toward refusing: an answer with no valid
  citation is downgraded to a refusal (see `features/chatbot.py`), so the
  system fails toward "I don't know" rather than confident invention.

  The 7-question probe is too small to separate the two models on its own
  — 5/7 vs 4/7 is a single question. On the full 18-pair golden set the
  gap is clearer: **phi3:3.8b at top_k=6 scores 0.536 average answer
  similarity, versus 0.458 for llama3.2:3b at top_k=4**, which is why
  phi3 is now the default. (That comparison changes both the model and
  the retrieval window together — it measures the shipped configuration,
  not the model in isolation.)

- **JSON compliance is enforced, not requested.** Both features send a
  JSON Schema via Ollama's `format` parameter so output is constrained
  during sampling. Before this, `llama3.2:3b` would intermittently emit
  invalid JSON (observed: an unquoted string value for `answer`) and a
  correct answer would be thrown away by the parser. The defensive
  parsing and `parse_error` flag remain as a backstop for backends that
  can't enforce a schema.
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
- **Document metadata lives in a real database** (`db.py`, SQLAlchemy Core
  over SQLite by default, or Postgres via `SANAD_DATABASE_URL`) and the
  original uploaded file in a real object store (`storage.py`, local disk
  by default, or an S3-compatible bucket via `SANAD_STORAGE_BACKEND=s3`).
  Both are still single-tenant: one shared ChromaDB collection filtered by
  `doc_id`, no row/object-level access control.
- **Authentication exists but is minimal.** Session-cookie login
  (`SANAD_AUTH_ENABLED=true`) gates every route behind one shared set of
  users (`SANAD_USERS`) — enough to keep a deployment off the open
  internet, not multi-tenancy or per-user document isolation.
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
  wrong answer that happens to reuse the same words. Note the measured
  baseline of 0.536 sits fairly close to the default drift threshold of
  0.35, so there is limited headroom before normal variation trips an
  alert; raise `MODELWATCH_LLM_SIMILARITY_THRESHOLD` or improve the
  chatbot's answer rate before reading much into small movements.
