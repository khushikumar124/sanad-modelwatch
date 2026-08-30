# Independent RAG example

Proves the model-agnostic claim concretely: this monitors a RAG
pipeline that has **never imported anything from `sanad/`**, using only
`modelwatch-client`'s public SDK and the same `"rag"` adapter Sanad's
own integration uses. Swap `rag_pipeline.py` for your own retrieval +
generation code and the rest of this example (registration, checking,
reading signals) stays identical.

## What's actually in here

- `rag_pipeline.py` — a real (not mocked) but deliberately simple RAG
  pipeline: TF-IDF retrieval (scikit-learn) over a small hardcoded
  product-FAQ corpus, and a template-based "generator" that returns the
  top match's answer, or refuses below a similarity floor. No LLM call,
  no API key, no network dependency of its own.
- `run_example.py` — registers this pipeline with a running ModelWatch
  server, runs a baseline batch and a "current" batch (with several
  genuinely off-topic questions mixed in), and checks for drift.

## Running it

```bash
# from the repo root
./run.sh                                    # starts ModelWatch on :8000
pip install -e ./modelwatch-client scikit-learn
python examples/independent_rag/run_example.py
```

Point at a different ModelWatch instance with `MODELWATCH_URL`.

## What you should see

The baseline batch grounds on every question (it's drawn from the
FAQ's own phrasing). The current batch mixes in real off-topic
questions ("What's the meaning of life?", "Tell me a joke.") that this
pipeline correctly refuses -- and `client.check()` correctly reports
`is_drifted: True`, with the `refusal` and `citation_validity` signals
flagged. This isn't a staged number: it's the same RAGAdapter
statistical tests (two-proportion z-test) Sanad's own live traffic
runs through, reacting to this pipeline's real behavior.

The `embedding` signal always reports `0.0`/not drifted here, honestly:
this pipeline uses TF-IDF, not dense embeddings, so it has no vectors
to feed `embedding_drift`'s Maximum Mean Discrepancy test. A pipeline
that does compute embeddings can pass them along the same way Sanad
does (see `sanad/api/telemetry.py`'s `question_embedding` field) to get
that signal too.

## A real limitation worth knowing about

This pipeline strips English stopwords before matching (`TfidfVectorizer
(stop_words="english")`) specifically because the naive version doesn't:
without it, "What is the capital of France?" scored a deceptively high
0.565 similarity against an FAQ entry, purely from sharing the words
"what is the" -- a genuine, measured failure mode of naive lexical
retrieval, not a hypothetical one. Even with stopwords removed, this
pipeline still can't match a real paraphrase with no shared vocabulary
("How fast is delivery?" vs. the FAQ's "How long does shipping take?")
-- the same lexical-vs-semantic limitation documented for ModelWatch's
own `LLMAdapter` (see the root `modelwatch/README.md`). A production
pipeline would use real embeddings for retrieval instead; this example
keeps it to TF-IDF so it needs no extra ML dependency to run.
