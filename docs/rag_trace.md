# Making the AI visible: RAG Trace, Claim Verification, RAG X-Ray

## Sanad's side: the RAG trace

Every answered question in Sanad now returns a `trace` alongside the
answer (`sanad/features/trace.py`, wired into `POST /api/documents/{id}/chat`).
It reconstructs the observable pipeline:

```
question -> embedding -> retrieval (ranked, scored) -> evidence
  -> generation -> claim-level verification -> grounding/citation scores
```

There is no hidden chain-of-thought here, by design: everything in the
trace is either a direct fact about the pipeline (which chunks were
retrieved, at what cosine similarity, which were cited) or a computed,
reproducible check (does this sentence of the answer overlap enough with
retrieved evidence to call it supported). An LLM asked to explain its
own reasoning is not a reliable source about that reasoning, so Sanad
doesn't ask it to.

**Retrieval Inspector**: every retrieved chunk, ranked, with its
similarity score and whether it was cited -- shown in Sanad's chat UI
under "AI / RAG trace" for every answer.

**Claim verification**: the answer is split into sentences; each is
scored (embedding cosine similarity) against the retrieved/cited
evidence, at the *sentence* level on both sides (a claim compared
against a whole multi-paragraph chunk is unfairly diluted by unrelated
surrounding text -- see `_evidence_sentence_pool` in trace.py, added
after a real test on real data showed a near-verbatim echo of a clause
scoring only "partial" until this fix). Labelled supported / partial /
unsupported against uncalibrated thresholds (0.55 / 0.35) -- a triage
aid, not a certified fact-check.

## The privacy tradeoff: full traces now leave Sanad

Sanad's telemetry ([telemetry.md](telemetry.md)) was originally
built to carry zero question/answer/chunk text. Making ModelWatch's RAG
X-Ray meaningful requires exactly that content, so
`SANAD_TELEMETRY_FULL_TRACE` (default **on**) now rides the full trace
along in each telemetry event. This is a real, deliberate reversal of
that original privacy stance, not a free feature -- turn it off
(`SANAD_TELEMETRY_FULL_TRACE=false`) for any deployment where the
monitor must never see contract content. With it off, ModelWatch falls
back to exactly what it had before: aggregate rates and statistical
signals, no text.

Because `/api/telemetry` can now carry real content, it's worth
remembering it already sits behind `require_user` when Sanad's auth is
on -- and now the telemetry *reporter itself* needs to authenticate too
(`SANAD_REPORTER_USERNAME`/`SANAD_REPORTER_PASSWORD`; see
`modelwatch/examples/telemetry_reporter.py::login_if_needed`). This was
a real bug caught while building this feature: turning Sanad's auth on
silently locked the reporter out with repeated 401s until this was
added.

## ModelWatch's side: the RAG X-Ray

`modelwatch/core/storage.py`'s `traces` table stores each full trace,
independent of the drift-check/adapter machinery (a trace is one
request's raw detail, not a statistical sample). The reporter forwards
every trace on every poll -- decoupled from `--min-batch`, which only
gates the drift-check batch, so the X-Ray doesn't sit empty waiting for
5 accumulated questions the way the drift signal correctly does.

`GET /traces`, `GET /traces/{id}`, `GET /traces/{id}/diagnosis` back a
"RAG X-Ray" panel in the dashboard: browse recent requests (filterable
by grounded/refused), click one to see the full pipeline plus a
per-request diagnosis.

## Per-trace diagnosis (`modelwatch/diagnosis/trace_diagnosis.py`)

Different question from `modelwatch/diagnosis/engine.py`'s batch-level
subsystem ranking: given **one** trace, why does this answer look the
way it does?

| category | when |
|---|---|
| `retrieval_miss` | nothing was retrieved at all |
| `irrelevant_retrieval` | refused; best match similarity < 0.30 |
| `insufficient_evidence` | refused; best match similarity 0.30-0.50 |
| `generation_problem` | refused despite good evidence (>=0.50), or grounded but claims/grounding score don't hold up |
| `citation_problem` | grounded but citation_score < 1.0 |
| `none` | retrieval, citations, and claims all consistent |

A single trace has no distribution to test, so there is no p-value and
none is invented -- `evidence` carries the actual numbers the rule
looked at (best similarity, citation/grounding scores, latency), not a
fabricated confidence. One documented gap: "query problem" (the
question itself was malformed or oddly embedded) is folded into
`irrelevant_retrieval` rather than separated out, because telling them
apart needs a reference distribution of normal queries to compare
against -- exactly what RAGAdapter's retrieval signal does at the batch
level, and exactly what a single trace doesn't have.

Verified against real traffic (2026-08-30, `phi3:3.8b`): a genuine
"When is the monthly rent due?" answer came back grounded, correctly
cited, and diagnosed `none`; a genuine "What brand of furniture is in
the property?" (not in the document) refused with best similarity 0.30
and was correctly diagnosed `irrelevant_retrieval`.
