# Telemetry schema

Every answered question in Sanad appends one event to an in-memory,
bounded buffer (`sanad/api/telemetry.py`), polled by
`modelwatch/examples/telemetry_reporter.py` and forwarded to ModelWatch.
Sanad never pushes anywhere and holds no reference to what reads the
buffer -- see [architecture.md](architecture.md).

## Fields

```python
@dataclass(frozen=True)
class ChatEvent:
    at: str
    grounded: bool
    citations: int
    latency_ms: float
    parse_error: bool
    retrieved: int

    # additive fields, defaulted so old call sites still work
    trace_id: str = ""
    doc_id: str = ""
    model_name: str = ""
    top_k: int = 0
    retrieval_scores: list[float] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    citations_requested: int = 0
```

The first five fields are the original schema; `LiveTelemetryAdapter`
still reads them by name and nothing about them changed. Everything below
`trace_id` was added to support real statistical drift detection
(`modelwatch/adapters/rag_adapter.py`) instead of aggregate-rate-only
monitoring.

## What's deliberately NOT here

No question text, no answer text, no chunk text, no document filename.
Contracts are confidential; a monitoring buffer is not where their
contents belong. `doc_id` is kept because Sanad assigns it as an opaque
identifier at upload time -- it lets a reader ask "is this concentrated on
one document?" without exposing what the document says.
`retrieval_scores` are cosine distances (bounded floats), never the
retrieved text.

## Where each field comes from

- `retrieval_scores`: `[c["distance"] for c in answer.retrieved_chunks]`
  in `sanad/api/app.py`'s chat endpoint -- straight from ChromaDB's query
  result.
- `retrieval_latency_ms` / `generation_latency_ms`: timed separately
  inside `sanad/features/chatbot.py::ask()`, around the vector-store
  query and the LLM call respectively, so a latency regression can be
  attributed to one stage rather than reported as one lump sum.
- `citations_requested`: the raw count of excerpt numbers the model named
  in `cited_excerpts`, before filtering to ones that actually exist.
  `citations / citations_requested` is a citation-validity ratio -- how
  often a citation the model claimed to make actually pointed at a real
  retrieved excerpt.

## Backward compatibility

`sanad.api.telemetry.record_chat()` accepts the new fields as optional
keyword arguments with safe defaults. A caller passing only the original
five arguments (as pre-existing code did) still works unchanged --
verified in `sanad/tests/test_telemetry.py::test_record_chat_with_only_original_fields_still_works`.
