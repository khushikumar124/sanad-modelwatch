# modelwatch-client

A thin HTTP client for reporting to a [ModelWatch](../modelwatch) server —
no dependency on ModelWatch's core engine, adapters, or dashboard. If your
app can send an HTTP request, it can report to ModelWatch.

## Install

From this repo (not yet published to PyPI):

```bash
pip install -e ./modelwatch-client
```

## Quickstart

```python
from modelwatch_client import ModelWatchClient

client = ModelWatchClient("http://localhost:8000")

# Register once, with a window of normal-operation data as the baseline.
if not client.is_registered("my-rag-app"):
    client.register_model(
        model_id="my-rag-app",
        name="My RAG App",
        adapter_name="rag",  # must be an adapter the server already knows
        baseline_data=baseline_events,
    )

# Report a new batch and act on the result.
result = client.check("my-rag-app", new_events)
if result["is_drifted"]:
    print("drifted:", result["signals"])

# Send a full per-request trace for the RAG X-Ray (optional).
client.record_trace(trace_id, "my-rag-app", trace_data)
```

`new_events` and `trace_data` need to match whatever shape the server's
registered adapter expects — this client doesn't validate or reshape
your data, it only forwards it. For the `rag` adapter and RAG X-Ray
traces specifically, see `sanad/features/trace.py` and
`sanad/api/telemetry.py` in the main repo for the exact shape Sanad
sends; anything producing the same shape works identically.

## What this is not

This is not a replacement for `modelwatch/examples/telemetry_reporter.py`
-- that script also knows *when* to poll and *how much* to batch for a
specific integration (Sanad). This client only knows how to make the
underlying HTTP calls; the polling/batching logic for your own app is
yours to write, the same way it already is in that reporter script.

## Errors

Every method raises `ModelWatchError` on a network failure or non-2xx
response. Check `.status_code` on the exception to distinguish "server
unreachable" (`None`) from an HTTP error code.

```python
from modelwatch_client import ModelWatchError

try:
    client.check("my-rag-app", new_events)
except ModelWatchError as e:
    if e.status_code == 404:
        print("model not registered yet")
    else:
        raise
```
