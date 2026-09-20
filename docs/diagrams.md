# Diagrams

Every diagram below is drawn from the actual class/function names in this
repo (see the file path noted under each one), not a conceptual sketch --
if a name here stops matching the code, the diagram is stale and should be
fixed, not the other way around. Rendered natively by GitHub; view locally
with any Mermaid-compatible Markdown viewer if needed.

## 1. System architecture

Two independently-runnable services plus Ollama, wired together by one
optional bridge process. `shared/jobs.py` is used by both apps but neither
app depends on the other's package.

```mermaid
graph TB
    subgraph "User's machine"
        Browser["Browser"]
    end

    subgraph "Sanad (port 8100)"
        SanadAPI["sanad/api/app.py<br/>FastAPI"]
        Ingestion["sanad/ingestion/<br/>extraction + chunking"]
        RAG["sanad/rag/<br/>embeddings + retrieval + llm_client"]
        Features["sanad/features/<br/>risk_flagger, chatbot, obligations..."]
        SanadDB[("SQLite / Postgres<br/>document registry")]
        Chroma[("ChromaDB<br/>vector store")]
    end

    subgraph "ModelWatch (port 8000)"
        MWAPI["modelwatch/api/app.py<br/>FastAPI"]
        Engine["modelwatch/core/engine.py<br/>MonitoringEngine"]
        Adapters["modelwatch/adapters/<br/>RAG / LLM / Classifier"]
        MWDB[("SQLite<br/>models, runs, alerts")]
        Dashboard["modelwatch/dashboard/<br/>index.html"]
    end

    Reporter["modelwatch/examples/<br/>telemetry_reporter.py<br/>(optional bridge)"]
    Ollama["Ollama<br/>local LLM inference"]

    Browser -->|"upload / ask / risk scan"| SanadAPI
    Browser -->|"dashboard"| Dashboard
    Dashboard -->|"REST"| MWAPI
    SanadAPI --> Ingestion --> RAG
    SanadAPI --> Features
    RAG --> Ollama
    RAG --> Chroma
    SanadAPI --> SanadDB
    SanadAPI -->|"/api/telemetry"| Reporter
    Reporter -->|"POST /models/sanad-live/check"| MWAPI
    MWAPI --> Engine --> Adapters
    Engine --> MWDB

    Independent["examples/independent_rag/<br/>never imports sanad/"] -.->|"modelwatch-client SDK only"| MWAPI
```

## 2. Class diagram -- ModelWatch's adapter pattern

The one seam the monitoring engine is allowed to depend on
(`modelwatch/core/adapter_base.py`, `modelwatch/core/engine.py`,
`modelwatch/adapters/*.py`). `MonitoringEngine` never branches on model
type; every model-specific statistic lives in a concrete `ModelAdapter`.

```mermaid
classDiagram
    class ModelAdapter {
        <<abstract>>
        +str adapter_name
        +build_baseline(data) dict
        +check_drift(baseline, new_data) DriftCheckResult
    }
    class RAGAdapter {
        +adapter_name = "rag"
        +build_baseline(events) dict
        +check_drift(baseline, events) DriftCheckResult
    }
    class LLMAdapter {
        +adapter_name = "llm"
        +build_baseline(qa_pairs) dict
        +check_drift(baseline, qa_pairs) DriftCheckResult
    }
    class ClassifierAdapter {
        +adapter_name = "classifier"
        +build_baseline(features) dict
        +check_drift(baseline, features) DriftCheckResult
    }
    class LiveTelemetryAdapter {
        +adapter_name = "live_telemetry"
        +build_baseline(events) dict
        +check_drift(baseline, events) DriftCheckResult
    }
    ModelAdapter <|-- RAGAdapter
    ModelAdapter <|-- LLMAdapter
    ModelAdapter <|-- ClassifierAdapter
    ModelAdapter <|-- LiveTelemetryAdapter

    class DriftCheckResult {
        +float drift_score
        +float quality_score
        +bool is_drifted
        +List~SignalResult~ signals
        +dict statistics
        +to_dict() dict
    }
    class SignalResult {
        +str name
        +float value
        +bool is_drifted
        +dict detail
    }
    DriftCheckResult "1" *-- "many" SignalResult

    class MonitoringEngine {
        -Storage storage
        -dict~str,ModelAdapter~ adapters
        +register_model(model_id, name, adapter, baseline_data) dict
        +attach_adapter(model_id, adapter) void
        +run_check(model_id, new_data) dict
        +list_models() list
        +get_history(model_id) list
        +get_alerts(model_id) list
        +diagnose_run(run_id) DiagnosisResult
        +trigger_retrain(model_id, new_training_data) dict
    }
    MonitoringEngine "1" o-- "many" ModelAdapter : attaches
    MonitoringEngine ..> DriftCheckResult : produces
```

## 3. Class diagram -- Sanad's ingestion + risk-flagging data model

```mermaid
classDiagram
    class ExtractedDocument {
        +str text
        +List~PageExtraction~ pages
    }
    class PageExtraction {
        +int page_number
        +str text
        +bool used_ocr
    }
    ExtractedDocument "1" *-- "many" PageExtraction

    class Chunk {
        +int index
        +str text
        +str heading
    }

    class RiskRule {
        +str severity
        +Pattern[] patterns
        +Tuple~RiskFactor~ factors
    }
    class RiskFactor {
        +str id
        +str description
        +str disadvantages
    }
    class FactorResult {
        +str factor_id
        +bool matched
    }
    class RiskFinding {
        +str rule_id
        +str label
        +str severity
        +str explanation
        +str affects
        +int chunk_index
        +str excerpt
        +List~FactorResult~ factors
    }
    class RiskReport {
        +List~RiskFinding~ findings
        +to_dict() dict
    }
    RiskRule "1" *-- "many" RiskFactor
    RiskFinding "1" *-- "many" FactorResult
    RiskReport "1" *-- "many" RiskFinding

    class LLMClient {
        <<abstract>>
        +generate(system_prompt, user_prompt, response_schema) str
    }
    class OllamaClient {
        +str model
        +str keep_alive
        +generate(...) str
    }
    LLMClient <|-- OllamaClient

    class Embedder {
        +str model_name
        +embed(texts) ndarray
    }

    ExtractedDocument --> Chunk : chunk_document()
    Chunk --> RiskFinding : flag_risks()
```

## 4. Sequence diagram -- Sanad: upload a contract, ask a grounded question

```mermaid
sequenceDiagram
    actor User
    participant UI as frontend/index.html
    participant API as sanad/api/app.py
    participant Extract as ingestion/extraction.py
    participant Chunk as ingestion/chunking.py
    participant Embed as rag/embeddings.py
    participant Store as rag/vector_store.py (ChromaDB)
    participant Fusion as rag/hybrid_retrieval.py
    participant LLM as rag/llm_client.py (Ollama)

    User->>UI: upload contract.pdf
    UI->>API: POST /api/documents
    API->>Extract: extract_document(path)
    Extract-->>API: ExtractedDocument (OCR fallback if no text layer)
    API->>Chunk: chunk_document(text)
    Chunk-->>API: List[Chunk]
    API->>Embed: embed(chunk texts)
    Embed-->>API: vectors
    API->>Store: upsert(doc_id, vectors, chunks)
    API-->>UI: doc_id, chunk_count

    User->>UI: "What is the security deposit?"
    UI->>API: POST /api/documents/{id}/chat
    API->>Store: query(doc_id, question)
    Store->>Store: _dense_query() (embedding similarity)
    Store->>Fusion: reciprocal_rank_fusion(dense + BM25 rankings)
    Fusion-->>Store: fused top-k chunks
    Store-->>API: top-k chunks
    API->>LLM: generate(system_prompt, question + chunks)
    LLM-->>API: answer + citation numbers
    API->>API: verify citations against retrieved text
    alt no valid citation
        API-->>UI: refusal ("I don't know")
    else grounded
        API-->>UI: answer + cited excerpt + RAG trace
    end
```

## 5. Sequence diagram -- ModelWatch: live drift detection, alert, and recovery

The actual flow exercised by `modelwatch.examples.simulate_drift_demo` and
documented with real measured numbers in `DEMO.md`.

```mermaid
sequenceDiagram
    participant Sanad as Sanad chatbot
    participant Reporter as telemetry_reporter.py
    participant Engine as MonitoringEngine
    participant Adapter as RAGAdapter / LLMAdapter
    participant DB as ModelWatch DB
    actor Dashboard

    Note over Sanad,DB: Baseline: healthy traffic
    Sanad->>Reporter: per-request telemetry (grounded, citations, latency)
    Reporter->>Engine: POST /models/sanad-live/check
    Engine->>Adapter: check_drift(baseline, new_events)
    Adapter-->>Engine: DriftCheckResult(is_drifted=False, score~0.33)
    Engine->>DB: persist run
    Dashboard->>Engine: GET /models/sanad-live/history
    Engine-->>Dashboard: "Healthy"

    Note over Sanad,DB: Operator swaps the underlying LLM (drift injected)
    Sanad->>Reporter: telemetry from degraded model (more refusals)
    Reporter->>Engine: POST /models/sanad-live/check
    Engine->>Adapter: check_drift(baseline, new_events)
    Adapter-->>Engine: DriftCheckResult(is_drifted=True, score=0.78)
    Engine->>DB: persist run + open alert
    Dashboard->>Engine: GET /alerts?active_only=true
    Engine-->>Dashboard: "Degraded -- drift detected"

    Note over Sanad,DB: Corrective action: swap model back, retrain
    Dashboard->>Engine: POST /models/sanad-live/retrain
    Engine->>Adapter: build_baseline(fresh good-traffic data)
    Engine->>DB: new baseline version, resolve alert
    Sanad->>Reporter: telemetry from restored model
    Reporter->>Engine: POST /models/sanad-live/check
    Engine-->>DB: DriftCheckResult(is_drifted=False) -- recovered
```
