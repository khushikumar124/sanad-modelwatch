"""Maps the adapter_name string used over the wire (REST payloads, storage
rows) to the concrete ModelAdapter class. Adding a new model type means
adding one entry here -- the API and engine code never change.
"""
from __future__ import annotations

from modelwatch.adapters.classifier_adapter import ClassifierAdapter
from modelwatch.adapters.live_telemetry_adapter import LiveTelemetryAdapter
from modelwatch.adapters.llm_adapter import LLMAdapter
from modelwatch.adapters.rag_adapter import RAGAdapter
from modelwatch.core.adapter_base import ModelAdapter

ADAPTER_REGISTRY: dict[str, type[ModelAdapter]] = {
    ClassifierAdapter.adapter_name: ClassifierAdapter,
    LLMAdapter.adapter_name: LLMAdapter,
    LiveTelemetryAdapter.adapter_name: LiveTelemetryAdapter,
    RAGAdapter.adapter_name: RAGAdapter,
}


def build_adapter(adapter_name: str) -> ModelAdapter:
    try:
        adapter_cls = ADAPTER_REGISTRY[adapter_name]
    except KeyError:
        raise ValueError(
            f"unknown adapter_name '{adapter_name}', expected one of {list(ADAPTER_REGISTRY)}"
        )
    return adapter_cls()
