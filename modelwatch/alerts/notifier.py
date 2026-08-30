"""Delivers an alert to an external webhook when one is created.

Before this, "an alert" meant a row in ModelWatch's own database --
real, and queryable, but only visible to someone who thought to look at
the dashboard. This module is what turns that into something that
actually reaches a person: a Slack channel or any generic webhook
receiver (PagerDuty, Discord, a custom endpoint, ...).

Deliberately does not add a dependency on a specific vendor SDK. A
webhook POST is the lowest common denominator every one of those
services already accepts, and it's the same integration surface Slack's
own "Incoming Webhooks" feature and PagerDuty's "Events API v2" both
expose.

Configured entirely through environment variables (see
modelwatch/config.py); unset by default, so a fresh local install stays
silent instead of failing to notify a webhook nobody configured. A
delivery failure is logged and swallowed, never raised -- a Slack
outage must never be the reason a drift check request fails.
"""
from __future__ import annotations

import logging
from typing import Any

import requests

from modelwatch.config import config

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 5


def _slack_payload(model_id: str, message: str, health_state: str) -> dict[str, Any]:
    """Slack's Incoming Webhooks API expects {"text": ...}; this also
    nests a few fields under "attachments" purely for readability in the
    Slack UI (a colored bar, a fallback text) -- Slack ignores fields it
    doesn't recognise, so this payload is safe to send to a generic
    webhook receiver too if someone points MODELWATCH_ALERT_WEBHOOK_URL
    at something other than Slack while leaving the format as "slack".
    """
    return {
        "text": f":rotating_light: ModelWatch alert — `{model_id}` is {health_state}",
        "attachments": [{"color": "#e5484d", "text": message}],
    }


def _generic_payload(model_id: str, message: str, health_state: str) -> dict[str, Any]:
    return {
        "source": "modelwatch",
        "model_id": model_id,
        "health_state": health_state,
        "message": message,
    }


def notify_alert(model_id: str, message: str, health_state: str) -> bool:
    """Best-effort webhook delivery. Returns True if a webhook was
    configured and the request succeeded, False otherwise -- callers
    that don't care can ignore the return value; tests use it to avoid
    asserting on log output."""
    url = config.alert_webhook_url
    if not url:
        return False

    payload = _slack_payload(model_id, message, health_state) if config.alert_webhook_format == "slack" \
        else _generic_payload(model_id, message, health_state)

    try:
        response = requests.post(url, json=payload, timeout=_TIMEOUT_SECONDS)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        logger.warning("alert webhook delivery failed", extra={"model_id": model_id, "error": str(e)})
        return False
