from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import requests


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
LOGGER = logging.getLogger(__name__)


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


def _build_prompt_payload(explanation_payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "model": os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        "temperature": 0.2,
        "max_completion_tokens": 300,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You explain ECG classifier outputs for a web app. "
                    "Use only the provided structured data. "
                    "Do not invent clinical facts, do not claim a diagnosis, "
                    "and do not mention information that was not supplied. "
                    "Return valid JSON with string fields: "
                    "summary, why_model_thinks_this, important_regions, limitations, clinical_warning."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(explanation_payload, ensure_ascii=True),
            },
        ],
    }


def _parse_report_content(content: str) -> Optional[Dict[str, str]]:
    if not content:
        return None

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or start >= end:
            return None
        try:
            parsed = json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            return None

    if not isinstance(parsed, dict):
        return None

    cleaned: Dict[str, str] = {}
    for key in (
        "summary",
        "why_model_thinks_this",
        "important_regions",
        "limitations",
        "clinical_warning",
    ):
        value = parsed.get(key)
        if value is not None:
            cleaned[key] = str(value).strip()

    return cleaned or None


def generate_groq_report(
    explanation_payload: Mapping[str, Any],
    timeout_seconds: int = 15,
    max_retries: int = 2,
) -> Optional[Dict[str, str]]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        LOGGER.warning("Groq explanation skipped: GROQ_API_KEY is missing.")
        return None

    payload = _build_prompt_payload(explanation_payload)
    model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "ecg-explainer/1.0",
    }

    LOGGER.info(
        "Sending Groq explanation request with model=%s for class=%s",
        model_name,
        explanation_payload.get("predicted_class_name", "unknown"),
    )

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )

            if response.status_code >= 400:
                error_text = response.text.strip()
                LOGGER.error(
                    "Groq explanation HTTP error %s: %s",
                    response.status_code,
                    error_text,
                )

                if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue

                return None

            body = response.json()
            LOGGER.info("Groq explanation response received with HTTP %s.", response.status_code)

            choices = body.get("choices") or []
            if not choices:
                LOGGER.error("Groq explanation response did not include choices.")
                return None

            message = choices[0].get("message") or {}
            content = message.get("content")

            if not isinstance(content, str):
                LOGGER.error("Groq explanation message content was missing or not a string.")
                return None

            report = _parse_report_content(content)
            if not report:
                LOGGER.error("Groq explanation content could not be parsed into the expected JSON fields.")
                return None

            LOGGER.info("Groq explanation generated successfully.")
            return report

        except requests.exceptions.Timeout:
            LOGGER.error("Groq explanation timed out after %s seconds.", timeout_seconds)
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None

        except requests.exceptions.ConnectionError as exc:
            LOGGER.error("Groq explanation connection error: %s", str(exc))
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None

        except requests.exceptions.RequestException as exc:
            LOGGER.error("Groq explanation request failed: %s", str(exc))
            return None

        except json.JSONDecodeError:
            LOGGER.error("Groq explanation response was not valid JSON.")
            return None

    return None


def format_report_for_ui(report: Optional[Mapping[str, str]]) -> Optional[str]:
    if not report:
        return None

    sections = [
        ("Summary", report.get("summary")),
        ("Why the model chose this", report.get("why_model_thinks_this")),
        ("Highlighted regions", report.get("important_regions")),
        ("Limitations", report.get("limitations")),
        ("Clinical warning", report.get("clinical_warning")),
    ]

    parts = [f"{label}: {text}" for label, text in sections if text]
    return " ".join(parts) if parts else None
