"""Pioneer (Fastino) API client — dataset upload, fine-tune, evaluate, infer.

API surface per https://agent.pioneer.ai/llms-full.txt (base https://api.pioneer.ai,
`X-API-Key` header, keys start with `pio_sk_`).

We frame match scoring as **single-label text classification**: each training row
is a serialised pair plus the label `connect` or `pass`, and we LoRA-fine-tune a
GLiNER2 encoder. That is the cheap, fast, task-specific shape Fastino's TLMs are
built for — a 205M encoder answering one yes/no question in milliseconds, which
is what the evolution loop needs when it scores a whole generation at once.

Stdlib only (urllib) so `pioneer/` stays a numpy-only package.

If no API key is present this module does nothing and `train.py` falls back to
the local head. It never fabricates a job id or a metric.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BASE_URL = os.environ.get("PIONEER_BASE_URL", "https://api.pioneer.ai")
DEFAULT_BASE_MODEL = os.environ.get("PIONEER_BASE_MODEL", "fastino/gliner2-base-v1")
DATASET_NAME = os.environ.get("PIONEER_DATASET", "kindred-match-pairs")
MODEL_NAME = os.environ.get("PIONEER_MODEL_NAME", "kindred-score-pair")

POSITIVE_LABEL = "connect"
NEGATIVE_LABEL = "pass"
CLASSIFICATION_TASK = "will_connect"

# The unified GLiNER2 schema dict the docs recommend over the legacy flat `task`.
INFERENCE_SCHEMA = {
    "classifications": [
        {"task": CLASSIFICATION_TASK, "labels": [POSITIVE_LABEL, NEGATIVE_LABEL]}
    ]
}


class PioneerError(RuntimeError):
    pass


def api_key() -> str | None:
    """PIONEER_API_KEY, or FASTINO_API_KEY for anyone who set the older name."""
    return os.environ.get("PIONEER_API_KEY") or os.environ.get("FASTINO_API_KEY") or None


def is_configured() -> bool:
    return api_key() is not None


@dataclass
class TrainingResult:
    """What a completed fine-tune hands back to the rest of the pipeline."""

    job_id: str
    status: str
    base_model: str
    dataset: str
    metrics: dict[str, Any] = field(default_factory=dict)
    logs_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "base_model": self.base_model,
            "dataset": self.dataset,
            "metrics": self.metrics,
        }


class PioneerClient:
    def __init__(self, key: str | None = None, base_url: str = BASE_URL, timeout: float = 60.0):
        self.key = key or api_key()
        if not self.key:
            raise PioneerError(
                "No Pioneer API key. Set PIONEER_API_KEY (keys start with 'pio_sk_')."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ---- transport ------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        raw: bytes | None = None,
        url: str | None = None,
        content_type: str = "application/json",
    ) -> Any:
        target = url or f"{self.base_url}{path}"
        data = raw if raw is not None else (json.dumps(body).encode("utf-8") if body is not None else None)
        req = urllib.request.Request(target, data=data, method=method)
        if url is None:  # presigned S3 PUTs must not carry our API key
            req.add_header("X-API-Key", self.key)
        if data is not None:
            req.add_header("Content-Type", content_type)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise PioneerError(f"{method} {target} -> {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise PioneerError(f"{method} {target} unreachable: {exc.reason}") from exc

        if not payload:
            return {}
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {"raw": payload.decode("utf-8", "replace")}

    # ---- datasets -------------------------------------------------------

    @staticmethod
    def build_classification_jsonl(rows: list[tuple[str, int]]) -> bytes:
        """rows are (pair_text, label) -> Pioneer single-label classification JSONL."""
        lines = [
            json.dumps({"text": text, "label": POSITIVE_LABEL if label == 1 else NEGATIVE_LABEL})
            for text, label in rows
        ]
        return ("\n".join(lines) + "\n").encode("utf-8")

    def upload_dataset(self, name: str, payload: bytes) -> str:
        """Presigned-URL upload flow: get URL -> PUT to S3 -> trigger processing."""
        handshake = self._request(
            "POST",
            "/felix/datasets/upload/url",
            {"dataset_name": name, "dataset_type": "classification", "format": "jsonl"},
        )
        upload_url = handshake.get("url") or handshake.get("upload_url") or handshake.get("presigned_url")
        dataset_id = handshake.get("dataset_id") or handshake.get("id")
        if not upload_url or not dataset_id:
            raise PioneerError(f"upload handshake missing url/dataset_id: {handshake}")

        self._request("PUT", "", raw=payload, url=upload_url, content_type="application/octet-stream")
        self._request("POST", "/felix/datasets/upload/process", {"dataset_id": dataset_id})
        return dataset_id

    def wait_for_dataset(self, name: str, poll_s: float = 5.0, timeout_s: float = 600.0) -> dict[str, Any]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            info = self._request("GET", f"/felix/datasets/{name}")
            status = str(_dig(info, "status") or "").lower()
            if status in {"ready", "complete", "completed", "succeeded"}:
                return info
            if status in {"failed", "error"}:
                raise PioneerError(f"dataset {name} failed to process: {info}")
            time.sleep(poll_s)
        raise PioneerError(f"dataset {name} not ready after {timeout_s:.0f}s")

    # ---- training -------------------------------------------------------

    def start_training(
        self,
        dataset_name: str,
        base_model: str = DEFAULT_BASE_MODEL,
        model_name: str = MODEL_NAME,
        epochs: int = 10,
        learning_rate: float = 5e-5,
        batch_size: int = 8,
    ) -> str:
        job = self._request(
            "POST",
            "/felix/training-jobs",
            {
                "model_name": model_name,
                "base_model": base_model,
                "datasets": [{"name": dataset_name}],
                "training_type": "lora",
                "nr_epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            },
        )
        job_id = job.get("id") or job.get("job_id") or job.get("training_job_id")
        if not job_id:
            raise PioneerError(f"training-jobs response has no job id: {job}")
        return str(job_id)

    def wait_for_training(
        self, job_id: str, poll_s: float = 15.0, timeout_s: float = 3600.0
    ) -> dict[str, Any]:
        deadline = time.time() + timeout_s
        last: dict[str, Any] = {}
        while time.time() < deadline:
            last = self._request("GET", f"/felix/training-jobs/{job_id}")
            status = str(_dig(last, "status") or "").lower()
            if status in {"complete", "completed", "succeeded", "success"}:
                return last
            if status in {"failed", "error", "cancelled", "canceled", "stopped"}:
                raise PioneerError(f"training job {job_id} ended as {status}: {last}")
            time.sleep(poll_s)
        raise PioneerError(f"training job {job_id} still {_dig(last, 'status')} after {timeout_s:.0f}s")

    def training_logs(self, job_id: str, tail: int = 2000) -> str:
        try:
            logs = self._request("GET", f"/felix/training-jobs/{job_id}/logs")
        except PioneerError:
            return ""
        text = logs if isinstance(logs, str) else json.dumps(logs)
        return text[-tail:]

    # ---- evaluation -----------------------------------------------------

    def evaluate(self, job_id: str, dataset_name: str) -> dict[str, Any]:
        """Pioneer's own F1/precision/recall on the dataset's held-out split.

        Reported alongside our numbers, not instead of them — our held-out split
        is the one the comparison against the baseline is computed on.
        """
        run = self._request(
            "POST", "/felix/evaluations", {"base_model": job_id, "dataset_name": dataset_name}
        )
        eval_id = run.get("id") or run.get("evaluation_id")
        if not eval_id:
            return run
        for _ in range(60):
            result = self._request("GET", f"/felix/evaluations/{eval_id}")
            status = str(_dig(result, "status") or "").lower()
            if status in {"complete", "completed", "succeeded", "success"}:
                return result
            if status in {"failed", "error"}:
                raise PioneerError(f"evaluation {eval_id} failed: {result}")
            time.sleep(10.0)
        raise PioneerError(f"evaluation {eval_id} did not finish in time")

    # ---- inference ------------------------------------------------------

    def score_text(self, model_id: str, text: str) -> float:
        response = self._request(
            "POST",
            "/inference",
            {"model_id": model_id, "text": text, "schema": INFERENCE_SCHEMA},
        )
        return extract_connect_probability(response)


def _dig(blob: Any, key: str) -> Any:
    """Find `key` anywhere in a nested response — status fields move around."""
    if isinstance(blob, dict):
        if key in blob:
            return blob[key]
        for value in blob.values():
            found = _dig(value, key)
            if found is not None:
                return found
    elif isinstance(blob, list):
        for item in blob:
            found = _dig(item, key)
            if found is not None:
                return found
    return None


def extract_connect_probability(response: Any) -> float:
    """Pull P(connect) out of a Pioneer classification response.

    The classification payload nests differently across model families, so this
    walks the response for any {label, score} record and returns the confidence
    attached to `connect` — using 1 - score(pass) if only the negative class
    came back. Raises rather than guessing if neither class is present, so a
    silent 0.5 can't leak into the loop's judgments.
    """
    found: dict[str, float] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            label = node.get("label") or node.get("class") or node.get("name")
            score = node.get("score", node.get("confidence", node.get("probability")))
            if isinstance(label, str) and isinstance(score, (int, float)):
                found.setdefault(label.strip().lower(), float(score))
            for key, value in node.items():
                # {"connect": 0.83, "pass": 0.17}
                if isinstance(value, (int, float)) and key.strip().lower() in {
                    POSITIVE_LABEL,
                    NEGATIVE_LABEL,
                }:
                    found.setdefault(key.strip().lower(), float(value))
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(response)

    if POSITIVE_LABEL in found:
        return _to_unit(found[POSITIVE_LABEL])
    if NEGATIVE_LABEL in found:
        return _to_unit(1.0 - found[NEGATIVE_LABEL])
    raise PioneerError(
        f"no '{POSITIVE_LABEL}'/'{NEGATIVE_LABEL}' confidence in Pioneer response: "
        f"{json.dumps(response)[:400]}"
    )


def _to_unit(value: float) -> float:
    """Accept 0-1 confidences or 0-100 percentages; clamp into [0, 1]."""
    if value > 1.0:
        value = value / 100.0
    return float(min(1.0, max(0.0, value)))


def save_dataset_preview(rows: list[tuple[str, int]], path: Path, limit: int = 5) -> None:
    """Write the exact JSONL we would upload — reviewable without an API key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = PioneerClient.build_classification_jsonl(rows[:limit] if limit else rows)
    path.write_bytes(payload)
