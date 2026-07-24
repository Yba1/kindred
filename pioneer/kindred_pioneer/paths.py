"""Canonical on-disk locations for the pioneer workstream."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PIONEER_ROOT = PACKAGE_ROOT.parent

DATA_DIR = PIONEER_ROOT / "data"
ARTIFACT_DIR = PIONEER_ROOT / "artifacts"

PAIRS_PATH = DATA_DIR / "pairs.jsonl"
COLDSTART_PATH = DATA_DIR / "pairs_coldstart.jsonl"
PIONEER_UPLOAD_PREVIEW = DATA_DIR / "pioneer_train.preview.jsonl"

MODEL_PATH = ARTIFACT_DIR / "model.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
REPORT_PATH = ARTIFACT_DIR / "report.md"
