"""Environment config + capability detection.

Everything is optional. `settings` reports which sponsors are wired so /health
and the profiler/embeddings/Actian modules can pick their live-vs-fallback path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip())

    # Actian VectorAI DB. HOST + PORT is all the container needs (gRPC on 6574,
    # auth disabled by default). DB is a collection-name prefix; USER/PASSWORD
    # are only used when the deployment has auth switched on.
    actian_host: str = field(default_factory=lambda: os.getenv("ACTIAN_HOST", "").strip())
    actian_port: str = field(default_factory=lambda: os.getenv("ACTIAN_PORT", "").strip())
    actian_db: str = field(default_factory=lambda: os.getenv("ACTIAN_DB", "").strip())
    actian_user: str = field(default_factory=lambda: os.getenv("ACTIAN_USER", "").strip())
    actian_password: str = field(default_factory=lambda: os.getenv("ACTIAN_PASSWORD", "").strip())
    # Keep this short: it bounds how long boot stalls before the numpy fallback engages.
    actian_timeout: float = field(default_factory=lambda: _env_float("ACTIAN_TIMEOUT", 5.0))

    embed_dim: int = 256

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def actian_enabled(self) -> bool:
        # Host alone is enough — no database/user/password on the local container.
        return bool(self.actian_host)


settings = Settings()
