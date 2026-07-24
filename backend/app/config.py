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


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip())

    actian_host: str = field(default_factory=lambda: os.getenv("ACTIAN_HOST", "").strip())
    actian_port: str = field(default_factory=lambda: os.getenv("ACTIAN_PORT", "").strip())
    actian_db: str = field(default_factory=lambda: os.getenv("ACTIAN_DB", "").strip())
    actian_user: str = field(default_factory=lambda: os.getenv("ACTIAN_USER", "").strip())
    actian_password: str = field(default_factory=lambda: os.getenv("ACTIAN_PASSWORD", "").strip())

    embed_dim: int = 256

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def actian_enabled(self) -> bool:
        return bool(self.actian_host and self.actian_db)


settings = Settings()
