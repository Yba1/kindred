"""Keep the test suite out of the demo's Actian collections.

`test_api.py` drives the real app, which upserts every profile it builds. Point
it at a scratch collection prefix (set before app.config is imported, and
load_dotenv never overrides an existing env var) so repeated test runs don't
silently accumulate "Test User" rows in the collections the demo reads from.
Collections are dropped at the end of the session; if Actian isn't running this
is all a no-op and the suite runs offline on the numpy fallback.
"""
import os

os.environ["ACTIAN_DB"] = "kindred_test"

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _drop_test_collections():
    yield
    try:
        from actian_vectorai import VectorAIClient

        from app.config import settings

        if not settings.actian_enabled:
            return
        with VectorAIClient(
            f"{settings.actian_host}:{settings.actian_port or '6574'}",
            timeout=settings.actian_timeout,
        ) as c:
            for name in c.collections.list():
                if name.startswith("kindred_test") or name.startswith("kindred_pytest"):
                    c.collections.delete(name, strict=False)
    except Exception:
        pass
