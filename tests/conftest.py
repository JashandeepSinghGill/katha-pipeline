import os
import subprocess
import sys

import pytest

from katha.config import Settings
from katha.db import connect

# The fallback password here must match POSTGRES_PASSWORD in your local .env
# (NOT the `change-me` placeholder in .env.example). Set KATHA_TEST_DATABASE_URL
# to override this whole URL if your local setup differs.
TEST_DB_URL = os.environ.get(
    "KATHA_TEST_DATABASE_URL",
    "postgresql://katha:katha@localhost:5432/katha_test",
)
TABLES = ("words", "segments", "jobs", "audio_files", "clips")


@pytest.fixture(scope="session")
def settings() -> Settings:
    # _env_file=None keeps this fixture from picking up unrelated values from a
    # dev .env (e.g. a live Sarvam key). `device` is the exception: it has no
    # safe universal default (Settings defaults to "cuda"), so pull it from the
    # real environment (falling back to "cpu" for this Windows/no-GPU dev box)
    # instead of silently forcing a dtype branch that doesn't match the machine.
    return Settings(
        _env_file=None,
        database_url=TEST_DB_URL,
        device=os.environ.get("KATHA_DEVICE", "cpu"),
    )


@pytest.fixture(scope="session", autouse=True)
def _migrate(settings):
    env = {**os.environ, "KATHA_DATABASE_URL": settings.database_url}
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"], env=env, check=True
    )


@pytest.fixture
def db(settings):
    conn = connect(settings)
    yield conn
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE")
    conn.commit()
    conn.close()
