import os
import subprocess

import pytest

from katha.config import Settings
from katha.db import connect

TEST_DB_URL = os.environ.get(
    "KATHA_TEST_DATABASE_URL",
    "postgresql://katha:katha@localhost:5432/katha_test",
)
TABLES = ("words", "segments", "jobs", "audio_files", "clips")


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(_env_file=None, database_url=TEST_DB_URL)


@pytest.fixture(scope="session", autouse=True)
def _migrate(settings):
    env = {**os.environ, "KATHA_DATABASE_URL": settings.database_url}
    subprocess.run(["alembic", "upgrade", "head"], env=env, check=True)


@pytest.fixture
def db(settings):
    conn = connect(settings)
    yield conn
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE")
    conn.commit()
    conn.close()
