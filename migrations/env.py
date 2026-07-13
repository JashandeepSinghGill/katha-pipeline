import os

from alembic import context
from sqlalchemy import create_engine

from katha.config import get_settings


def _url() -> str:
    return os.environ.get("KATHA_DATABASE_URL") or get_settings().database_url


def _sqlalchemy_url() -> str:
    # The project standardizes on psycopg (v3) everywhere (see katha.db.connect)
    # and never installs psycopg2. SQLAlchemy's default dialect for a bare
    # "postgresql://" URL is psycopg2, so route it to the psycopg3 driver here
    # without changing the KATHA_DATABASE_URL convention used elsewhere.
    url = _url()
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def run_migrations_offline() -> None:
    context.configure(url=_url(), literal_binds=True, target_metadata=None)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_sqlalchemy_url())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
