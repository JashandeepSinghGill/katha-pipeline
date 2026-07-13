import psycopg
from psycopg.rows import dict_row

from katha.config import Settings


def connect(settings: Settings) -> psycopg.Connection:
    return psycopg.connect(settings.database_url, row_factory=dict_row)
