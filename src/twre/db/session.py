import os
import atexit
from contextlib import contextmanager
from pathlib import Path
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool

load_dotenv()

_pool: ConnectionPool | None = None

def get_db_str() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL environment variable is missing.")
    return url

def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(conninfo=get_db_str(), min_size=1, max_size=5, open=True)
    return _pool

@contextmanager
def get_connection():
    pool = get_pool()
    with pool.connection() as conn:
        yield conn

def init_db() -> None:
    ddl_path = Path(__file__).resolve().parents[3] / "sql" / "01_ddl.sql"
    sql = ddl_path.read_text()
    with get_connection() as conn:
        conn.execute(sql)
    print("DB initialized successfully")

def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None

atexit.register(close_pool)

if __name__ == "__main__":
    init_db()