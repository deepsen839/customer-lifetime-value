import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

def _configured_url():
    # Environment variables take precedence (Docker/local production convention).
    url = os.getenv("DATABASE_URL")
    if not url:
        # Streamlit Community Cloud injects values via st.secrets, not environment vars.
        try:
            import streamlit as st
            url = st.secrets.get("DATABASE_URL")
            if not url and "database" in st.secrets:
                cfg = st.secrets["database"]
                if cfg.get("url"):
                    url = cfg.get("url")
                elif all(cfg.get(k) for k in ("host", "database", "username", "password")):
                    url = URL.create(
                        "postgresql+psycopg",
                        username=cfg["username"],
                        password=cfg["password"],
                        host=cfg["host"],
                        port=int(cfg.get("port", 5432)),
                        database=cfg["database"],
                        query={"sslmode": cfg.get("sslmode", "require")},
                    )
        except (ImportError, FileNotFoundError, KeyError, AttributeError):
            # Streamlit secrets are optional for local CLI training/tests.
            url = None
    if url:
        # Preserve SQLAlchemy URL objects so passwords with reserved characters
        # remain correctly encoded rather than being stringified/masked.
        if isinstance(url, URL):
            return url
        # Managed providers often return postgres:// or postgresql://. Explicitly
        # select psycopg v3, which is installed by requirements.txt.
        url = str(url)
        if url.startswith("postgres://"):
            url = "postgresql+psycopg://" + url[len("postgres://"):]
        elif url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url[len("postgresql://"):]
        return url
    # Compose can pass discrete credentials to avoid breaking passwords that
    # contain URI-reserved characters such as @, /, or #.
    if os.getenv("POSTGRES_HOST") and os.getenv("POSTGRES_DB") and os.getenv("POSTGRES_USER") and os.getenv("POSTGRES_PASSWORD"):
        return URL.create(
            "postgresql+psycopg",
            username=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
            host=os.environ["POSTGRES_HOST"],
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.environ["POSTGRES_DB"],
        )
    return None

def get_engine():
    url = _configured_url()
    if url is None:
        return None
    return create_engine(url, pool_pre_ping=True, future=True)

def init_db(engine):
    if engine is None:
        return
    sql_path = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"
    # Execute statements individually: psycopg prepared execution can reject a
    # multi-statement SQL string even though PostgreSQL accepts each statement.
    statements = [part.strip() for part in sql_path.read_text(encoding="utf-8").split(";") if part.strip()]
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))

def replace_table(engine, table, df):
    if engine is None:
        raise RuntimeError("DATABASE_URL is not configured")
    if not table.replace("_", "").isalnum():
        raise ValueError("Invalid table name")
    # Preserve schema constraints/indexes created by sql/schema.sql. Fail loudly if
    # the dataframe has columns not present in the declared table.
    with engine.begin() as conn:
        conn.execute(text(f'DELETE FROM "{table}"'))
        df.to_sql(table, conn, if_exists="append", index=False, method="multi", chunksize=500)

def read_table(engine, table, limit=1000000):
    if engine is None:
        raise RuntimeError("DATABASE_URL is not configured")
    if not table.replace("_", "").isalnum():
        raise ValueError("Invalid table name")
    return pd.read_sql_query(text(f'SELECT * FROM "{table}" LIMIT :n'), engine, params={"n": int(limit)})
