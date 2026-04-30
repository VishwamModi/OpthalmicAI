from __future__ import annotations

import socket
import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "ophthalmic_ai.db"
load_dotenv(BASE_DIR / ".env", override=True)
load_dotenv(BASE_DIR / "api_keys.env", override=True)
load_dotenv(BASE_DIR / "api_key.env", override=True)

def _clean_url(value: str) -> str:
    cleaned = (value or "").strip().strip('"').strip("'")
    if "postgresql://" in cleaned:
        cleaned = cleaned[cleaned.index("postgresql://") :]
    elif "postgres://" in cleaned:
        cleaned = cleaned[cleaned.index("postgres://") :]
    if cleaned.startswith("postgres://"):
        cleaned = "postgresql://" + cleaned[len("postgres://") :]
    return cleaned


# Prefer an explicit Supabase URL if provided, otherwise fall back to DATABASE_URL.
raw_db_url = _clean_url(os.getenv("SUPABASE_DATABASE_URL") or os.getenv("DATABASE_URL") or "")

DATABASE_URL = raw_db_url or None
SQLALCHEMY_DATABASE_URL = DATABASE_URL or f"sqlite:///{DB_PATH}"
IS_SQLITE = SQLALCHEMY_DATABASE_URL.startswith("sqlite")

if IS_SQLITE:
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    # PostgreSQL / cloud database mode
    # Supabase Postgres typically requires SSL and benefits from a short connect timeout.
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args={"sslmode": "require", "connect_timeout": 5},
    )

    # If Postgres isn't reachable (DNS / network / credentials), fall back to SQLite
    # so local development can continue.
    try:
        host = SQLALCHEMY_DATABASE_URL.split("@", 1)[1].split("/", 1)[0].split(":", 1)[0]
        try:
            socket.getaddrinfo(host, None)
        except socket.gaierror as dns_error:
            raise OperationalError(f"DNS lookup failed for database host {host}", params=None, orig=dns_error) from dns_error
        with engine.connect() as _conn:
            pass
    except OperationalError as e:  # pragma: no cover
        print(
            "[database] WARNING: Postgres unreachable, falling back to SQLite. "
            f"URL={SQLALCHEMY_DATABASE_URL.split('@', 1)[-1]} Details: {e}"
        )
        SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"
        IS_SQLITE = True
        engine = create_engine(
            SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
        )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

