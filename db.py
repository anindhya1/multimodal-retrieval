"""
Database clients for Genery.

Usage:
    from db import pg, qdrant, Collections

    # PostgreSQL — raw SQL
    with pg.connect() as conn:
        rows = conn.execute("SELECT id, title, slug FROM titles LIMIT 10").fetchall()

    # PostgreSQL — SQLAlchemy ORM / Core
    from sqlalchemy import text
    with pg.connect() as conn:
        rows = conn.execute(text("SELECT * FROM media_items WHERE title_id = :tid"), {"tid": 1}).fetchall()

    # Qdrant — vector search
    hits = qdrant.query(
        collection_name=Collections.STILLS,
        query_vector=my_vector,   # list[float], length 768
        limit=10,
    )
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from sqlalchemy import create_engine, Engine
from qdrant_client import QdrantClient

load_dotenv()


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------

def _make_pg_engine() -> Engine:
    url = os.environ["DATABASE_URL"]
    return create_engine(url, pool_pre_ping=True)

pg: Engine = _make_pg_engine()


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

def _make_qdrant_client() -> QdrantClient:
    endpoint = os.environ["QDRANT_ENDPOINT"]
    api_key = os.environ["QDRANT_API_KEY"]
    return QdrantClient(url=endpoint, api_key=api_key, timeout=10)

qdrant: QdrantClient = _make_qdrant_client()


# ---------------------------------------------------------------------------
# Collection names
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Collections:
    STILLS: str
    VIDEOS: str
    EFFECTS: str
    PERSONS: str
    PREDICT: str
    TREATMENTS: str


def _make_collections() -> _Collections:
    missing = [
        key for key in (
            "QDRANT_STILLS_COLLECTION",
            "QDRANT_VIDEOS_COLLECTION",
            "QDRANT_EFFECTS_COLLECTION",
            "QDRANT_PERSONS_COLLECTION",
            "QDRANT_PREDICT_COLLECTION",
            "QDRANT_TREATMENTS_COLLECTION",
        )
        if not os.environ.get(key)
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    return _Collections(
        STILLS     = os.environ["QDRANT_STILLS_COLLECTION"],
        VIDEOS     = os.environ["QDRANT_VIDEOS_COLLECTION"],
        EFFECTS    = os.environ["QDRANT_EFFECTS_COLLECTION"],
        PERSONS    = os.environ["QDRANT_PERSONS_COLLECTION"],
        PREDICT    = os.environ["QDRANT_PREDICT_COLLECTION"],
        TREATMENTS = os.environ["QDRANT_TREATMENTS_COLLECTION"],
    )

Collections = _make_collections()
