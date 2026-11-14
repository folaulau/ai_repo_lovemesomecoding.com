"""Chunks and their vectors in Postgres + pgvector (lesson 4).

One table per embedding model (chunks_small, chunks_base): a vector column has a fixed number of
dimensions, and vectors from different models are not comparable anyway.
"""

from __future__ import annotations

import os

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql

from .embed import MODELS


def connect() -> psycopg.Connection:
    # autocommit matters: without it the first query opens a transaction that stays open, every
    # `with conn.transaction()` below becomes a savepoint inside it, and `SET LOCAL` in search()
    # then outlives the query - silently turning the index off for everything that follows.
    conn = psycopg.connect(
        os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5434/rag"), autocommit=True
    )
    register_vector(conn)  # lets psycopg send and receive NumPy arrays as `vector`
    return conn


def table(model: str) -> sql.Identifier:
    return sql.Identifier(f"chunks_{model}")


def recreate(conn: psycopg.Connection, model: str) -> None:
    dim = MODELS[model][1]
    conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(table(model)))
    conn.execute(
        sql.SQL("""
            CREATE TABLE {} (
                id        bigserial PRIMARY KEY,
                url       text NOT NULL,
                category  text NOT NULL,
                title     text NOT NULL,
                heading   text NOT NULL,
                content   text NOT NULL,      -- the chunk text exactly as embedded
                embedding vector({}) NOT NULL
            )""").format(table(model), sql.Literal(dim))
    )


def load_rows(conn: psycopg.Connection, model: str, chunks, vectors) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            sql.SQL("INSERT INTO {} (url, category, title, heading, content, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s)").format(table(model)),
            [(c.url, c.category, c.title, c.heading, c.text, v) for c, v in zip(chunks, vectors)],
        )


def create_index(conn: psycopg.Connection, model: str) -> None:
    # Built after loading: one bulk build is much faster than updating the graph on every insert.
    # vector_cosine_ops matches the <=> (cosine distance) operator used in search().
    conn.execute(
        sql.SQL("CREATE INDEX ON {} USING hnsw (embedding vector_cosine_ops)").format(table(model))
    )
    conn.execute(sql.SQL("ANALYZE {}").format(table(model)))


def search(conn: psycopg.Connection, model: str, query_vec, k: int = 5, exact: bool = False) -> list[dict]:
    """Top-k chunks by cosine similarity. exact=True turns the index off to get the true nearest
    neighbours; the HNSW index is approximate and can miss some."""
    with conn.transaction():
        if exact:
            conn.execute("SET LOCAL enable_indexscan = off")
        rows = conn.execute(
            sql.SQL("""
                SELECT url, title, content, 1 - (embedding <=> %s) AS score
                FROM {}
                ORDER BY embedding <=> %s
                LIMIT %s""").format(table(model)),
            (query_vec, query_vec, k),
        ).fetchall()
    return [{"url": u, "title": t, "text": c, "score": float(s)} for u, t, c, s in rows]
