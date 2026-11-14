"""Lesson 4: move the vectors into Postgres + pgvector.

Same chunks as lesson 3. New: a real vector store with an HNSW index, and a second embedding model to
see what a bigger one costs.

    python lessons/04_pgvector.py ingest --model small        # chunk, embed, load, index
    python lessons/04_pgvector.py ingest --model base
    python lessons/04_pgvector.py ask "What is a Python decorator?" [--model base] [--retrieve-only]
    python lessons/04_pgvector.py bench                        # speed, size, HNSW recall, per model
"""

import argparse
import glob
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
from psycopg import sql  # noqa: E402

from ask_lmsc import embed, store  # noqa: E402
from ask_lmsc.chunking import chunk_post  # noqa: E402


def all_chunks():
    chunks = []
    for path in sorted(glob.glob(str(ROOT / "data" / "posts" / "*.json"))):
        chunks += chunk_post(json.loads(Path(path).read_text(encoding="utf-8")))
    return chunks


def ingest(model: str) -> None:
    chunks = all_chunks()
    t0 = time.perf_counter()
    vectors = embed.embed_passages(model, [c.text for c in chunks])
    t_embed = time.perf_counter() - t0
    with store.connect() as conn:
        store.recreate(conn, model)
        t0 = time.perf_counter()
        store.load_rows(conn, model, chunks, vectors)
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        store.create_index(conn, model)
        t_index = time.perf_counter() - t0
    print(f"{model}: {len(chunks)} chunks  embed {t_embed:.0f}s  insert {t_load:.1f}s  "
          f"HNSW build {t_index:.1f}s")


def ask(question: str, model: str, retrieve_only: bool) -> None:
    with store.connect() as conn:
        hits = store.search(conn, model, embed.embed_query(model, question))
    for n, h in enumerate(hits, 1):
        print(f"  [{n}] {h['score']:.3f}  {h['url'].removeprefix('https://lovemesomecoding.com')}")
    if retrieve_only:
        return
    from ask_lmsc.generate import answer

    print("\n" + answer(question, hits))


def bench(n_queries: int = 50) -> None:
    """Post titles stand in for questions here; lesson 5 replaces them with a real golden set."""
    titles = sorted({c.title for c in all_chunks()})
    queries = random.Random(42).sample(titles, n_queries)
    with store.connect() as conn:
        for model in embed.MODELS:
            exists = conn.execute("SELECT to_regclass(%s)", (f"chunks_{model}",)).fetchone()[0]
            if not exists:
                print(f"{model}: not ingested, skipping")
                continue
            size = conn.execute(
                sql.SQL("SELECT pg_size_pretty(pg_total_relation_size({}))").format(
                    sql.Literal(f"chunks_{model}"))
            ).fetchone()[0]
            vecs = [embed.embed_query(model, q) for q in queries]
            store.search(conn, model, vecs[0])  # warm-up
            t_ann, t_exact, recall = [], [], []
            for v in vecs:
                t0 = time.perf_counter()
                ann = store.search(conn, model, v)
                t_ann.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                exact = store.search(conn, model, v, exact=True)
                t_exact.append(time.perf_counter() - t0)
                recall.append(len({h["text"] for h in ann} & {h["text"] for h in exact}) / len(exact))
            print(f"{model}: table+index {size}  query HNSW {statistics.median(t_ann) * 1000:.1f} ms  "
                  f"exact {statistics.median(t_exact) * 1000:.1f} ms  "
                  f"HNSW recall@5 vs exact {statistics.mean(recall):.1%}")


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest")
    p.add_argument("--model", choices=embed.MODELS, default="small")
    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--model", choices=embed.MODELS, default="small")
    p.add_argument("--retrieve-only", action="store_true")
    sub.add_parser("bench")
    args = ap.parse_args()

    if args.cmd == "ingest":
        ingest(args.model)
    elif args.cmd == "ask":
        ask(args.question, args.model, args.retrieve_only)
    else:
        bench()


if __name__ == "__main__":
    main()
