"""Lesson 3: chunk by structure, not by character count.

Same embeddings, same NumPy search, same prompt as lesson 2. Only the chunking changes
(see ask_lmsc/chunking.py), so any difference in results comes from chunking alone.

    python lessons/03_chunking.py "What is a Python decorator?"
    python lessons/03_chunking.py --retrieve-only "What is a Python decorator?"
    python lessons/03_chunking.py --compare "What is a Python decorator?"   # lesson 2 vs lesson 3
"""

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from ask_lmsc.chunking import chunk_post  # noqa: E402

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K = 5
INDEXES = {"naive": ROOT / "data" / "naive_index.npz", "chunked": ROOT / "data" / "chunked_index.npz"}


def build_index(model: SentenceTransformer) -> None:
    chunks = []
    for path in sorted(glob.glob(str(ROOT / "data" / "posts" / "*.json"))):
        chunks += chunk_post(json.loads(Path(path).read_text(encoding="utf-8")))
    print(f"Embedding {len(chunks)} chunks (one-time, cached in {INDEXES['chunked'].name})...")
    texts = [c.text for c in chunks]
    vectors = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    np.savez(INDEXES["chunked"], vectors=vectors, chunks=texts,
             urls=[c.url for c in chunks], titles=[c.title for c in chunks])


def retrieve(model: SentenceTransformer, question: str, index: str) -> list[dict]:
    data = np.load(INDEXES[index])
    scores = data["vectors"] @ model.encode(QUERY_PREFIX + question, normalize_embeddings=True)
    best = np.argsort(-scores)[:TOP_K]
    return [{"score": float(scores[i]), "url": str(data["urls"][i]), "title": str(data["titles"][i]),
             "text": str(data["chunks"][i])} for i in best]


def show(label: str, hits: list[dict]) -> None:
    print(f"\n{label}")
    for n, h in enumerate(hits, 1):
        print(f"  [{n}] {h['score']:.3f}  {h['url'].removeprefix('https://lovemesomecoding.com')}")


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--retrieve-only", action="store_true", help="show retrieved chunks, skip Claude")
    ap.add_argument("--compare", action="store_true", help="top 5 from lesson 2's index vs this one")
    args = ap.parse_args()

    model = SentenceTransformer(EMBED_MODEL)
    if not INDEXES["chunked"].exists():
        build_index(model)

    if args.compare:
        if not INDEXES["naive"].exists():
            sys.exit("Run lessons/02_naive_rag.py once first to build the naive index.")
        show("lesson 2 (fixed 1,000-char chunks)", retrieve(model, args.question, "naive"))
        show("lesson 3 (structure-aware chunks)", retrieve(model, args.question, "chunked"))
        return

    hits = retrieve(model, args.question, "chunked")
    show(args.question, hits)
    if args.retrieve_only:
        for n, h in enumerate(hits, 1):
            print(f"\n--- [{n}] ---\n{h['text'][:500]}")
        return

    from ask_lmsc.generate import answer

    print("\n" + answer(args.question, hits))


if __name__ == "__main__":
    main()
