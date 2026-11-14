"""Lesson 2: naive RAG over lovemesomecoding.com posts.

Deliberately simple, so later lessons have something to improve:
  - HTML flattened to text with a regex
  - fixed 1,000-character chunks that ignore headings and code blocks
  - vectors kept in a NumPy array (no database yet)

    python lessons/02_naive_rag.py "How do I create a thread in Java?"
    python lessons/02_naive_rag.py --retrieve-only "What is a Python decorator?"
"""

import argparse
import glob
import html
import json
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "data" / "naive_index.npz"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge models are trained with this prefix on queries (not on passages).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHUNK_CHARS = 1000
TOP_K = 5


def html_to_text(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)  # naive: drops every tag
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def build_index(model: SentenceTransformer) -> None:
    chunks, urls, titles = [], [], []
    for path in sorted(glob.glob(str(ROOT / "data" / "posts" / "*.json"))):
        post = json.loads(Path(path).read_text(encoding="utf-8"))
        text = html_to_text(post["html"])
        for start in range(0, len(text), CHUNK_CHARS):  # naive: fixed-size slices
            chunks.append(text[start : start + CHUNK_CHARS])
            urls.append(post["url"])
            titles.append(post["title"])
    print(f"Embedding {len(chunks)} chunks (one-time, cached in {INDEX.name})...")
    vectors = model.encode(chunks, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    np.savez(INDEX, vectors=vectors, chunks=chunks, urls=urls, titles=titles)


def retrieve(model: SentenceTransformer, question: str) -> list[dict]:
    data = np.load(INDEX)
    query = model.encode(QUERY_PREFIX + question, normalize_embeddings=True)
    scores = data["vectors"] @ query  # vectors are normalized, so dot product = cosine similarity
    best = np.argsort(-scores)[:TOP_K]
    return [
        {"score": float(scores[i]), "url": str(data["urls"][i]),
         "title": str(data["titles"][i]), "text": str(data["chunks"][i])}
        for i in best
    ]


def answer(question: str, hits: list[dict]) -> str:
    import anthropic

    context = "\n\n".join(
        f'<source id="{n}" title="{h["title"]}" url="{h["url"]}">\n{h["text"]}\n</source>'
        for n, h in enumerate(hits, 1)
    )
    response = anthropic.Anthropic().messages.create(
        model="claude-opus-5-5",
        max_tokens=4000,
        system=(
            "You answer programming questions using only the sources provided. "
            "Cite sources inline as [1], [2]. If the sources do not contain the answer, say so."
        ),
        messages=[{"role": "user", "content": f"{context}\n\nQuestion: {question}"}],
    )
    u = response.usage
    # $4 / $20 per million input / output tokens for claude-opus-5-5
    print(f"tokens: {u.input_tokens} in, {u.output_tokens} out "
          f"(~${(u.input_tokens * 4 + u.output_tokens * 20) / 1e6:.4f})")
    if response.stop_reason == "refusal":
        return "(The model declined to answer.)"
    return "".join(b.text for b in response.content if b.type == "text")


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--retrieve-only", action="store_true", help="show retrieved chunks, skip Claude")
    args = ap.parse_args()

    model = SentenceTransformer(EMBED_MODEL)
    if not INDEX.exists():
        build_index(model)

    hits = retrieve(model, args.question)
    for n, h in enumerate(hits, 1):
        print(f"[{n}] {h['score']:.3f}  {h['title']}  {h['url']}")
    if args.retrieve_only:
        for n, h in enumerate(hits, 1):
            print(f"\n--- [{n}] ---\n{h['text'][:400]}...")
        return
    print("\n" + answer(args.question, hits))


if __name__ == "__main__":
    main()
