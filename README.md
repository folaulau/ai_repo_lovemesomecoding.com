# Ask LoveMeSomeCoding: a RAG tutorial project

Code for the RAG tutorial series on [lovemesomecoding.com/rag](https://lovemesomecoding.com/rag).
We build a question-answering assistant over the site's own ~989 tutorials that answers from the posts
and cites the ones it used. Everything runs locally except the call to Claude.

| Piece | Choice |
|---|---|
| Vector DB | Postgres + pgvector (Docker) |
| Keyword search | Postgres full-text search |
| Embeddings | `BAAI/bge-small-en-v1.5`, local via sentence-transformers |
| LLM | Claude API |

## Setup

Requires Python 3.12+ and Docker.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env            # add your ANTHROPIC_API_KEY (needed from the RAG lessons on)
docker compose up -d --wait     # Postgres + pgvector on localhost:5434
```

## Get the data

```bash
python scripts/export_posts.py             # ~1 minute, ~24 MB into data/posts/
python scripts/export_posts.py --limit 20  # or a quick sample
```

It reads the public `search-index.json`, then saves each post's `<article>` body as one JSON file:

```json
{"url": "...", "category": "java-8", "slug": "...", "title": "...", "excerpt": "...",
 "published": "...", "modified": "...", "html": "<p>..."}
```

We keep HTML, not plain text: the headings and `<pre>` code blocks are what we chunk on later.
`data/` is gitignored, so everyone generates their own copy.

## Lessons

| # | Run | What it shows |
|---|---|---|
| 2 | `python lessons/02_naive_rag.py "How do I create a thread in Java?"` | Naive RAG: regex HTML-to-text, fixed 1,000-char chunks, NumPy cosine search, Claude answers with citations. Add `--retrieve-only` to see the chunks without calling Claude. The first run embeds ~8,000 chunks (under a minute on Apple Silicon) and caches them in `data/naive_index.npz`. |
| 3 | `python lessons/03_chunking.py --compare "What is a Python decorator?"` | Structure-aware chunking (`ask_lmsc/chunking.py`): split at headings, keep code blocks whole and exact, drop Copy buttons, prefix each chunk with category/title/section, skip stub posts. `--compare` prints lesson 2's top 5 next to lesson 3's. |

## Check the database

```bash
docker exec ask-lmsc-db psql -U rag -d rag -c "select '[1,2,3]'::vector <=> '[1,2,4]'::vector;"
```

A small number back (cosine distance) means pgvector is working.

`docker compose down` stops it; `docker compose down -v` also deletes the data volume.
