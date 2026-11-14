"""FastAPI backend for the test UI in web/. Wraps the same code the lessons use.

    uvicorn api.main:app --port 8010 --reload       # from the repo root, venv active

POST /api/search  retrieval only (free)
POST /api/ask     retrieval + Claude answer (costs ~$0.02)
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ask_lmsc import embed, store
from ask_lmsc.generate import MODEL, PRICE_IN, PRICE_OUT, SYSTEM

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@asynccontextmanager
async def lifespan(app: FastAPI):
    embed.load("small")  # load the default model at startup, not on the first request
    yield


app = FastAPI(title="Ask LoveMeSomeCoding", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5180", "http://127.0.0.1:5180"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    model: Literal["small", "base"] = "small"
    top_k: int = Field(5, ge=1, le=10)


class Source(BaseModel):
    n: int
    url: str
    title: str
    score: float
    text: str


class SearchResult(BaseModel):
    sources: list[Source]
    retrieve_ms: float


class AskResult(SearchResult):
    answer: str
    generate_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _retrieve(q: Query) -> SearchResult:
    t0 = time.perf_counter()
    with store.connect() as conn:
        exists = conn.execute("SELECT to_regclass(%s)", (f"chunks_{q.model}",)).fetchone()[0]
        if not exists:
            raise HTTPException(
                409, f"Model '{q.model}' is not ingested. Run: python lessons/04_pgvector.py ingest --model {q.model}"
            )
        hits = store.search(conn, q.model, embed.embed_query(q.model, q.question), k=q.top_k)
    return SearchResult(
        sources=[Source(n=i, **h) for i, h in enumerate(hits, 1)],
        retrieve_ms=(time.perf_counter() - t0) * 1000,
    )


@app.get("/api/health")
def health() -> dict:
    with store.connect() as conn:
        counts = {
            m: conn.execute(
                "SELECT CASE WHEN to_regclass(%s) IS NULL THEN 0 ELSE "
                "(SELECT reltuples::bigint FROM pg_class WHERE oid = to_regclass(%s)) END",
                (f"chunks_{m}", f"chunks_{m}"),
            ).fetchone()[0]
            for m in embed.MODELS
        }
    return {"ok": True, "chunks": counts, "llm": MODEL}


@app.post("/api/search")
def search(q: Query) -> SearchResult:
    return _retrieve(q)


@app.post("/api/ask")
def ask(q: Query) -> AskResult:
    found = _retrieve(q)
    context = "\n\n".join(
        f'<source id="{s.n}" url="{s.url}">\n{s.text}\n</source>' for s in found.sources
    )
    t0 = time.perf_counter()
    try:
        response = anthropic.Anthropic().messages.create(
            model=MODEL,
            max_tokens=4000,
            system=SYSTEM,
            messages=[{"role": "user", "content": f"{context}\n\nQuestion: {q.question}"}],
        )
    except anthropic.AuthenticationError:
        raise HTTPException(500, "ANTHROPIC_API_KEY in .env is missing or invalid")
    except anthropic.RateLimitError:
        raise HTTPException(429, "Claude API rate limit hit; try again shortly")
    except anthropic.APIConnectionError:
        raise HTTPException(502, "Could not reach the Claude API")
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"Claude API error {e.status_code}: {e.message}")

    text = "".join(b.text for b in response.content if b.type == "text")
    if response.stop_reason == "refusal":
        text = "(The model declined to answer.)"
    u = response.usage
    return AskResult(
        **found.model_dump(),
        answer=text,
        generate_ms=(time.perf_counter() - t0) * 1000,
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cost_usd=(u.input_tokens * PRICE_IN + u.output_tokens * PRICE_OUT) / 1e6,
    )
