"""Embedding models used in the tutorial. Both are local and read at most 512 tokens."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

# name -> (Hugging Face id, dimensions). Same family, same tokenizer, so the chunks are identical and
# a comparison measures the model alone.
MODELS = {
    "small": ("BAAI/bge-small-en-v1.5", 384),
    "base": ("BAAI/bge-base-en-v1.5", 768),
}
# bge models expect this prefix on queries, not on passages.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def load(name: str) -> SentenceTransformer:
    return SentenceTransformer(MODELS[name][0])


def embed_passages(name: str, texts: list[str]):
    return load(name).encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)


def embed_query(name: str, question: str):
    return load(name).encode(QUERY_PREFIX + question, normalize_embeddings=True)
