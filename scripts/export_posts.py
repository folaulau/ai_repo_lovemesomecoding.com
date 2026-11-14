"""Download every lovemesomecoding.com post into data/posts/ as one JSON file per post.

Uses only public URLs, so anyone can run it:
  1. https://lovemesomecoding.com/search-index.json lists every post (url, title, category, excerpt).
  2. Each post page is fetched and the <article class="prose"> body is kept as HTML.

We keep HTML rather than plain text on purpose: headings and <pre> code blocks are what the chunking
lesson splits on, and they are gone once the page is flattened to text.

Re-running skips posts already on disk; pass --refresh to download everything again.

    python scripts/export_posts.py            # all posts
    python scripts/export_posts.py --limit 20 # a quick sample
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# requests (not urllib) because it ships its own CA bundle via certifi. The python.org macOS build of
# Python has no root certificates until you run "Install Certificates.command", so urllib fails there
# with CERTIFICATE_VERIFY_FAILED.
import requests

SITE = "https://lovemesomecoding.com"
USER_AGENT = "ask-lmsc-rag-tutorial/0.1"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "posts"

# The post body. Exactly one per page; nested <article> tags do not occur on the site.
ARTICLE_RE = re.compile(r'<article class="prose">(.*?)</article>', re.S)
META_RE = re.compile(r'<meta property="article:(published_time|modified_time)" content="([^"]*)"')


def fetch(url: str, retries: int = 3) -> str:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            return resp.text
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def export_one(entry: dict, refresh: bool) -> tuple[str, str]:
    """Returns (status, url). status is 'saved', 'skipped' or 'error: ...'."""
    path = entry["u"]  # e.g. /java-8/java-8-lambda
    category, slug = path.strip("/").split("/")
    out = OUT_DIR / f"{category}__{slug}.json"
    if out.exists() and not refresh:
        return "skipped", path

    try:
        page = fetch(SITE + path)
    except Exception as exc:  # noqa: BLE001 - report and keep going
        return f"error: {exc}", path

    article = ARTICLE_RE.search(page)
    if not article:
        return "error: no <article class=\"prose\"> on page", path

    meta = dict(META_RE.findall(page))
    record = {
        "url": SITE + path,
        "category": category,
        "slug": slug,
        "title": entry["t"],
        "excerpt": entry.get("x", ""),
        "published": meta.get("published_time"),
        "modified": meta.get("modified_time"),
        "html": article.group(1).strip(),
    }
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return "saved", path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="only export the first N posts")
    ap.add_argument("--refresh", action="store_true", help="re-download posts already on disk")
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads (be polite; default 4)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = json.loads(fetch(f"{SITE}/search-index.json"))
    if args.limit:
        index = index[: args.limit]
    print(f"{len(index)} posts listed in search-index.json")

    counts = {"saved": 0, "skipped": 0, "error": 0}
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(export_one, e, args.refresh) for e in index]
        for i, fut in enumerate(as_completed(futures), 1):
            status, path = fut.result()
            key = "error" if status.startswith("error") else status
            counts[key] += 1
            if key == "error":
                errors.append(f"{path}: {status}")
            if i % 100 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)}  saved={counts['saved']} skipped={counts['skipped']} errors={counts['error']}")

    for line in errors:
        print("ERROR", line, file=sys.stderr)
    print(f"done -> {OUT_DIR}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
