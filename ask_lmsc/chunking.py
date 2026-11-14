"""Structure-aware chunking for lovemesomecoding.com posts (lesson 3).

Rules, each fixing something lesson 2 got wrong:
  1. Split at headings (h2/h3), so a chunk is one section about one thing.
  2. Never split inside a code block, and rebuild code text exactly: the page wraps every token in a
     Prism <span>, and naive tag-stripping turned `System.out.println(` into `System . out . println (`.
  3. Drop the "Copy" buttons the site adds to code blocks.
  4. Prefix every chunk with category, post title and section heading. Lesson 2 matched a TypeScript
     post for "Python decorator" because no chunk said which language it was about.
  5. Skip stub posts with almost no text.

The exported HTML is the rendered page, where code is properly escaped, so an HTML parser is safe
here. (On the raw WordPress bodies it was not: they had unescaped <script> inside code samples.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag

TARGET_CHARS = 1200  # aim: a section, or a slice of a long one
MIN_POST_CHARS = 200  # below this a post is a stub ("Source code on Github")
# bge-small reads at most 512 tokens and silently drops the rest. Code runs ~3 chars/token, so a
# code block longer than this is split at line breaks rather than losing its tail.
MAX_CODE_CHARS = 1400

HEADINGS = {"h1", "h2", "h3", "h4"}
TEXT_BLOCKS = {"p", "li", "blockquote", "figcaption", "dt", "dd"}


@dataclass
class Chunk:
    url: str
    title: str
    category: str
    heading: str
    body: str

    @property
    def text(self) -> str:
        """What gets embedded and shown to the model: context line, then the content."""
        where = f"{self.title} > {self.heading}" if self.heading else self.title
        return f"[{self.category}] {where}\n\n{self.body}"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_code(code: str) -> list[str]:
    """Whole block if it fits; otherwise line-aligned pieces, preferring blank lines as cut points."""
    if not code.strip():
        return []
    if len(code) <= MAX_CODE_CHARS:
        return [code]
    pieces, current = [], []
    for line in code.split("\n"):
        size = sum(len(x) + 1 for x in current)
        at_blank = not line.strip()
        if current and (size + len(line) > MAX_CODE_CHARS or (at_blank and size > MAX_CODE_CHARS * 0.6)):
            pieces.append("\n".join(current).strip("\n"))
            current = []
        current.append(line)
    if "\n".join(current).strip():
        pieces.append("\n".join(current).strip("\n"))
    return pieces


def _blocks(node: Tag):
    """Yield ("heading", text) / ("text", text) / ("code", text) in document order.

    Recurses through wrapper <div>s (142 posts nest content inside WordPress layout divs) but stops
    at the first block-level element, so a paragraph is never emitted twice.
    """
    for child in node.children:
        if isinstance(child, NavigableString):
            if _clean(child):
                yield "text", _clean(child)
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name
        if name == "button":  # the "Copy" button on every code block
            continue
        if name in HEADINGS:
            yield "heading", _clean(child.get_text(" "))
        elif name == "pre":
            lang = next((c[9:] for c in child.get("class", []) if c.startswith("language-")), "")
            # get_text() with no separator joins the Prism spans back into the original code.
            code = child.get_text().strip("\n")
            for piece in _split_code(code):
                yield "code", f"```{lang}\n{piece}\n```"
        elif name in TEXT_BLOCKS:
            prefix = "- " if name == "li" else ""
            # A <li> can hold a nested list; take only its own text, the nested items come next.
            if name == "li":
                own = "".join(
                    c.get_text(" ") if isinstance(c, Tag) and c.name not in ("ul", "ol") else str(c)
                    for c in child.children
                    if not (isinstance(c, Tag) and c.name in ("ul", "ol"))
                )
                if _clean(own):
                    yield "text", prefix + _clean(own)
                for sub in child.find_all(["ul", "ol"], recursive=False):
                    yield from _blocks(sub)
            elif _clean(child.get_text(" ")):
                yield "text", _clean(child.get_text(" "))
        elif name == "table":
            rows = [
                " | ".join(_clean(c.get_text(" ")) for c in tr.find_all(["th", "td"]))
                for tr in child.find_all("tr")
            ]
            if rows:
                yield "text", "\n".join(rows)
        else:  # div, section, ul, ol, figure, ... : look inside
            yield from _blocks(child)


def chunk_post(post: dict) -> list[Chunk]:
    soup = BeautifulSoup(post["html"], "html.parser")
    if len(_clean(soup.get_text(" "))) < MIN_POST_CHARS:
        return []

    chunks: list[Chunk] = []
    heading, parts, size = "", [], 0

    def flush() -> None:
        nonlocal parts, size
        if parts:
            chunks.append(Chunk(post["url"], post["title"], post["category"], heading, "\n\n".join(parts)))
        parts, size = [], 0

    for kind, text in _blocks(soup):
        if kind == "heading":
            flush()
            heading = text
            continue
        # Start a new chunk at a block boundary once this one is full. A single block bigger than the
        # target (a long code sample) still goes in whole: half a function is worse than a big chunk.
        if parts and size + len(text) > TARGET_CHARS:
            flush()
        parts.append(text)
        size += len(text)
    flush()
    return chunks
