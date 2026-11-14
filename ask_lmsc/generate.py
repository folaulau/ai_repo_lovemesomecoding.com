"""Answer a question from retrieved sources with Claude (same prompt as lesson 2)."""

import anthropic

MODEL = "claude-opus-5-5"
PRICE_IN, PRICE_OUT = 4, 20  # $ per million tokens for claude-opus-5-5

SYSTEM = (
    "You answer programming questions using only the sources provided. "
    "Cite sources inline as [1], [2]. If the sources do not contain the answer, say so."
)


def answer(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(
        f'<source id="{n}" url="{h["url"]}">\n{h["text"]}\n</source>' for n, h in enumerate(hits, 1)
    )
    response = anthropic.Anthropic().messages.create(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"{context}\n\nQuestion: {question}"}],
    )
    u = response.usage
    print(f"tokens: {u.input_tokens} in, {u.output_tokens} out "
          f"(~${(u.input_tokens * PRICE_IN + u.output_tokens * PRICE_OUT) / 1e6:.4f})")
    if response.stop_reason == "refusal":
        return "(The model declined to answer.)"
    return "".join(b.text for b in response.content if b.type == "text")
