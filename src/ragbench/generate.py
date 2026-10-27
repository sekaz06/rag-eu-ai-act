"""Grounded answer generation with numbered citations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .chunking import Chunk
from .config import Settings

REFUSAL = {
    "en": "I cannot find this information in the provided sources.",
    "fr": "Je ne trouve pas cette information dans les sources fournies.",
}

SYSTEM_PROMPT = {
    "en": f"""You are a legal assistant specialised in the EU Artificial Intelligence Act and the GDPR.
Rules:
1. Answer ONLY from the numbered sources provided.
2. Cite every statement with the source number in square brackets, e.g. [2].
3. If the sources do not contain the answer, reply exactly: "{REFUSAL["en"]}"
4. Answer in English, concisely and precisely. Mention the relevant article whenever possible.""",
    "fr": f"""Tu es un assistant juridique spécialisé dans le règlement européen sur l'IA (AI Act) et le RGPD.
Règles :
1. Réponds UNIQUEMENT à partir des sources numérotées fournies.
2. Cite chaque affirmation avec le numéro de la source entre crochets, par exemple [2].
3. Si les sources ne permettent pas de répondre, réponds exactement : "{REFUSAL["fr"]}"
4. Réponds en français, de façon concise et précise. Mentionne l'article concerné quand c'est possible.""",
}

LABELS = {"en": ("Sources", "Question"), "fr": ("Sources", "Question")}


@dataclass
class Answer:
    text: str
    sources: list[Chunk]


def build_context(chunks: Sequence[Chunk]) -> str:
    return "\n\n".join(f"[{i}] ({c.doc_id} — {c.section})\n{c.text}" for i, c in enumerate(chunks, start=1))


def build_messages(question: str, chunks: Sequence[Chunk], language: str = "en") -> list[dict]:
    sources_label, question_label = LABELS[language]
    user = f"{sources_label}:\n{build_context(chunks)}\n\n{question_label}: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT[language]}, {"role": "user", "content": user}]


def generate_answer(question: str, chunks: Sequence[Chunk], settings: Settings) -> Answer:
    from openai import OpenAI

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=build_messages(question, chunks, settings.language),
        temperature=settings.llm_temperature,
    )
    return Answer(text=resp.choices[0].message.content or "", sources=list(chunks))
