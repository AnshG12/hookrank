"""Text normalisation and evidence matching used to verify model claims against the script."""
from __future__ import annotations

import re
import unicodedata

_PUNCT_MAP = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"', "‐": "-", "‑": "-",
    "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-", " ": " ",
})
_ELLIPSIS_RE = re.compile(r"\.{3,}|…")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def normalize(text: str) -> str:
    """Casefold, unify Unicode punctuation and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text or "").translate(_PUNCT_MAP)
    return " ".join(text.split()).casefold()


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize(text))


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    it = iter(haystack)
    return all(token in it for token in needle)


def quote_match(quote: str, source: str) -> str | None:
    """How a model quote relates to its cited source text.

    Returns ``"exact"`` (contiguous after normalisation), ``"elided"`` (the quote's words
    all occur in order, i.e. the model dropped a phrase or used an ellipsis), or ``None``.
    Elided matches need at least three words, so short invented fragments cannot pass.
    """
    q_tokens, s_tokens = tokens(quote), tokens(source)
    if not q_tokens:
        return None
    if " ".join(q_tokens) in " ".join(s_tokens):
        return "exact"
    fragments = [tokens(part) for part in _ELLIPSIS_RE.split(quote)]
    fragments = [f for f in fragments if f]
    if len(q_tokens) >= 3 and _is_subsequence(q_tokens, s_tokens):
        return "elided"
    if len(fragments) > 1 and all(" ".join(f) in " ".join(s_tokens) for f in fragments):
        return "elided"
    return None


def name_in_text(name: str, text: str) -> bool:
    """True if every word (3+ letters) of ``name`` appears as a whole word in ``text``."""
    words = [w for w in tokens(name) if len(w) >= 3]
    if not words:
        return False
    haystack: set[str] = set()
    for token in tokens(text):
        haystack.add(token)
        if token.endswith("'s"):
            haystack.add(token[:-2])  # "Juliana's" also attests "Juliana"
    return all(w in haystack for w in words)
