"""Fixed character-level tokenizer for arithmetic CoT."""

from __future__ import annotations

import json
from pathlib import Path

CHARS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "+", "-", "=", " ", "\n", "c", "b", "N", "E", "G",
    "[PAD]", "[EOS]",
]


class CharTokenizer:
    """Character-level tokenizer with a fixed vocabulary for arithmetic CoT."""

    def __init__(self):
        self._c2i = {c: i for i, c in enumerate(CHARS)}
        self._i2c = dict(enumerate(CHARS))
        self.pad_id = self._c2i["[PAD]"]
        self.eos_id = self._c2i["[EOS]"]

    @property
    def n_vocab(self) -> int:
        return len(CHARS)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            if text[i] == "[":
                end = text.index("]", i)
                token = text[i : end + 1]
                ids.append(self._c2i[token])
                i = end + 1
            else:
                ids.append(self._c2i[text[i]])
                i += 1
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self._i2c[i] for i in ids if i not in (self.pad_id, self.eos_id))

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump({"chars": CHARS}, f)


def build_tokenizer() -> CharTokenizer:
    """Create a fresh character-level tokenizer."""
    return CharTokenizer()


def get_tokenizer() -> CharTokenizer:
    """Convenience alias for ``build_tokenizer``."""
    return CharTokenizer()
