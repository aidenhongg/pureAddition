"""Fixed character-level tokenizer for arithmetic CoT."""

from __future__ import annotations

import json
from pathlib import Path

CHARS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "+", "-", "=", " ", "\n", "c", "b", "N", "E", "G",
    "[PAD]",
]


class CharTokenizer:
    """Character-level tokenizer with a fixed vocabulary for arithmetic CoT."""

    def __init__(self):
        self._c2i = {c: i for i, c in enumerate(CHARS)}
        self._i2c = dict(enumerate(CHARS))
        self.pad_id = self._c2i["[PAD]"]

    @property
    def n_vocab(self) -> int:
        return len(CHARS)

    def encode(self, text: str) -> list[int]:
        return [self._c2i[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._i2c[i] for i in ids if i != self.pad_id)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump({"chars": CHARS}, f)

    def load(self, path: str | Path) -> CharTokenizer:
        return self


def build_tokenizer() -> CharTokenizer:
    return CharTokenizer()


def get_tokenizer(vocab_path: str | Path | None = None) -> CharTokenizer:
    return CharTokenizer()
