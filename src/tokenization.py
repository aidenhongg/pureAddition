"""WordPiece tokenizer via HuggingFace tokenizers."""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

DIGIT_TOKENS = [str(d) for d in range(10)]


class WordPieceTokenizer:
    """WordPiece tokenizer trained from a text corpus."""

    def __init__(self, vocab_size: int = 512):
        self._vocab_size = vocab_size
        self._tokenizer: Tokenizer | None = None

    @property
    def n_vocab(self) -> int:
        return self._tokenizer.get_vocab_size() if self._tokenizer else 0

    def fit(self, texts: list[str]) -> WordPieceTokenizer:
        tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.Whitespace(),
        ])
        tok.decoder = decoders.WordPiece()
        trainer = trainers.WordPieceTrainer(
            vocab_size=self._vocab_size,
            special_tokens=["[UNK]", "[PAD]"] + DIGIT_TOKENS,
        )
        tok.train_from_iterator(texts, trainer=trainer)
        self._tokenizer = tok
        return self

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids)

    def save(self, path: str | Path) -> None:
        self._tokenizer.save(str(path))

    def load(self, path: str | Path) -> WordPieceTokenizer:
        self._tokenizer = Tokenizer.from_file(str(path))
        return self


def build_tokenizer(texts: list[str], vocab_size: int = 512) -> WordPieceTokenizer:
    return WordPieceTokenizer(vocab_size).fit(texts)


def get_tokenizer(vocab_path: str | Path | None = None) -> WordPieceTokenizer:
    tok = WordPieceTokenizer()
    if vocab_path is not None and Path(vocab_path).exists():
        tok.load(vocab_path)
    return tok


if __name__ == "__main__":
    enc = get_tokenizer()
    print(f"Vocab size: {enc.n_vocab}")
