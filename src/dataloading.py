"""Chain-of-Thought data generation and dataset for arithmetic."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.model import IGNORE_INDEX


# ── CoT formatting ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CoTExample:
    """A single arithmetic problem with step-by-step reasoning."""

    prompt: str
    reasoning: str
    answer: str

    @property
    def full_text(self) -> str:
        return self.prompt + self.reasoning + self.answer + "[EOS]"


class CoTFormatter:
    """Generates digit-by-digit reasoning traces for addition and subtraction.

    Example output for ``123 + 456``::

        123 + 456
        3+6+0=9 c0
        2+5+0=7 c0
        1+4+0=5 c0
        = 579
    """

    @staticmethod
    def format(a: int, b: int, op: str) -> CoTExample:
        if a < 0 or b < 0:
            raise ValueError("Operands must be non-negative")
        if op == "+":
            return CoTFormatter._addition(a, b)
        if op == "-":
            return CoTFormatter._subtraction(a, b)
        raise ValueError(f"Unsupported operator: {op!r}")

    # ── public helpers for direct use ────────────────────────────────────

    @staticmethod
    def _addition(a: int, b: int) -> CoTExample:
        result = a + b
        steps = CoTFormatter._add_digit_steps(a, b)
        return CoTExample(
            prompt=f"{a} + {b}\n",
            reasoning="\n".join(steps) + "\n",
            answer=f"= {result}",
        )

    @staticmethod
    def _subtraction(a: int, b: int) -> CoTExample:
        result = a - b
        if a >= b:
            steps = CoTFormatter._sub_digit_steps(a, b)
        else:
            # |a| < |b|  →  negate the unsigned result
            steps = ["NEG"] + CoTFormatter._sub_digit_steps(b, a)
        return CoTExample(
            prompt=f"{a} - {b}\n",
            reasoning="\n".join(steps) + "\n",
            answer=f"= {result}",
        )

    # ── digit-level step generators ──────────────────────────────────────

    @staticmethod
    def _add_digit_steps(a: int, b: int) -> list[str]:
        """Right-to-left column addition with carry tracking."""
        sa, sb = str(a), str(b)
        width = max(len(sa), len(sb))
        sa, sb = sa.zfill(width), sb.zfill(width)

        carry = 0
        steps: list[str] = []
        for i in range(width - 1, -1, -1):
            da, db = int(sa[i]), int(sb[i])
            total = da + db + carry
            digit = total % 10
            new_carry = total // 10
            steps.append(f"{da}+{db}+{carry}={digit} c{new_carry}")
            carry = new_carry
        if carry:
            steps.append(f"c{carry}")
        return steps

    @staticmethod
    def _sub_digit_steps(a: int, b: int) -> list[str]:
        """Right-to-left column subtraction with borrow tracking.  Requires a >= b."""
        sa, sb = str(a), str(b)
        width = len(sa)
        sb = sb.zfill(width)

        borrow = 0
        steps: list[str] = []
        for i in range(width - 1, -1, -1):
            da, db = int(sa[i]), int(sb[i])
            diff = da - db - borrow
            new_borrow = 0
            if diff < 0:
                diff += 10
                new_borrow = 1
            steps.append(f"{da}-{db}-{borrow}={diff} b{new_borrow}")
            borrow = new_borrow
        return steps


def _tokenize_pair(
    enc, prompt: str, answer: str, max_seq_len: int
) -> tuple[Tensor, Tensor] | None:
    """Tokenize a (prompt, answer) pair with prompt-masking on the target."""
    full_tokens = enc.encode(prompt + answer)
    if len(full_tokens) > max_seq_len:
        return None
    prompt_len = len(enc.encode(prompt))
    tokens = torch.tensor(full_tokens, dtype=torch.long)
    inp = tokens[:-1]
    tgt = tokens[1:].clone()
    if prompt_len > 1:
        tgt[: prompt_len - 1] = IGNORE_INDEX
    return inp, tgt


# ── Dataset ──────────────────────────────────────────────────────────────────

Pool = list[tuple[Tensor, Tensor]]


def _rand_with_digits(rng: random.Random, d: int) -> int:
    """Return a random non-negative integer with exactly *d* digits."""
    if d == 1:
        return rng.randint(0, 9)
    return rng.randint(10 ** (d - 1), 10 ** d - 1)


def _generate_equation_pairs(
    n: int, max_digits: int, seed: int, enc, max_seq_len: int,
) -> Pool:
    """Generate *n* fresh tokenized equation pairs from a seeded PRNG."""
    rng = random.Random(seed)
    pairs: Pool = []
    while len(pairs) < n:
        a = _rand_with_digits(rng, rng.randint(1, max_digits))
        b = _rand_with_digits(rng, rng.randint(1, max_digits))
        op = rng.choice(["+", "-"])
        ex = CoTFormatter.format(a, b, op)
        pair = _tokenize_pair(enc, ex.prompt, ex.reasoning + ex.answer + "[EOS]", max_seq_len)
        if pair:
            pairs.append(pair)
    return pairs


def build_val_set(
    n: int, max_digits: int, seed: int, enc, max_seq_len: int,
) -> "EpochDataset":
    """Build a fixed validation set of synthetic equations."""
    return EpochDataset(_generate_equation_pairs(n, max_digits, seed, enc, max_seq_len))


def sample_epoch(
    epoch_size: int, epoch_seed: int, enc, max_seq_len: int, max_digits: int,
) -> "EpochDataset":
    """Build one epoch's dataset of fresh synthetic equations."""
    items = _generate_equation_pairs(epoch_size, max_digits, epoch_seed, enc, max_seq_len)
    random.Random(epoch_seed).shuffle(items)
    return EpochDataset(items)


class EpochDataset(Dataset):
    """Thin wrapper around a list of tokenized (input, target) pairs."""

    def __init__(self, items: Pool):
        self.inputs = [x[0] for x in items]
        self.targets = [x[1] for x in items]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.inputs[idx], self.targets[idx]


def collate_cot(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """Pad variable-length CoT examples into a uniform batch.

    Padding positions in ``targets`` are set to ``IGNORE_INDEX`` so they are
    ignored by the loss.
    """
    inputs, targets = zip(*batch)
    max_len = max(x.size(0) for x in inputs)

    padded_inputs = torch.zeros(len(inputs), max_len, dtype=torch.long)
    padded_targets = torch.full(
        (len(targets), max_len), IGNORE_INDEX, dtype=torch.long
    )

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        length = inp.size(0)
        padded_inputs[i, :length] = inp
        padded_targets[i, :length] = tgt

    return padded_inputs, padded_targets
