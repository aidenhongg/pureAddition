"""Chain-of-Thought data generation and dataset for arithmetic."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.model import IGNORE_INDEX
from src.tokenization import get_tokenizer


# ── CoT formatting ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CoTExample:
    """A single arithmetic problem with step-by-step reasoning."""

    prompt: str
    reasoning: str
    answer: str

    @property
    def full_text(self) -> str:
        return self.prompt + self.reasoning + self.answer


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


# ── Analogy helpers ──────────────────────────────────────────────────────────


def _parse_analogy(doc: str) -> tuple[str, str, str, str] | None:
    """Parse 'A : B = C : D' → (A, B, C, D) or None on failure."""
    parts = doc.split(" = ", 1)
    if len(parts) != 2:
        return None
    left = parts[0].split(" : ", 1)
    right = parts[1].split(" : ", 1)
    if len(left) != 2 or len(right) != 2:
        return None
    return left[0].strip(), left[1].strip(), right[0].strip(), right[1].strip()


def _format_analogy_reasoning(a: str, b: str, c: str, d: str) -> tuple[str, str]:
    """Build (prompt, supervised) for an analogy with symbolic reasoning."""
    prompt = f"{c} - {a} + {b} is \n"
    reasoning = (
        f"C + -A + B\n"
        f"(C + -A) + B\n"
        f"({c} + -({a})) + {b}\n"
    )
    return prompt, reasoning + f"= {d}"


def _iter_analogies(datasets: dict):
    """Yield (prompt, supervised) pairs from the hyperprobe analogy dataset."""
    for split in datasets.get("analogies", {}).values():
        for row in split:
            if "math" in (row.get("doc", "") + row.get("test", "") + row.get("domain", "")).lower():
                continue
            parsed = _parse_analogy(row["doc"])
            if parsed is None:
                continue
            a, b, c, d = parsed
            yield _format_analogy_reasoning(a, b, c, d)


# ── Text collection for tokenizer training ───────────────────────────────────


def collect_texts(
    datasets: dict, max_texts: int = 50_000, seed: int = 42
) -> list[str]:
    """Extract raw text from all data sources for tokenizer vocabulary building."""
    texts: list[str] = []
    for a, b, op in datasets.get("math_equations", []):
        ex = CoTFormatter.format(a, b, op)
        texts.append(ex.full_text)
    for split in datasets["math_stories"].values():
        for row in split:
            texts.append(row["story_1_qs"] + "\n" + str(row["answer"]))
    for split in datasets["stories"].values():
        for row in split:
            texts.append(row["text"])
    for prompt, answer in _iter_analogies(datasets):
        texts.append(prompt + answer)
    if len(texts) > max_texts:
        rng = random.Random(seed)
        texts = rng.sample(texts, max_texts)
    return texts


# ── Equation generation ──────────────────────────────────────────────────────


def generate_math_equations(
    n: int, max_operand: int, seed: int
) -> list[tuple[int, int, str]]:
    """Generate *n* random ``(a, b, op)`` tuples using a seeded PRNG."""
    rng = random.Random(seed)
    equations: list[tuple[int, int, str]] = []
    for _ in range(n // 2):
        a = rng.randint(0, max_operand)
        b = rng.randint(0, max_operand)
        equations.append((a, b, "+"))
        equations.append((a, b, "-"))
    return equations


def _augment_addition_story(
    row: dict, rng: random.Random, max_operand: int,
) -> dict | None:
    """Replace numbers in an addition-only math story with new random values."""
    eq_str = row.get("eq_qs", "")
    clean = eq_str.split("=")[0].strip()
    tokens = re.findall(r'\d+|[+\-]', clean)
    if len(tokens) < 3 or any(t == '-' for t in tokens):
        return None
    orig_nums = [t for t in tokens if t != '+']
    new_nums = [str(rng.randint(0, max_operand)) for _ in orig_nums]
    story = row["story_1_qs"]
    for old, new in zip(orig_nums, new_nums):
        story = re.sub(r'\b' + re.escape(old) + r'\b', new, story, count=1)
    return {
        "eq_qs": " + ".join(new_nums) + " = ?",
        "story_1_qs": story,
        "answer": sum(int(n) for n in new_nums),
    }


def _chain_cot_reasoning(eq_str: str) -> str | None:
    """Parse equation like '2385 + 761 - 1063 = ?' and produce chained CoT."""
    clean = eq_str.split("=")[0].strip()
    tokens = re.findall(r'\d+|[+\-]', clean)
    if len(tokens) < 3:
        return None
    try:
        result = int(tokens[0])
    except ValueError:
        return None
    lines = [clean] if len(tokens) > 3 else []
    i = 1
    while i + 1 < len(tokens):
        op = tokens[i]
        if op not in ('+', '-'):
            return None
        try:
            num = int(tokens[i + 1])
        except ValueError:
            return None
        if result < 0 or num < 0:
            return None
        ex = CoTFormatter.format(result, num, op)
        lines.append(ex.prompt.strip())
        lines.append(ex.reasoning.strip())
        result = int(ex.answer.split("=")[1].strip())
        lines.append(f"= {result}")
        i += 2
    return "\n".join(lines) + "\n"


def _tokenize_full(
    enc, text: str, max_seq_len: int
) -> tuple[Tensor, Tensor] | None:
    """Tokenize full text for standard causal LM (no prompt masking). Truncates to max_seq_len."""
    tokens = enc.encode(text)
    if len(tokens) < 2:
        return None
    tokens = tokens[:max_seq_len]
    tokens = torch.tensor(tokens, dtype=torch.long)
    return tokens[:-1], tokens[1:]


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


class MathCoTDataset(Dataset):
    """Tokenised (prompt, reasoning+answer) pairs.

    Sources:
    - ``stories``: causal LM on full text (no prompt masking)
    - ``math_stories``: story questions → chained digit-level CoT reasoning
    - ``analogies``: hyperprobe analogies → symbolic decomposition reasoning

    Each item returns ``(input_ids, target_ids)`` where prompt tokens in
    ``target_ids`` are set to ``IGNORE_INDEX`` so the loss only covers the
    supervised portion.
    """

    def __init__(self, datasets: dict, max_seq_len: int = 512, seed: int = 42, enc=None,
                 max_operand: int = 999_999, n_augments: int = 0):
        if enc is None:
            enc = get_tokenizer()
        rng = random.Random(seed)

        self.inputs: list[Tensor] = []
        self.targets: list[Tensor] = []

        # Tiny stories – standard causal LM (no prompt masking)
        for split in datasets["stories"].values():
            for row in split:
                pair = _tokenize_full(enc, row["text"], max_seq_len)
                if pair:
                    self.inputs.append(pair[0])
                    self.targets.append(pair[1])

        # Math stories – CoT reasoning with prompt masking
        for split in datasets["math_stories"].values():
            for row in split:
                reasoning = _chain_cot_reasoning(row.get("eq_qs", ""))
                supervised = reasoning if reasoning else f"= {row['answer']}"
                pair = _tokenize_pair(enc, row["story_1_qs"] + "\n", supervised, max_seq_len)
                if pair:
                    self.inputs.append(pair[0])
                    self.targets.append(pair[1])
                for _ in range(n_augments):
                    aug = _augment_addition_story(row, rng, max_operand)
                    if aug is None:
                        break
                    aug_reasoning = _chain_cot_reasoning(aug["eq_qs"])
                    aug_sup = aug_reasoning if aug_reasoning else f"= {aug['answer']}"
                    pair = _tokenize_pair(enc, aug["story_1_qs"] + "\n", aug_sup, max_seq_len)
                    if pair:
                        self.inputs.append(pair[0])
                        self.targets.append(pair[1])

        # Analogies – symbolic decomposition reasoning
        for prompt, supervised in _iter_analogies(datasets):
            pair = _tokenize_pair(enc, prompt, supervised, max_seq_len)
            if pair:
                self.inputs.append(pair[0])
                self.targets.append(pair[1])

        indices = list(range(len(self.inputs)))
        rng.shuffle(indices)
        self.inputs = [self.inputs[i] for i in indices]
        self.targets = [self.targets[i] for i in indices]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.inputs[idx], self.targets[idx]


class EquationDataset(Dataset):
    """Synthetic equation dataset regenerated fresh each epoch from a PRNG seed."""

    def __init__(self, n: int, max_operand: int, seed: int, enc, max_seq_len: int = 512):
        rng = random.Random(seed)
        self.inputs: list[Tensor] = []
        self.targets: list[Tensor] = []

        for _ in range(n // 2):
            a = rng.randint(0, max_operand)
            b = rng.randint(0, max_operand)
            for op in ("+", "-"):
                ex = CoTFormatter.format(a, b, op)
                pair = _tokenize_pair(enc, ex.prompt, ex.reasoning + ex.answer, max_seq_len)
                if pair:
                    self.inputs.append(pair[0])
                    self.targets.append(pair[1])

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.inputs[idx], self.targets[idx]


# ── Pool-based data for ladder training ──────────────────────────────────────

Pool = list[tuple[Tensor, Tensor]]


def build_pools(
    datasets: dict, enc, max_seq_len: int, seed: int,
) -> dict[str, Pool | list[dict]]:
    """Pre-tokenize lang/analogy pools; store raw math story rows for on-the-fly augmentation."""
    pools: dict[str, Pool | list[dict]] = {"lang": [], "story_rows": [], "analogy": []}

    for split in datasets["stories"].values():
        for row in split:
            pair = _tokenize_full(enc, row["text"], max_seq_len)
            if pair:
                pools["lang"].append(pair)

    for split in datasets["math_stories"].values():
        for row in split:
            pools["story_rows"].append(dict(row))

    for prompt, supervised in _iter_analogies(datasets):
        pair = _tokenize_pair(enc, prompt, supervised, max_seq_len)
        if pair:
            pools["analogy"].append(pair)

    return pools


def _sample_pool(pool: list, n: int, rng: random.Random) -> list:
    """Sample *n* items from *pool*, with replacement if n > len(pool)."""
    if n <= 0 or not pool:
        return []
    return rng.sample(pool, n) if n <= len(pool) else rng.choices(pool, k=n)


def _tokenize_story_row(
    row: dict, rng: random.Random, max_operand: int, enc, max_seq_len: int,
) -> tuple[Tensor, Tensor] | None:
    """Augment a math story row with fresh random numbers (when possible), then tokenize."""
    aug = _augment_addition_story(row, rng, max_operand)
    src = aug if aug is not None else row
    reasoning = _chain_cot_reasoning(src.get("eq_qs", ""))
    supervised = reasoning if reasoning else f"= {src['answer']}"
    return _tokenize_pair(enc, src["story_1_qs"] + "\n", supervised, max_seq_len)


def _generate_equation_pairs(
    n: int, max_operand: int, seed: int, enc, max_seq_len: int,
) -> Pool:
    """Generate *n* fresh tokenized equation pairs from a seeded PRNG."""
    rng = random.Random(seed)
    pairs: Pool = []
    while len(pairs) < n:
        a = rng.randint(0, max_operand)
        b = rng.randint(0, max_operand)
        op = rng.choice(["+", "-"])
        ex = CoTFormatter.format(a, b, op)
        pair = _tokenize_pair(enc, ex.prompt, ex.reasoning + ex.answer, max_seq_len)
        if pair:
            pairs.append(pair)
    return pairs


def sample_epoch(
    pools: dict, ratios: dict[str, float], epoch_size: int,
    epoch_seed: int, enc, max_seq_len: int, max_operand: int,
) -> "EpochDataset":
    """Build one epoch's dataset by sampling pools according to *ratios*.

    Stories are augmented and tokenized on-the-fly so numbers are fresh each epoch.
    """
    rng = random.Random(epoch_seed)
    n_lang = int(epoch_size * ratios["lang"])
    n_eq = int(epoch_size * ratios["eq"])
    n_story = int(epoch_size * ratios["story"])
    n_analogy = epoch_size - n_lang - n_eq - n_story

    items: Pool = []
    items += _sample_pool(pools["lang"], n_lang, rng)
    items += _sample_pool(pools["analogy"], n_analogy, rng)
    items += _generate_equation_pairs(n_eq, max_operand, epoch_seed, enc, max_seq_len)

    # Stories: sample raw rows, augment with fresh numbers, tokenize
    story_rows = _sample_pool(pools["story_rows"], n_story, rng)
    for row in story_rows:
        pair = _tokenize_story_row(row, rng, max_operand, enc, max_seq_len)
        if pair:
            items.append(pair)

    rng.shuffle(items)
    return EpochDataset(items)


def build_val_set(
    pools: dict[str, Pool], n: int, max_operand: int,
    seed: int, enc, max_seq_len: int,
) -> "EpochDataset":
    """Build a fixed 50/50 (language / equation) validation set."""
    rng = random.Random(seed)
    n_lang = n // 2
    n_eq = n - n_lang
    items = _sample_pool(pools["lang"], n_lang, rng)
    items += _generate_equation_pairs(n_eq, max_operand, seed, enc, max_seq_len)
    rng.shuffle(items)
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
