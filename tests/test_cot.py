"""Tests for Chain-of-Thought formatting, dataset, and model loss masking."""

import unittest

import torch

from src.dataloading import CoTFormatter, CoTExample, EpochDataset, collate_cot, _generate_equation_pairs
from src.model import AdditionLM, IGNORE_INDEX


class TestCoTFormatterAddition(unittest.TestCase):
    """Verify digit-by-digit addition traces are arithmetically correct."""

    def test_simple_no_carry(self):
        ex = CoTFormatter.format(123, 456, "+")
        self.assertIsInstance(ex, CoTExample)
        self.assertEqual(ex.prompt, "123 + 456\n")
        self.assertIn("= 579", ex.answer)
        # Three digit columns, no carry
        self.assertIn("3+6+0=9 c0", ex.reasoning)
        self.assertIn("2+5+0=7 c0", ex.reasoning)
        self.assertIn("1+4+0=5 c0", ex.reasoning)

    def test_with_carry(self):
        ex = CoTFormatter.format(199, 1, "+")
        self.assertEqual(ex.answer, "= 200")
        self.assertIn("9+1+0=0 c1", ex.reasoning)
        self.assertIn("9+0+1=0 c1", ex.reasoning)
        self.assertIn("1+0+1=2 c0", ex.reasoning)

    def test_carry_extends_digits(self):
        ex = CoTFormatter.format(999, 1, "+")
        self.assertEqual(ex.answer, "= 1000")
        self.assertIn("c1", ex.reasoning)

    def test_zero_operands(self):
        ex = CoTFormatter.format(0, 0, "+")
        self.assertEqual(ex.answer, "= 0")

    def test_full_text_roundtrip(self):
        ex = CoTFormatter.format(42, 58, "+")
        self.assertTrue(ex.full_text.startswith("42 + 58\n"))
        self.assertTrue(ex.full_text.endswith("= 100"))


class TestCoTFormatterSubtraction(unittest.TestCase):
    """Verify digit-by-digit subtraction traces with borrow handling."""

    def test_simple_no_borrow(self):
        ex = CoTFormatter.format(456, 123, "-")
        self.assertEqual(ex.answer, "= 333")
        self.assertIn("6-3-0=3 b0", ex.reasoning)

    def test_with_borrow(self):
        ex = CoTFormatter.format(200, 1, "-")
        self.assertEqual(ex.answer, "= 199")
        self.assertIn("b1", ex.reasoning)

    def test_negative_result(self):
        ex = CoTFormatter.format(1, 200, "-")
        self.assertEqual(ex.answer, "= -199")
        self.assertIn("NEG", ex.reasoning)

    def test_equal_operands(self):
        ex = CoTFormatter.format(42, 42, "-")
        self.assertEqual(ex.answer, "= 0")


class TestCoTFormatterValidation(unittest.TestCase):
    """Edge cases and input validation."""

    def test_negative_operand_raises(self):
        with self.assertRaises(ValueError):
            CoTFormatter.format(-1, 5, "+")

    def test_unsupported_op_raises(self):
        with self.assertRaises(ValueError):
            CoTFormatter.format(1, 2, "*")

    def test_large_operands(self):
        ex = CoTFormatter.format(999_999, 999_999, "+")
        self.assertEqual(ex.answer, "= 1999998")


def _mock_datasets(**overrides):
    """Minimal datasets dict for testing without network access."""
    defaults = {
        "math_equations": [(10, 5, "+"), (20, 3, "-"), (99, 1, "+")],
        "math_stories": {"train": [
            {"eq_qs": "10 + 5 = ?", "story_1_qs": "A has 10. B gives 5. How many?",
             "story_2_qs": "X has 10 cats. Y gives 5 cats.",
             "story_3_qs": "Z has 10. Gets 5. How many?", "answer": 15},
        ]},
        "stories": {"train": [
            {"text": "Once upon a time there was a little cat who loved to play."},
        ]},
    }
    defaults.update(overrides)
    return defaults


class TestMathCoTDataset(unittest.TestCase):
    """Verify dataset construction and prompt masking."""

    def setUp(self):
        self.ds = MathCoTDataset(
            datasets=_mock_datasets(), max_seq_len=256, seed=0,
        )

    def test_nonempty(self):
        self.assertGreater(len(self.ds), 0)

    def test_item_shapes(self):
        inp, tgt = self.ds[0]
        self.assertEqual(inp.dim(), 1)
        self.assertEqual(tgt.dim(), 1)
        self.assertEqual(inp.shape, tgt.shape)

    def test_prompt_masked(self):
        """At least some leading target tokens should be IGNORE_INDEX."""
        inp, tgt = self.ds[0]
        masked = (tgt == IGNORE_INDEX).sum().item()
        self.assertGreater(masked, 0, "Prompt tokens should be masked")

    def test_has_supervised_tokens(self):
        """Not all targets should be masked."""
        inp, tgt = self.ds[0]
        supervised = (tgt != IGNORE_INDEX).sum().item()
        self.assertGreater(supervised, 0)


class TestCollateCot(unittest.TestCase):
    """Verify batch collation pads correctly."""

    def test_padding(self):
        ds = MathCoTDataset(
            datasets=_mock_datasets(
                math_equations=[(i, i + 1, "+") for i in range(5)],
            ),
            max_seq_len=256, seed=1,
        )
        batch = [ds[i] for i in range(min(4, len(ds)))]
        x, y = collate_cot(batch)

        self.assertEqual(x.dim(), 2)
        self.assertEqual(y.dim(), 2)
        self.assertEqual(x.shape, y.shape)

        # Padding in targets should be IGNORE_INDEX
        for i, (inp, _) in enumerate(batch):
            length = inp.size(0)
            if length < x.size(1):
                self.assertTrue((y[i, length:] == IGNORE_INDEX).all())


class TestModelCoT(unittest.TestCase):
    """Verify model forward, loss, and generation work with CoT data."""

    @classmethod
    def setUpClass(cls):
        cls.model = AdditionLM(
            vocab_size=256, d_model=64, n_heads=2, n_layers=2,
            d_ff=128, max_seq_len=128, dropout=0.0,
        )
        cls.model.eval()

    def test_compute_loss_runs(self):
        x = torch.randint(0, 100, (2, 20))
        y = torch.randint(0, 100, (2, 20))
        y[:, :5] = IGNORE_INDEX  # mask prompt
        loss = self.model.compute_loss(x, y)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0)

    def test_prompt_masking_reduces_loss_scope(self):
        """Loss with masked prompt should differ from unmasked."""
        x = torch.randint(0, 100, (2, 20))
        y_unmasked = torch.randint(0, 100, (2, 20))
        y_masked = y_unmasked.clone()
        y_masked[:, :10] = IGNORE_INDEX
        loss_full = self.model.compute_loss(x, y_unmasked)
        loss_masked = self.model.compute_loss(x, y_masked)
        # They should generally differ since different tokens contribute
        self.assertFalse(torch.isclose(loss_full, loss_masked))

    def test_generate_extends_sequence(self):
        prompt = torch.randint(0, 100, (1, 5))
        out = self.model.generate(prompt, max_new_tokens=10, temperature=1.0)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 15)

    def test_generate_greedy(self):
        prompt = torch.randint(0, 100, (1, 5))
        out1 = self.model.generate(prompt, max_new_tokens=10, temperature=0.0)
        out2 = self.model.generate(prompt, max_new_tokens=10, temperature=0.0)
        self.assertTrue(torch.equal(out1, out2))


if __name__ == "__main__":
    unittest.main()
