"""Generate README figures for pureAddition."""

import sys
import os
import math

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)

# Style
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#0d1117",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})

ACCENT = "#58a6ff"
ACCENT2 = "#f78166"
ACCENT3 = "#7ee787"
ACCENT4 = "#d2a8ff"


# ── Figure 1: Architecture diagram (parameter breakdown) ─────────────────────

def fig_architecture():
    components = [
        "Token\nEmbedding", "Attention\n(per block)", "MLP\n(per block)",
        "LayerNorm\n(per block)", "Final\nLayerNorm", "LM Head"
    ]
    params = [7040, 410880, 820800, 1280, 640, 7062]
    colors = [ACCENT4, ACCENT, ACCENT2, ACCENT3, ACCENT3, ACCENT4]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(components, params, color=colors, edgecolor="#30363d", linewidth=0.5)

    for bar, val in zip(bars, params):
        label = f"{val:,}"
        ax.text(bar.get_width() + 8000, bar.get_y() + bar.get_height()/2,
                label, va="center", fontsize=10, color="#c9d1d9")

    ax.set_xlabel("Parameters")
    ax.set_title("Parameter Distribution per Component")
    ax.set_xlim(0, max(params) * 1.2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    ax.grid(axis="x")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "param_breakdown.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  param_breakdown.png")


# ── Figure 2: CoT sequence length distribution ──────────────────────────────

def fig_seq_length_dist():
    from src.dataloading import CoTFormatter
    import random

    rng = random.Random(42)
    lengths = []
    digit_counts = {d: [] for d in range(1, 20)}

    for _ in range(50000):
        d1 = rng.randint(1, 19)
        d2 = rng.randint(1, 19)
        a = rng.randint(10**(d1-1) if d1 > 1 else 0, 10**d1 - 1)
        b = rng.randint(10**(d2-1) if d2 > 1 else 0, 10**d2 - 1)
        op = rng.choice(["+", "-"])
        ex = CoTFormatter.format(a, b, op)
        seq_len = len(ex.full_text)
        lengths.append(seq_len)
        max_d = max(d1, d2)
        if max_d in digit_counts:
            digit_counts[max_d].append(seq_len)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of sequence lengths
    ax1.hist(lengths, bins=60, color=ACCENT, edgecolor="#0d1117", alpha=0.85)
    ax1.set_xlabel("Sequence Length (chars)")
    ax1.set_ylabel("Count")
    ax1.set_title("CoT Sequence Length Distribution")
    ax1.axvline(np.median(lengths), color=ACCENT2, linestyle="--", label=f"Median: {np.median(lengths):.0f}")
    ax1.axvline(np.mean(lengths), color=ACCENT3, linestyle="--", label=f"Mean: {np.mean(lengths):.0f}")
    ax1.legend(facecolor="#161b22", edgecolor="#30363d")
    ax1.grid(axis="y")

    # Sequence length vs max operand digits
    digits = sorted([d for d in digit_counts if len(digit_counts[d]) > 10])
    means = [np.mean(digit_counts[d]) for d in digits]
    stds = [np.std(digit_counts[d]) for d in digits]

    ax2.errorbar(digits, means, yerr=stds, fmt="o-", color=ACCENT, ecolor=ACCENT2,
                 elinewidth=1, capsize=3, markersize=5)
    ax2.set_xlabel("Max Operand Digits")
    ax2.set_ylabel("Sequence Length (chars)")
    ax2.set_title("Sequence Length vs Operand Size")
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "seq_length_dist.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  seq_length_dist.png")


# ── Figure 3: Learning rate schedule ────────────────────────────────────────

def fig_lr_schedule():
    warmup = 100
    epoch_size = 100000
    batch_size = 32
    max_epochs = 100
    steps_per_epoch = math.ceil(epoch_size / batch_size)
    total_steps = steps_per_epoch * max_epochs
    base_lr = 1e-3

    steps = np.arange(total_steps)
    lrs = []
    for s in steps:
        if s < warmup:
            lr_mult = s / max(1, warmup)
        else:
            progress = (s - warmup) / max(1, total_steps - warmup)
            lr_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
        lrs.append(base_lr * lr_mult)

    fig, ax = plt.subplots(figsize=(10, 4))
    # Downsample for plotting
    step_idx = np.linspace(0, len(steps)-1, 2000, dtype=int)
    ax.plot(steps[step_idx], np.array(lrs)[step_idx], color=ACCENT, linewidth=1.5)
    ax.axvline(warmup, color=ACCENT2, linestyle="--", alpha=0.7, label=f"Warmup ends (step {warmup})")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule: Linear Warmup + Cosine Decay")
    ax.set_xlim(0, total_steps)
    ax.legend(facecolor="#161b22", edgecolor="#30363d")
    ax.grid(True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "lr_schedule.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  lr_schedule.png")


# ── Figure 4: CoT reasoning trace visualization ─────────────────────────────

def fig_cot_trace():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Chain-of-Thought Reasoning Trace", pad=20)

    # Problem
    ax.text(0.5, 9.2, "Input:", fontsize=12, fontweight="bold", color=ACCENT)
    ax.text(2.0, 9.2, "4827 + 593", fontsize=14, fontfamily="monospace", color="#c9d1d9")

    # Steps
    steps = [
        ("Step 1 (ones):", "7 + 3 + 0 = 0", "carry 1", 7.8),
        ("Step 2 (tens):", "2 + 9 + 1 = 2", "carry 1", 6.6),
        ("Step 3 (hundreds):", "8 + 5 + 1 = 4", "carry 1", 5.4),
        ("Step 4 (thousands):", "4 + 0 + 1 = 5", "carry 0", 4.2),
    ]

    for label, calc, carry, y in steps:
        ax.text(0.5, y, label, fontsize=10, color=ACCENT3)
        ax.text(3.0, y, calc, fontsize=12, fontfamily="monospace", color="#c9d1d9")
        carry_color = ACCENT2 if "1" in carry else "#8b949e"
        ax.text(7.0, y, carry, fontsize=10, color=carry_color, style="italic")

    # Arrow
    ax.annotate("", xy=(5, 3.0), xytext=(5, 3.6),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))

    # Result
    ax.add_patch(plt.Rectangle((1.5, 1.8), 7, 1.2, fill=True,
                                facecolor="#161b22", edgecolor=ACCENT, linewidth=2,
                                joinstyle="round"))
    ax.text(2.0, 2.2, "Output:", fontsize=12, fontweight="bold", color=ACCENT)
    ax.text(4.0, 2.2, "= 5420", fontsize=16, fontfamily="monospace", color=ACCENT3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cot_trace.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  cot_trace.png")


# ── Figure 5: Token vocabulary visualization ─────────────────────────────────

def fig_vocab():
    from src.tokenization import CHARS

    categories = {
        "Digits": [c for c in CHARS if c.isdigit()],
        "Operators": [c for c in CHARS if c in "+-="],
        "Whitespace": [c for c in CHARS if c in " \n"],
        "CoT Symbols": [c for c in CHARS if c in "cbNEG"],
        "Special": [c for c in CHARS if c.startswith("[")],
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    cat_names = list(categories.keys())
    cat_sizes = [len(v) for v in categories.values()]
    colors_list = [ACCENT, ACCENT2, ACCENT3, ACCENT4, "#8b949e"]

    bars = ax.barh(cat_names, cat_sizes, color=colors_list, edgecolor="#30363d", linewidth=0.5)

    for bar, size, cat in zip(bars, cat_sizes, cat_names):
        tokens = categories[cat]
        token_str = ", ".join(repr(t) if t in " \n" else t for t in tokens)
        if len(token_str) > 40:
            token_str = token_str[:40] + "..."
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f"{size}  ({token_str})", va="center", fontsize=9, color="#8b949e")

    ax.set_xlabel("Number of Tokens")
    ax.set_title(f"Character-Level Vocabulary ({len(CHARS)} tokens)")
    ax.set_xlim(0, max(cat_sizes) * 2.5)
    ax.grid(axis="x")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "vocab.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  vocab.png")


# ── Figure 6: Carry propagation complexity ───────────────────────────────────

def fig_carry_complexity():
    """Show how carry chains scale with operand size — the core difficulty."""
    import random

    rng = random.Random(42)
    digit_range = range(1, 20)
    avg_carries = []
    max_carries = []

    for d in digit_range:
        carries = []
        for _ in range(5000):
            a = rng.randint(10**(d-1) if d > 1 else 0, 10**d - 1)
            b = rng.randint(10**(d-1) if d > 1 else 0, 10**d - 1)
            # Count carry chain
            sa, sb = str(a).zfill(d), str(b).zfill(d)
            carry = 0
            carry_count = 0
            for i in range(d - 1, -1, -1):
                total = int(sa[i]) + int(sb[i]) + carry
                carry = total // 10
                if carry:
                    carry_count += 1
            carries.append(carry_count)
        avg_carries.append(np.mean(carries))
        max_carries.append(np.max(carries))

    fig, ax = plt.subplots(figsize=(10, 5))
    digits = list(digit_range)
    ax.fill_between(digits, avg_carries, max_carries, alpha=0.15, color=ACCENT)
    ax.plot(digits, avg_carries, "o-", color=ACCENT, linewidth=2, markersize=5, label="Mean carries")
    ax.plot(digits, max_carries, "s--", color=ACCENT2, linewidth=1.5, markersize=4, label="Max carries")
    ax.plot(digits, [d * 0.5 for d in digits], ":", color=ACCENT3, linewidth=1.5, label="Theoretical ~d/2")

    ax.set_xlabel("Number of Digits per Operand")
    ax.set_ylabel("Carry Operations")
    ax.set_title("Carry Chain Complexity vs Operand Size")
    ax.legend(facecolor="#161b22", edgecolor="#30363d")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "carry_complexity.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  carry_complexity.png")


if __name__ == "__main__":
    print("Generating figures...")
    fig_architecture()
    fig_seq_length_dist()
    fig_lr_schedule()
    fig_cot_trace()
    fig_vocab()
    fig_carry_complexity()
    print("Done.")
