"""Run 5 test queries against the best.pt checkpoint."""

import json
from pathlib import Path

import torch

from src.model import AdditionLM
from src.tokenization import get_tokenizer

CKPT_DIR = Path("src") / "checkpoints"


def _latest_run() -> Path:
    """Return the most recently created run_* directory."""
    runs = sorted(CKPT_DIR.glob("run_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No run directories found in {CKPT_DIR}")
    return runs[-1]


def load_model(device: torch.device) -> tuple[AdditionLM, object]:
    run_dir = _latest_run()
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)

    enc = get_tokenizer(run_dir / "vocab.json")
    cfg["vocab_size"] = enc.n_vocab

    model = AdditionLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
    ).to(device)

    model.load_state_dict(
        torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    )
    model.eval()
    return model, enc


def run_query(model: AdditionLM, enc, prompt: str, device: torch.device) -> str:
    ids = enc.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    output = model.generate(idx, max_new_tokens=128, temperature=0.0)
    return enc.decode(output[0].tolist())


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, enc = load_model(device)
    print(f"Loaded best.pt ({model.param_count():,} params)\n")

    queries = [
        "123 + 456\n",
        "999 + 1\n",
        "500 - 123\n",
        "1 - 200\n",
        "9999 + 9999\n",
    ]

    for i, prompt in enumerate(queries, 1):
        result = run_query(model, enc, prompt, device)
        print(f"Query {i}: {prompt.strip()}")
        print(f"Output:  {result}")
        print("-" * 40)


if __name__ == "__main__":
    main()
