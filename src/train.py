"""Training & evaluation loop for AdditionLM with Chain-of-Thought supervision."""

import json
import logging
import math
import re
import random
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataloading import build_val_set, collate_cot, sample_epoch
from src.model import AdditionLM
from src.tokenization import build_tokenizer, get_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

CKPT_DIR = Path("src") / "checkpoints"


# ── Config ───────────────────────────────────────────────────────────────────


def load_config(path: str = "./config.json") -> dict:
    with open(path) as f:
        return json.load(f)


# ── Scheduler (linear warmup → cosine decay) ────────────────────────────────


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer, warmup: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Early stopping ───────────────────────────────────────────────────────────


class EarlyStopping:
    def __init__(self, patience: int, min_delta : float):
        self.patience = patience
        self.best_loss = float("inf")
        self.min_delta = min_delta
        self.counter = 0
        
    def step(self, val_loss: float) -> bool:
        """Returns True when training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ── Evaluation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_loss(
    model: AdditionLM, loader: DataLoader, device: torch.device
) -> float:
    """Mean cross-entropy over a dataloader (prompt-masked via IGNORE_INDEX)."""
    model.eval()
    amp_enabled = device.type == "cuda"
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            loss = model.compute_loss(x, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


def evaluate_accuracy(
    model: AdditionLM,
    num_samples: int,
    max_operand: int,
    device: torch.device,
    seed: int = 0,
    enc=None,
) -> float:
    """Generate CoT solutions and check whether the final numeric answer is correct."""
    rng = random.Random(seed)
    if enc is None:
        enc = get_tokenizer()
    correct = 0

    for _ in range(num_samples):
        a = rng.randint(0, max_operand)
        b = rng.randint(0, max_operand)
        op = rng.choice(["+", "-"])
        expected = (a + b) if op == "+" else (a - b)

        prompt = f"{a} {op} {b}\n"
        prompt_ids = enc.encode(prompt)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        output_ids = model.generate(idx, max_new_tokens=128, temperature=0.0)
        output_text = enc.decode(output_ids[0].tolist())

        # Extract the final answer: look for "\n= <number>" (the answer line
        # in CoT format) anywhere in the output, taking the last match.
        # Using \n= distinguishes the answer line from inline = in CoT steps.
        matches = re.findall(r'\n=\s*(-?\d+)', output_text)
        if matches and int(matches[-1]) == expected:
            correct += 1

    return correct / max(1, num_samples)


# ── Training ─────────────────────────────────────────────────────────────────


def _log_config(cfg: dict) -> None:
    """Log all hyperparameters and seed for reproducibility."""
    log.info("─── Config ───")
    for k, v in sorted(cfg.items()):
        log.info("  %-25s %s", k, v)
    log.info("──────────────")


def _log_model_attrs(model: AdditionLM, device: torch.device) -> None:
    """Log model architecture summary."""
    log.info("─── Model ───")
    log.info("  Device:          %s", device)
    log.info("  Parameters:      %s", f"{model.param_count():,}")
    log.info("  Layers:          %d", len(model.blocks))
    log.info("  d_model:         %d", model.d_model)
    log.info("  max_seq_len:     %d", model.max_seq_len)
    log.info("  Vocab size:      %d", model.tok_emb.num_embeddings)
    log.info("─────────────")


def _save_final(
    run_dir: Path,
    model: AdditionLM,
    cfg: dict,
    val_loss: float,
    epoch: int,
    lr: float,
    enc=None,
) -> Path:
    """Save model weights and config into the run directory."""
    torch.save(model.state_dict(), run_dir / "final.pt")
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    if enc is not None:
        enc.save(run_dir / "vocab.json")
    log.info("Saved final.pt + config.json + vocab.json → %s", run_dir)
    return run_dir


def train(cfg: dict) -> tuple[AdditionLM, object]:
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = cfg["seed"]

    _log_config(cfg)

    # ── Data ─────────────────────────────────────────────────────────────
    max_operand = cfg.get("max_operand", 999_999)
    epoch_size = cfg.get("epoch_size", 100_000)
    max_epochs = cfg.get("max_epochs", 100)

    # ── Tokenizer ────────────────────────────────────────────────────────
    enc = build_tokenizer()
    cfg["vocab_size"] = enc.n_vocab
    log.info("Tokenizer: %d vocab (char-level)", enc.n_vocab)

    # ── Fixed validation set ─────────────────────────────────────────────
    val_ds = build_val_set(
        cfg.get("val_size", 5000), max_operand, seed, enc, cfg["max_seq_len"]
    )
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], collate_fn=collate_cot)

    # ── Model ────────────────────────────────────────────────────────────
    model = AdditionLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
    ).to(device)
    _log_model_attrs(model, device)

    # ── Optimizer ────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.param_groups(cfg["weight_decay"]), lr=cfg["lr"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # ── Checkpointing ────────────────────────────────────────────────────
    run_dir = CKPT_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    log.addHandler(fh)

    # ── Training loop ────────────────────────────────────────────────────
    warmup_steps = cfg["warmup_steps"]
    batch_size = cfg["batch_size"]
    eval_samples = cfg.get("eval_samples", 200)
    log_every = cfg.get("log_every_n_batches", 50)

    steps_per_epoch = math.ceil(epoch_size / batch_size)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, steps_per_epoch * max_epochs)
    stopper = EarlyStopping(cfg["patience"], cfg["min_delta"])

    best_val_loss = float("inf")
    val_loss = float("inf")
    epoch = 0

    try:
        for epoch in range(1, max_epochs + 1):
            train_ds = sample_epoch(
                epoch_size, seed + epoch, enc, cfg["max_seq_len"], max_operand,
            )
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_cot
            )

            model.train()
            epoch_loss, n = 0.0, 0

            for batch_idx, (x, y) in enumerate(train_loader, 1):
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    loss = model.compute_loss(x, y)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss * x.size(0)
                n += x.size(0)

                if batch_idx % log_every == 0:
                    log.info(
                        "  E%3d | batch %5d | loss %.4f | lr %.2e",
                        epoch, batch_idx, batch_loss,
                        scheduler.get_last_lr()[0],
                    )

            train_loss = epoch_loss / n
            val_loss = evaluate_loss(model, val_loader, device)
            lr = scheduler.get_last_lr()[0]

            acc = evaluate_accuracy(
                model, eval_samples, max_operand, device,
                seed=epoch, enc=enc,
            )

            log.info(
                "E%3d | train %.4f | val %.4f | lr %.2e | acc %s",
                epoch, train_loss, val_loss, lr, f"{acc:.2%}",
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), run_dir / "best.pt")

            if stopper.step(val_loss):
                log.info("Early stopping at epoch %d", epoch)
                break

    except KeyboardInterrupt:
        log.info("Training interrupted at epoch %d", epoch)
    finally:
        best_pt = run_dir / "best.pt"
        if best_pt.exists():
            model.load_state_dict(torch.load(best_pt, weights_only=True))
        final_lr = scheduler.get_last_lr()[0] if epoch > 0 else cfg["lr"]
        _save_final(run_dir, model, cfg, val_loss=best_val_loss, epoch=epoch, lr=final_lr, enc=enc)
        log.removeHandler(fh)
        fh.close()
    return model, enc


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    train(cfg)
