"""Training & evaluation loop for AdditionLM with Chain-of-Thought supervision."""

import json
import logging
import math
import re
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from init import download_datasets
from src.dataloading import MathCoTDataset, collate_cot, collect_texts
from src.model import AdditionLM
from src.tokenization import build_tokenizer, get_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


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
    def __init__(self, patience: int):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        """Returns True when training should stop."""
        if val_loss < self.best_loss:
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

        # Extract the final answer after the last "= "
        match = re.search(r"=\s*(-?\d+)\s*$", output_text)
        if match and int(match.group(1)) == expected:
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
    log.info("  d_emb:           %d", model.tok_emb.embedding_dim)
    log.info("  d_model:         %d", model.d_model)
    log.info("  max_seq_len:     %d", model.max_seq_len)
    log.info("  Vocab size:      %d", model.tok_emb.num_embeddings)
    log.info("─────────────")


def _save_final(
    model: AdditionLM,
    cfg: dict,
    val_loss: float,
    epoch: int,
    lr: float,
    enc=None,
) -> Path:
    """Save model weights and config into a descriptive run directory."""
    run_name = f"loss_{val_loss:.4f}_epochs_{epoch}_lr_{lr:.2e}"
    run_dir = Path("src") / "checkpoints" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), run_dir / "final.pt")
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    if enc is not None:
        enc.save(run_dir / "vocab.json")
    log.info("Saved final.pt + config.json + vocab.json → %s", run_dir)
    return run_dir


def train(cfg: dict) -> AdditionLM:
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _log_config(cfg)

    # ── Data ─────────────────────────────────────────────────────────────
    datasets = download_datasets(
        max_math_stories=cfg.get("max_math_stories", 500_000),
        max_tiny_stories=cfg.get("max_tiny_stories", 500_000),
    )

    # ── Tokenizer ────────────────────────────────────────────────────────
    texts = collect_texts(datasets, max_texts=50_000, seed=cfg["seed"])
    enc = build_tokenizer(texts, vocab_size=cfg["vocab_size"])
    cfg["vocab_size"] = enc.n_vocab
    log.info("Tokenizer: %d vocab (WordPiece)", enc.n_vocab)

    ds = MathCoTDataset(
        datasets=datasets,
        max_seq_len=cfg["max_seq_len"],
        seed=cfg["seed"],
        enc=enc,
    )
    val_size = max(1, int(len(ds) * cfg["val_split"]))
    train_ds, val_ds = random_split(ds, [len(ds) - val_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_cot
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], collate_fn=collate_cot
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = AdditionLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
        d_emb=cfg["d_emb"],
    ).to(device)
    _log_model_attrs(model, device)

    # ── Optimiser & scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.param_groups(cfg["weight_decay"]), lr=cfg["lr"]
    )
    total_steps = len(train_loader) * cfg["max_epochs"]
    scheduler = get_lr_scheduler(optimizer, cfg["warmup_steps"], total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # ── Checkpointing & early stopping ───────────────────────────────────
    tmp_ckpt_dir = Path("src") / "checkpoints" / "_running"
    tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)
    stopper = EarlyStopping(cfg["patience"])

    # ── Loop ─────────────────────────────────────────────────────────────
    eval_every = cfg.get("eval_every", 5)
    eval_samples = cfg.get("eval_samples", 200)
    max_operand = cfg.get("max_operand", 999_999)

    log_every_n_batches = cfg.get("log_every_n_batches", 50)

    for epoch in range(1, cfg["max_epochs"] + 1):
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

            if batch_idx % log_every_n_batches == 0:
                log.info(
                    "  Epoch %3d | batch %5d | loss %.4f | lr %.2e",
                    epoch, batch_idx, batch_loss, scheduler.get_last_lr()[0],
                )

        train_loss = epoch_loss / n
        val_loss = evaluate_loss(model, val_loader, device)
        lr = scheduler.get_last_lr()[0]

        # Periodic exact-match accuracy on freshly generated problems
        acc_str = ""
        if epoch % eval_every == 0:
            acc = evaluate_accuracy(
                model, eval_samples, max_operand, device, seed=epoch, enc=enc
            )
            acc_str = f" | acc {acc:.2%}"

        log.info(
            "Epoch %3d | train %.4f | val %.4f | lr %.2e%s",
            epoch, train_loss, val_loss, lr, acc_str,
        )

        # Checkpoint on best val loss
        if val_loss <= stopper.best_loss:
            torch.save(model.state_dict(), tmp_ckpt_dir / "best.pt")

        if stopper.step(val_loss):
            log.info("Early stopping at epoch %d", epoch)
            break

    # Restore best weights and save into descriptive run directory
    model.load_state_dict(torch.load(tmp_ckpt_dir / "best.pt", weights_only=True))
    final_lr = scheduler.get_last_lr()[0]
    enc.save(tmp_ckpt_dir / "vocab.json")
    _save_final(model, cfg, val_loss=stopper.best_loss, epoch=epoch, lr=final_lr, enc=enc)
    return model, enc


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    train(cfg)
