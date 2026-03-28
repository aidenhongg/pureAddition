"""Decoder-only transformer with RoPE and Chain-of-Thought generation support."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

IGNORE_INDEX = -100


# ── RoPE ─────────────────────────────────────────────────────────────────────


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return x * cos + _rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE sin/cos tables."""

    def __init__(self, head_dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[Tensor, Tensor]:
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


# ── Transformer components ───────────────────────────────────────────────────


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.resid_drop = nn.Dropout(dropout)
        self.dropout = dropout

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        drop_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop_p)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with RoPE attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


# ── Model ────────────────────────────────────────────────────────────────────


class AdditionLM(nn.Module):
    """Decoder-only transformer LM with RoPE for arithmetic CoT reasoning."""

    def __init__(
        self,
        vocab_size: int = 21,
        d_model: int = 320,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 1280,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, idx: Tensor) -> Tensor:
        B, T = idx.shape
        cos, sin = self.rope(T)

        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.lm_head(self.ln_f(x))

    def compute_loss(self, idx: Tensor, targets: Tensor) -> Tensor:
        """Cross-entropy with prompt masking (targets=IGNORE_INDEX are ignored)."""
        logits = self(idx)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=IGNORE_INDEX,
        )

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        eos_token: int | None = None,
    ) -> Tensor:
        """Autoregressively generate tokens (greedy when temperature ~ 0)."""
        self.eval()
        amp_enabled = idx.is_cuda
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.max_seq_len :]
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = self(idx_crop)[:, -1, :]
            if temperature < 1e-8:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits.float() / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
            if eos_token is not None and (next_tok == eos_token).all():
                break
        return idx

    def param_groups(self, weight_decay: float) -> list[dict]:
        """Separate params into decay (dim >= 2) and no-decay groups."""
        decay, no_decay = [], []
        for p in self.parameters():
            (decay if p.dim() >= 2 else no_decay).append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

