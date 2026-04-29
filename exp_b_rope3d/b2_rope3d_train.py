"""
Experiment B2: RoPE-2D vs RoPE-3D head-to-head on a length-extrapolation task.

A small GPT-style decoder is trained from scratch on a synthetic copy task.
The two variants differ only in the positional encoding:

  - RoPE-2D: standard, head_dim split into d/2 pairs, each rotated by R(m*theta_k).
  - RoPE-3D: head_dim split into d/3 triplets, each rotated by exp(m*M_k) in
    SO(3). |M_k| follows the same RoPE frequency schedule; the axis direction
    is a deterministic "rotational sweep" across triplets so each triplet
    senses a different SO(3) generator.

Task: "repeat after delimiter".
    sequence = [R_1, R_2, ..., R_L, SEP, R_1, R_2, ..., R_L]
The loss is cross-entropy on the post-SEP positions only. Position is
strictly necessary -- without RoPE the task is unlearnable. We train at
L=32 (total length 65) and evaluate at L=32 (in-distribution) and at
L=64, 96, 128 (out-of-distribution length extrapolation).

Hardware: GPU. Should complete in <10 minutes on RTX 4090.
"""

import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR.parent / "figures"
RES_DIR = OUT_DIR.parent / "results"

# ---------------------------------------------------------------------------
# RoPE-2D (canonical) and RoPE-3D (framework antisymmetric M tensor).
# ---------------------------------------------------------------------------

def build_rope2d_freqs(head_dim: int, max_pos: int, base: float = 10000.0,
                       device="cpu"):
    """Cache cos/sin tables for RoPE-2D.

    Returns (cos, sin) of shape (max_pos, head_dim/2).
    """
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_pos, device=device).float()
    freqs = pos[:, None] * inv_freq[None, :]   # (P, d/2)
    return freqs.cos(), freqs.sin()


def apply_rope2d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, d). cos, sin: (T, d/2)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]   # (B, H, T, d/2)
    cos = cos[None, None, :x.shape[2], :]
    sin = sin[None, None, :x.shape[2], :]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = torch.stack([rot1, rot2], dim=-1).flatten(-2)
    return out


def build_rope3d_M(head_dim: int, base: float = 10000.0, device="cpu"):
    """Construct one antisymmetric M per triplet.

    Frequency magnitudes follow the same RoPE schedule. The axis direction
    rotates through three canonical SO(3) generators across triplets, so the
    first triplet rotates about z, the next about x, the next about y, etc.
    This guarantees the third rotational DoF is engaged across head channels.

    Returns Ms: (n_triplets, 3, 3).
    """
    assert head_dim % 3 == 0
    n_trip = head_dim // 3
    inv_freq = 1.0 / (base ** (torch.arange(0, n_trip, device=device).float() * 2.0 / head_dim))
    Ms = torch.zeros(n_trip, 3, 3, device=device)
    # Axis rotates through (-z, -x, -y) cyclically.
    # Convention from B1: M_xy in M[0,1] gives rotation about -z (eq. 10).
    # We pick antisymmetric structures such that |M_k| = inv_freq[k].
    for k in range(n_trip):
        axis = k % 3
        m = float(inv_freq[k])
        if axis == 0:        # rotate about -z
            Ms[k, 0, 1], Ms[k, 1, 0] = m, -m
        elif axis == 1:      # rotate about -x
            Ms[k, 1, 2], Ms[k, 2, 1] = m, -m
        else:                # rotate about -y
            Ms[k, 0, 2], Ms[k, 2, 0] = m, -m
    return Ms


def build_rope3d_rotations(Ms: torch.Tensor, max_pos: int) -> torch.Tensor:
    """Pre-compute R(m) = exp(m * M_k) for m in [0, max_pos), k in [0, n_trip).

    Uses Rodrigues' formula: for axis-magnitude m_scalar, R = I + sin(m*ms)/ms * M
    + (1-cos(m*ms))/ms^2 * M @ M. Returns (max_pos, n_trip, 3, 3).
    """
    n_trip = Ms.shape[0]
    device = Ms.device
    # |M_k| per triplet (each Ms[k] has only one nonzero pair, so this is just
    # the absolute value of the nonzero entry).
    m_scalars = torch.zeros(n_trip, device=device)
    for k in range(n_trip):
        m_scalars[k] = Ms[k].abs().max()

    pos = torch.arange(max_pos, device=device).float()             # (P,)
    angles = pos[:, None] * m_scalars[None, :]                     # (P, n_trip)
    sin_a = torch.sin(angles)
    cos_a = torch.cos(angles)

    # Normalized M_k = M_k / |M_k| (so |M_unit_k| = 1).
    Ms_unit = Ms / m_scalars[:, None, None].clamp(min=1e-30)
    Ms_unit_sq = torch.einsum("kij,kjl->kil", Ms_unit, Ms_unit)    # (n_trip, 3, 3)

    eye = torch.eye(3, device=device).expand(max_pos, n_trip, 3, 3)
    Rs = (eye
          + sin_a[..., None, None] * Ms_unit[None, :, :, :]
          + (1 - cos_a)[..., None, None] * Ms_unit_sq[None, :, :, :])
    return Rs   # (max_pos, n_trip, 3, 3)


def apply_rope3d(x: torch.Tensor, Rs: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, d). Rs: (P, n_trip, 3, 3) with d = 3 * n_trip and P >= T."""
    B, H, T, d = x.shape
    n_trip = d // 3
    x_grouped = x.view(B, H, T, n_trip, 3)
    R = Rs[:T]                                                     # (T, n_trip, 3, 3)
    # einsum: out[b,h,t,k,i] = R[t,k,i,j] * x[b,h,t,k,j]
    out = torch.einsum("tkij,bhtkj->bhtki", R, x_grouped)
    return out.reshape(B, H, T, d)


# ---------------------------------------------------------------------------
# Tiny GPT.
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int, rope_kind: str,
                 max_pos: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_kind = rope_kind
        self.qkv = nn.Linear(d_model, 3 * n_heads * head_dim, bias=False)
        self.o = nn.Linear(n_heads * head_dim, d_model, bias=False)
        if rope_kind == "rope2d":
            cos, sin = build_rope2d_freqs(head_dim, max_pos)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        elif rope_kind == "rope3d":
            Ms = build_rope3d_M(head_dim)
            Rs = build_rope3d_rotations(Ms, max_pos)
            self.register_buffer("Rs", Rs, persistent=False)
        else:
            raise ValueError(rope_kind)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]    # (B, H, T, d)
        if self.rope_kind == "rope2d":
            q = apply_rope2d(q, self.cos, self.sin)
            k = apply_rope2d(k, self.cos, self.sin)
        else:
            q = apply_rope3d(q, self.Rs)
            k = apply_rope3d(k, self.Rs)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(att.transpose(1, 2).reshape(B, T, -1))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, rope_kind, max_pos):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, head_dim, rope_kind, max_pos)
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, head_dim, n_layers,
                 rope_kind, max_pos):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, head_dim, rope_kind, max_pos)
            for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight   # tied

    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))


# ---------------------------------------------------------------------------
# Synthetic task: repeat after delimiter.
# ---------------------------------------------------------------------------

VOCAB_SIZE = 32      # 0..29 random tokens, 30 = SEP, 31 = PAD
SEP = 30
PAD = 31


def make_batch(batch_size: int, L: int, device: str):
    """Generate (input, target_mask) for repeat-after-delimiter task.

    Input:  R_1, R_2, ..., R_L, SEP, R_1, R_2, ..., R_{L-1}
    Target: shifted by 1 -- we predict R_{i+1} given R_1..R_i and the delimiter.
    Loss is computed only on positions that come after SEP (the repeat half).
    """
    # Random tokens drawn from 0..VOCAB_SIZE-3 (exclude SEP, PAD).
    R = torch.randint(0, VOCAB_SIZE - 2, (batch_size, L), device=device)
    sep_col = torch.full((batch_size, 1), SEP, dtype=torch.long, device=device)
    inp = torch.cat([R, sep_col, R], dim=1)              # (B, 2L+1)

    targets = inp[:, 1:].contiguous()
    inputs  = inp[:, :-1].contiguous()

    # Mask: only count positions corresponding to the repeat half.
    # In `inputs`, positions [L, ..., 2L-1] are the repeat tokens; the model
    # predicts them from earlier context. Their corresponding target positions
    # in `targets` are [L, ..., 2L-1] (shifted-left input).
    mask = torch.zeros_like(targets, dtype=torch.bool)
    mask[:, L:2 * L] = True
    return inputs, targets, mask


def evaluate(model, L: int, n_batches: int, batch_size: int, device: str):
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for _ in range(n_batches):
            inp, tgt, mask = make_batch(batch_size, L, device)
            logits = model(inp)
            loss = F.cross_entropy(
                logits[mask], tgt[mask], reduction="mean"
            )
            preds = logits.argmax(-1)
            acc = (preds[mask] == tgt[mask]).float().mean()
            losses.append(loss.item())
            accs.append(acc.item())
    model.train()
    return float(np.mean(losses)), float(np.mean(accs))


# ---------------------------------------------------------------------------
# Training loop.
# ---------------------------------------------------------------------------

def train_one(rope_kind: str, device: str, max_pos: int, train_L: int = 32,
              steps: int = 4000, lr: float = 3e-4, batch_size: int = 64,
              eval_lengths=(32, 64, 96, 128), eval_every: int = 200):
    print(f"\n=== training {rope_kind} ===")
    model = TinyGPT(
        vocab_size=VOCAB_SIZE,
        d_model=216, n_heads=6, head_dim=36, n_layers=4,
        rope_kind=rope_kind, max_pos=max_pos,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                            betas=(0.9, 0.95))

    history = {"step": [], "train_loss": []}
    eval_history = {f"L={L}": {"step": [], "loss": [], "acc": []} for L in eval_lengths}
    t0 = time.time()
    model.train()
    for step in range(steps + 1):
        inp, tgt, mask = make_batch(batch_size, train_L, device)
        logits = model(inp)
        loss = F.cross_entropy(logits[mask], tgt[mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            history["step"].append(step)
            history["train_loss"].append(float(loss.item()))
        if step % eval_every == 0:
            line = [f"step {step:5d}  train_loss={loss.item():.3f}"]
            for L in eval_lengths:
                el, ea = evaluate(model, L, n_batches=4, batch_size=batch_size, device=device)
                eval_history[f"L={L}"]["step"].append(step)
                eval_history[f"L={L}"]["loss"].append(el)
                eval_history[f"L={L}"]["acc"].append(ea)
                line.append(f"L{L}: loss={el:.3f} acc={ea:.3f}")
            print("  " + " | ".join(line))
    t1 = time.time()
    print(f"  trained in {t1 - t0:.1f}s")
    return history, eval_history


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        torch.cuda.manual_seed_all(0)
    max_pos = 512

    h2d, eh2d = train_one("rope2d", device, max_pos)
    h3d, eh3d = train_one("rope3d", device, max_pos)

    # ---- Plot losses + accuracies ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    eval_lengths = list(eh2d.keys())
    colors = {f"L=32": "#1f77b4", f"L=64": "#ff7f0e", f"L=96": "#2ca02c", f"L=128": "#d62728"}
    for L_key in eval_lengths:
        c = colors.get(L_key, "k")
        axes[0].plot(eh2d[L_key]["step"], eh2d[L_key]["loss"], "-",  c=c, label=f"2D {L_key}")
        axes[0].plot(eh3d[L_key]["step"], eh3d[L_key]["loss"], "--", c=c, label=f"3D {L_key}")
        axes[1].plot(eh2d[L_key]["step"], eh2d[L_key]["acc"], "-",  c=c, label=f"2D {L_key}")
        axes[1].plot(eh3d[L_key]["step"], eh3d[L_key]["acc"], "--", c=c, label=f"3D {L_key}")
    axes[0].set_yscale("log"); axes[0].set_xlabel("step"); axes[0].set_ylabel("eval loss")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("eval accuracy")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.suptitle("B2: RoPE-2D vs RoPE-3D on repeat-after-delimiter (train L=32)")
    fig.tight_layout()
    fig_path = FIG_DIR / "b2_rope_train.png"
    fig.savefig(fig_path, dpi=140)
    print(f"  saved figure: {fig_path}")

    res = {
        "rope2d": {"train": h2d, "eval": eh2d},
        "rope3d": {"train": h3d, "eval": eh3d},
        "config": {"vocab": VOCAB_SIZE, "max_pos": max_pos, "train_L": 32,
                   "eval_lengths": [32, 64, 96, 128]},
    }
    res_path = RES_DIR / "b2_results.json"
    res_path.write_text(json.dumps(res, indent=2))
    print(f"  saved results: {res_path}")

    # Final-step comparison
    print("\n=== Final eval (last logged step) ===")
    print(f"{'length':>8s}  {'rope2d_loss':>12s}  {'rope2d_acc':>10s}  {'rope3d_loss':>12s}  {'rope3d_acc':>10s}")
    for L_key in eval_lengths:
        l2 = eh2d[L_key]["loss"][-1]; a2 = eh2d[L_key]["acc"][-1]
        l3 = eh3d[L_key]["loss"][-1]; a3 = eh3d[L_key]["acc"][-1]
        print(f"  {L_key:>6s}  {l2:>12.4f}  {a2:>10.4f}  {l3:>12.4f}  {a3:>10.4f}")


if __name__ == "__main__":
    main()
