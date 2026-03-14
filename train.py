#!/usr/bin/env python3
"""
OpenAI-ES vs Adam on MNIST  —  GPU-vectorized, H200-optimized
==============================================================
Fixed topology: 784 → ReLU(H) → 10  |  4 hidden sizes × 5 seeds

Speed strategy
--------------
Naive ES loops over pop members in Python → very slow.
Here the entire population is evaluated in **one batched einsum** per gen:
  thetas : (pop, D)
  X      : (N, 784)
  h      : relu( einsum('ni,phi->pnh', X, W1) + b1 )   → (pop, N, H)
  out    :       einsum('pnh,poh->pno', h, W2) + b2    → (pop, N, 10)
  acc    : (out.argmax(-1) == y).mean(-1)               → (pop,)

This turns the inner ES loop into pure tensor ops — no Python overhead,
fully pipelined on the GPU.  Also compiled with torch.compile.

Install: pip install torch torchvision pandas matplotlib
"""

import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_SIZES = [32, 64, 128, 256]
SEEDS        = list(range(5))

# ES
ES_POP       = 1000      # antithetic: 500 unique + 500 mirrors
ES_SIGMA     = 0.05      # perturbation std
ES_LR        = 0.05      # Adam-ES learning rate
ES_GENS      = 500       # generations per run
ES_EVAL_N    = 10_000    # training samples used for fitness (set 0 → 60k)

# GD
GD_EPOCHS    = 30
GD_LR        = 1e-3
GD_BATCH     = 512

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "./data"

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_mnist():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tf)
    test  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    return train, test


def to_device_tensors(dataset, n=None, device=DEVICE):
    """Load (subset of) a dataset into a single flat GPU tensor pair."""
    idxs   = list(range(n if n else len(dataset)))
    loader = DataLoader(
        torch.utils.data.Subset(dataset, idxs),
        batch_size=8192, shuffle=False, num_workers=4, pin_memory=True,
    )
    Xs, ys = [], []
    for x, y in loader:
        Xs.append(x.view(x.size(0), -1))
        ys.append(y)
    X = torch.cat(Xs).to(device)
    y = torch.cat(ys).to(device)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Shared 2-layer FFN
# ─────────────────────────────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def total_params(hidden: int) -> int:
    return hidden * 784 + hidden + 10 * hidden + 10


def flat_init(hidden: int, seed: int, device=DEVICE) -> torch.Tensor:
    """Kaiming-uniform init matching nn.Linear default."""
    torch.manual_seed(seed)
    model = FFN(hidden)
    return torch.cat([p.data.ravel() for p in model.parameters()]).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# GPU-vectorized population evaluation  (the hot path)
# ─────────────────────────────────────────────────────────────────────────────

def make_eval_fn(hidden: int):
    """
    Returns a torch.compiled function:
        eval_pop(thetas, X, y) -> (pop,) accuracy tensor

    thetas : (pop, D)   all perturbed weight vectors on GPU
    X      : (N, 784)   data on GPU
    y      : (N,)       labels on GPU
    """
    @torch.no_grad()
    def _eval_pop(thetas: torch.Tensor,
                  X:      torch.Tensor,
                  y:      torch.Tensor) -> torch.Tensor:
        pop = thetas.shape[0]

        # Unpack flat → layer tensors
        i  = 0
        W1 = thetas[:, i : i + hidden * 784].view(pop, hidden, 784); i += hidden * 784
        b1 = thetas[:, i : i + hidden];                               i += hidden
        W2 = thetas[:, i : i + 10 * hidden].view(pop, 10, hidden);   i += 10 * hidden
        b2 = thetas[:, i : i + 10]                                    # (pop, 10)

        # Batched forward: entire population in two matrix ops
        # h : (pop, N, hidden)
        h   = torch.relu(torch.einsum("ni,phi->pnh", X, W1) + b1[:, None, :])
        # out: (pop, N, 10)
        out = torch.einsum("pnh,poh->pno", h, W2) + b2[:, None, :]

        # accuracy (pop,)
        return (out.argmax(-1) == y[None, :]).float().mean(-1)

    return torch.compile(_eval_pop, mode="reduce-overhead")


# ─────────────────────────────────────────────────────────────────────────────
# Rank normalization
# ─────────────────────────────────────────────────────────────────────────────

def rank_normalize(f: torch.Tensor) -> torch.Tensor:
    """Map (pop,) fitnesses to [-0.5, 0.5] via rank."""
    n    = f.shape[0]
    rank = torch.empty_like(f)
    rank[f.argsort()] = torch.arange(n, device=f.device, dtype=f.dtype)
    return rank / (n - 1) - 0.5


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-ES  (pure torch, no Python loop in the inner step)
# ─────────────────────────────────────────────────────────────────────────────

def run_es(hidden: int, seed: int,
           X_fit: torch.Tensor, y_fit: torch.Tensor,
           X_test: torch.Tensor, y_test: torch.Tensor,
           es_gens: int, es_pop: int) -> dict:

    torch.manual_seed(seed)
    gen   = torch.Generator(device=DEVICE).manual_seed(seed)
    D     = total_params(hidden)
    theta = flat_init(hidden, seed)         # (D,)  on GPU

    eval_pop = make_eval_fn(hidden)

    # Adam state (GPU tensors)
    m1     = torch.zeros(D, device=DEVICE)
    m2     = torch.zeros(D, device=DEVICE)
    b1_a, b2_a, eps_a = 0.9, 0.999, 1e-8

    n_half = es_pop // 2   # antithetic sampling

    acc_curve = []
    t0 = time.time()

    for g in range(es_gens):
        # Antithetic perturbations: (es_pop, D)
        eps_half = torch.randn(n_half, D, device=DEVICE, generator=gen)
        eps      = torch.cat([eps_half, -eps_half], dim=0)

        thetas_p = theta.unsqueeze(0) + ES_SIGMA * eps             # (pop, D)

        # Entire population in one GPU call
        F = eval_pop(thetas_p, X_fit, y_fit)                       # (pop,)
        F = rank_normalize(F)

        # ES gradient estimate
        grad = (eps.T @ F) / (es_pop * ES_SIGMA)                   # (D,)

        # Adam update
        t    = g + 1
        m1   = b1_a * m1 + (1 - b1_a) * grad
        m2   = b2_a * m2 + (1 - b2_a) * grad ** 2
        m1h  = m1 / (1 - b1_a ** t)
        m2h  = m2 / (1 - b2_a ** t)
        theta = theta + ES_LR * m1h / (m2h.sqrt() + eps_a)

        # Logging (cheap: reuse eval_pop on single theta)
        test_acc = eval_pop(theta.unsqueeze(0), X_test, y_test).item()
        acc_curve.append(test_acc)

        if (g + 1) % 50 == 0 or g == 0:
            print(f"    gen {g+1:3d}/{es_gens}  test={test_acc:.4f}")

    elapsed   = time.time() - t0
    train_acc = eval_pop(theta.unsqueeze(0), X_fit, y_fit).item()

    # Save model — unpack flat theta back into an FFN state dict
    save_model = FFN(hidden)
    idx = 0
    for p in save_model.parameters():
        n = p.numel()
        p.data.copy_(theta[idx : idx + n].cpu().reshape(p.shape))
        idx += n
    save_path = f"es_h{hidden}_s{seed}.pt"
    torch.save(save_model.state_dict(), save_path)

    return dict(
        train_acc=train_acc,
        test_acc=acc_curve[-1],
        time_s=elapsed,
        acc_curve=acc_curve,
        save_path=save_path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Adam / GD baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_gd(hidden: int, seed: int,
           X_train: torch.Tensor, y_train: torch.Tensor,
           X_test:  torch.Tensor, y_test:  torch.Tensor,
           gd_epochs: int) -> dict:
    """
    GD with data already on GPU — no DataLoader, no CPU<->GPU transfers,
    just manual index shuffling on GPU tensors. Same speed class as ES.
    """
    torch.manual_seed(seed)
    N = X_train.shape[0]

    model   = torch.compile(FFN(hidden).to(DEVICE), mode="reduce-overhead")
    opt     = optim.Adam(model.parameters(), lr=GD_LR)
    loss_fn = nn.CrossEntropyLoss()

    @torch.no_grad()
    def eval_acc(X, y):
        model.eval()
        return (model(X).argmax(1) == y).float().mean().item()

    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    acc_curve = []
    t0 = time.time()

    for epoch in range(gd_epochs):
        model.train()
        perm = torch.randperm(N, device=DEVICE, generator=gen)
        for start in range(0, N, GD_BATCH):
            idx = perm[start : start + GD_BATCH]
            opt.zero_grad(set_to_none=True)
            loss_fn(model(X_train[idx]), y_train[idx]).backward()
            opt.step()
        acc_curve.append(eval_acc(X_train, y_train))

    elapsed  = time.time() - t0
    test_acc = eval_acc(X_test, y_test)

    save_path = f"gd_h{hidden}_s{seed}.pt"
    # torch.compile wraps the module — unwrap before saving
    torch.save(model._orig_mod.state_dict(), save_path)

    return dict(
        train_acc=acc_curve[-1],
        test_acc=test_acc,
        time_s=elapsed,
        acc_curve=acc_curve,
        save_path=save_path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(df, es_curves, gd_curves, hidden_sizes):
    COLORS = {32: "#E63946", 64: "#F4A261", 128: "#457B9D", 256: "#2A9D8F"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("OpenAI-ES vs Adam (GD) on MNIST — fixed topology", fontsize=13)

    ax  = axes[0]
    summary = (df.groupby(["method", "hidden"])["test_acc"]
               .agg(mean="mean", std="std").reset_index())
    xs, w = np.arange(len(hidden_sizes)), 0.35
    for offset, method in zip([-w/2, w/2], ["ES", "GD"]):
        g = summary[summary.method == method].set_index("hidden")
        ax.bar(xs + offset,
               [g.loc[h, "mean"] for h in hidden_sizes], w,
               yerr=[g.loc[h, "std"] for h in hidden_sizes],
               capsize=4, label=method, alpha=0.85)
    ax.set_xticks(xs); ax.set_xticklabels([f"H={h}" for h in hidden_sizes])
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Final Test Accuracy (mean±std, 5 seeds)")
    ax.set_ylim(0, 1.0); ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    for hidden, curves in es_curves.items():
        c = COLORS[hidden]
        for curve in curves:
            ax.plot(curve, color=c, alpha=0.2, linewidth=0.8)
        ax.plot(np.mean(curves, axis=0), color=c, linewidth=2.0, label=f"H={hidden}")
    ax.set_xlabel("Generation"); ax.set_ylabel("Test Accuracy")
    ax.set_title("ES — Test Accuracy vs Generation")
    ax.set_ylim(0, 1.0); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[2]
    for hidden, curves in gd_curves.items():
        c = COLORS[hidden]
        for curve in curves:
            ax.plot(curve, color=c, alpha=0.2, linewidth=0.8)
        ax.plot(np.mean(curves, axis=0), color=c, linewidth=2.0, label=f"H={hidden}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Train Accuracy")
    ax.set_title("GD — Train Accuracy vs Epoch")
    ax.set_ylim(0, 1.0); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results_plot.png", dpi=150)
    print("Plot saved → results_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="1 hidden, 1 seed, 20 ES gens, 3 GD epochs")
    p.add_argument("--es-eval-n", type=int, default=ES_EVAL_N,
                   help=f"Fitness eval subset size (0 = full 60k, default {ES_EVAL_N})")
    p.add_argument("--es-pop",    type=int, default=ES_POP)
    p.add_argument("--es-gens",   type=int, default=ES_GENS)
    p.add_argument("--gd-epochs", type=int, default=GD_EPOCHS)
    return p.parse_args()


def main():
    args = parse_args()

    hidden_sizes = HIDDEN_SIZES
    seeds        = SEEDS
    es_gens      = args.es_gens
    es_eval_n    = args.es_eval_n if args.es_eval_n > 0 else None
    gd_epochs    = args.gd_epochs
    es_pop       = args.es_pop

    if args.quick:
        hidden_sizes = [64]
        seeds        = [0]
        es_gens      = 20
        es_eval_n    = 2000
        gd_epochs    = 3
        es_pop       = 200
        print("── Quick mode ──────────────────────────────────────────────")

    print(f"Device    : {DEVICE}")
    print(f"Hidden    : {hidden_sizes}")
    print(f"Seeds     : {seeds}")
    print(f"ES        : {es_gens} gens, pop={es_pop}, "
          f"σ={ES_SIGMA}, lr={ES_LR}, eval_N={es_eval_n or '60k (full)'}")
    print(f"GD        : {gd_epochs} epochs, Adam lr={GD_LR}, batch={GD_BATCH}")

    print("\nLoading MNIST …")
    train_ds, test_ds = load_mnist()

    print(f"Moving data to {DEVICE} …")
    X_train, y_train = to_device_tensors(train_ds)          # full 60k for GD
    X_test,  y_test  = to_device_tensors(test_ds)
    # ES uses a fixed random subset for fast fitness evaluation
    X_fit = X_train[:es_eval_n] if es_eval_n else X_train
    y_fit = y_train[:es_eval_n] if es_eval_n else y_train
    print(f"  train : {tuple(X_train.shape)}  "
          f"ES fitness : {tuple(X_fit.shape)}  "
          f"test : {tuple(X_test.shape)}")

    rows      = []
    es_curves = {h: [] for h in hidden_sizes}
    gd_curves = {h: [] for h in hidden_sizes}
    n_runs    = len(hidden_sizes) * len(seeds) * 2
    run_idx   = 0

    for hidden in hidden_sizes:
        for seed in seeds:
            run_idx += 1
            print(f"\n[{run_idx}/{n_runs}]  H={hidden}  seed={seed}  ── ES ──")
            er = run_es(hidden, seed, X_fit, y_fit, X_test, y_test, es_gens, es_pop)
            print(f"  DONE  test={er['test_acc']:.4f}  "
                  f"time={er['time_s']:.1f}s  "
                  f"({er['time_s']/es_gens*1000:.1f} ms/gen)  "
                  f"saved → {er['save_path']}")
            rows.append(dict(method="ES", hidden=hidden, seed=seed,
                             train_acc=er["train_acc"], test_acc=er["test_acc"],
                             time_s=er["time_s"]))
            es_curves[hidden].append(er["acc_curve"])

            run_idx += 1
            print(f"[{run_idx}/{n_runs}]  H={hidden}  seed={seed}  ── GD ──")
            gr = run_gd(hidden, seed, X_train, y_train, X_test, y_test, gd_epochs)
            print(f"  DONE  test={gr['test_acc']:.4f}  "
                  f"time={gr['time_s']:.1f}s  "
                  f"saved → {gr['save_path']}")
            rows.append(dict(method="GD", hidden=hidden, seed=seed,
                             train_acc=gr["train_acc"], test_acc=gr["test_acc"],
                             time_s=gr["time_s"]))
            gd_curves[hidden].append(gr["acc_curve"])

    df = pd.DataFrame(rows)
    df.to_csv("results.csv", index=False)
    print("\nResults → results.csv")

    print("\n" + "═" * 64)
    print("  SUMMARY — mean ± std test accuracy (5 seeds)")
    print("═" * 64)
    summary = (df.groupby(["method", "hidden"])["test_acc"]
               .agg(mean="mean", std="std").reset_index())
    for h in hidden_sizes:
        em = summary[(summary.method == "ES") & (summary.hidden == h)].iloc[0]
        gm = summary[(summary.method == "GD") & (summary.hidden == h)].iloc[0]
        print(f"  H={h:<4d}  ES {em['mean']:.4f}±{em['std']:.4f}"
              f"   GD {gm['mean']:.4f}±{gm['std']:.4f}"
              f"   Δ(GD−ES)={gm['mean']-em['mean']:+.4f}")
    print("═" * 64)

    plot_results(df, es_curves, gd_curves, hidden_sizes)


if __name__ == "__main__":
    main()