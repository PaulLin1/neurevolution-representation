#!/usr/bin/env python3
"""
Neuroevolution Analysis: ES vs GD on MNIST
==========================================
Analyses inspired by Jeff Clune et al. — measures applied to every saved
es_h{H}_s{S}.pt and gd_h{H}_s{S}.pt checkpoint.

Metrics
-------
1. Modularity (Q)       — how much weight mass concentrates in class-specific
                          sub-circuits (Clune et al. 2013)
2. Regularity           — Kolmogorov proxy: zlib compressibility of weight bytes
3. Effective rank       — dimensionality of the learned weight spaces (Roy & Vetterli)
4. Robustness           — accuracy vs Gaussian weight noise (σ sweep)
5. Dead neurons         — fraction of ReLU neurons silent on entire test set
6. Neuron selectivity   — fractured (class-selective) vs entangled (polysemantic)
                          via a normalised selectivity index per neuron
7. CKA similarity       — Centred Kernel Alignment between ES and GD hidden
                          representations (Kornblith et al. 2019)

Outputs
-------
  analysis_results.csv   — one row per (method, hidden, seed, metric)
  analysis_plot.png      — 3×3 summary figure
"""

import os, glob, warnings
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import zlib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Config — match whatever you used in train.py
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_SIZES    = [32, 64, 128, 256]
SEEDS           = list(range(5))
NOISE_SIGMAS    = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR        = "./data"
N_CLASSES       = 10

COLORS_METHOD   = {"ES": "#E63946", "GD": "#457B9D"}
COLORS_HIDDEN   = {32: "#E63946", 64: "#F4A261", 128: "#457B9D", 256: "#2A9D8F"}


# ─────────────────────────────────────────────────────────────────────────────
# Model definition (must match train.py)
# ─────────────────────────────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def load_model(path: str, hidden: int) -> FFN:
    model = FFN(hidden)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(DEVICE).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_test_data(n=10_000) -> tuple[torch.Tensor, torch.Tensor]:
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    X, y = next(iter(loader))
    X = X.view(X.size(0), -1)[:n].to(DEVICE)
    y = y[:n].to(DEVICE)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 1. Modularity Q
# ─────────────────────────────────────────────────────────────────────────────

def modularity_q(model: FFN) -> float:
    """
    Clune-style modularity over the hidden→output weight matrix W2.

    Each hidden neuron is assigned to the output class it connects to most
    strongly (argmax |W2[:, h]|).  Q is the normalised fraction of weight
    mass that lies on "within-module" connections, baseline-corrected for
    chance:

        raw  = Σ_h |W2[c*(h), h]| / Σ |W2|
        Q    = (raw − 1/K) / (1 − 1/K)     K = number of classes

    Q = 0  → as modular as chance
    Q = 1  → perfectly modular (each neuron exclusively serves one class)
    """
    W2 = model.fc2.weight.data.cpu().numpy()   # (10, H)
    W2_abs = np.abs(W2)
    total  = W2_abs.sum()
    if total == 0:
        return 0.0

    # Dominant class for each hidden neuron
    dominant = W2_abs.argmax(axis=0)           # (H,)
    on_module = sum(W2_abs[dominant[h], h] for h in range(W2_abs.shape[1]))
    raw = on_module / total
    K   = W2_abs.shape[0]
    return (raw - 1.0 / K) / (1.0 - 1.0 / K)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Regularity (Kolmogorov proxy)
# ─────────────────────────────────────────────────────────────────────────────

def regularity(model: FFN) -> dict:
    """
    Compresses each weight matrix with zlib.
    regularity = 1 − compressed_bytes / original_bytes
    Higher → more compressible → more regular / repetitive structure.
    Also returns per-layer values for W1 and W2.
    """
    results = {}
    for name, param in [("W1", model.fc1.weight), ("W2", model.fc2.weight)]:
        arr  = param.data.cpu().numpy().astype(np.float32)
        raw  = arr.tobytes()
        comp = zlib.compress(raw, level=9)
        results[name] = 1.0 - len(comp) / len(raw)

    results["mean"] = np.mean([results["W1"], results["W2"]])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Effective rank  (Roy & Vetterli 2007)
# ─────────────────────────────────────────────────────────────────────────────

def effective_rank(W: np.ndarray) -> float:
    """
    erank(W) = exp( H(σ / ||σ||₁) )
    where σ are the singular values and H is Shannon entropy.
    erank ∈ [1, min(m,n)] — higher means the matrix uses more dimensions.
    """
    sv = np.linalg.svd(W, compute_uv=False)
    sv = sv[sv > 1e-10]
    p  = sv / sv.sum()
    H  = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(H))


def effective_ranks(model: FFN) -> dict:
    W1 = model.fc1.weight.data.cpu().numpy()
    W2 = model.fc2.weight.data.cpu().numpy()
    return {
        "erank_W1": effective_rank(W1),
        "erank_W2": effective_rank(W2),
        "erank_W1_norm": effective_rank(W1) / min(W1.shape),
        "erank_W2_norm": effective_rank(W2) / min(W2.shape),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Robustness (accuracy vs weight noise)
# ─────────────────────────────────────────────────────────────────────────────

def weight_rms(model: FFN) -> float:
    """RMS weight magnitude across all parameters."""
    total_sq = sum(p.data.pow(2).sum().item() for p in model.parameters())
    total_n  = sum(p.numel() for p in model.parameters())
    return float((total_sq / total_n) ** 0.5)


@torch.no_grad()
def robustness_curve(model: FFN, X: torch.Tensor, y: torch.Tensor,
                     sigmas=NOISE_SIGMAS, n_trials=5) -> tuple:
    """
    Returns (abs_curve, rel_curve).

    abs_curve: noise added as absolute N(0, sigma^2) — confounded by weight scale.
    rel_curve: noise scaled by per-parameter RMS, so sigma is SNR-matched.
               sigma_eff = sigma * rms(param) per tensor.

    If ES weights are smaller than GD weights, absolute noise is already a
    larger relative hit on ES — so ES being *more* robust on abs_curve is if
    anything a conservative estimate. rel_curve removes this doubt entirely.
    """
    original = {n: p.data.clone() for n, p in model.named_parameters()}
    abs_accs, rel_accs = [], []

    for sigma in sigmas:
        abs_trials, rel_trials = [], []
        for _ in range(n_trials):
            # absolute noise
            for p in model.parameters():
                p.data += torch.randn_like(p.data) * sigma
            abs_trials.append((model(X).argmax(1) == y).float().mean().item())
            for name, p in model.named_parameters():
                p.data.copy_(original[name])

            # relative noise: sigma * rms(param) per tensor
            for p in model.parameters():
                rms = p.data.pow(2).mean().sqrt().clamp(min=1e-8)
                p.data += torch.randn_like(p.data) * sigma * rms
            rel_trials.append((model(X).argmax(1) == y).float().mean().item())
            for name, p in model.named_parameters():
                p.data.copy_(original[name])

        abs_accs.append(float(np.mean(abs_trials)))
        rel_accs.append(float(np.mean(rel_trials)))

    return abs_accs, rel_accs


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dead neurons
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def dead_neuron_fraction(model: FFN, X: torch.Tensor) -> float:
    """
    Fraction of hidden ReLU neurons that produce zero activation for every
    sample in X.  Dead neurons contribute nothing to the output.
    """
    h = torch.relu(model.fc1(X))              # (N, H)
    active = (h > 0).any(dim=0)               # (H,) — True if ever active
    return 1.0 - active.float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Neuron selectivity — fractured vs entangled
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def selectivity_index(model: FFN, X: torch.Tensor, y: torch.Tensor) -> dict:
    """
    For each hidden neuron, compute its mean activation per class, then:

        SI = (μ_max − μ_mean_others) / (μ_max + μ_mean_others + ε)

    SI ∈ [-1, 1].  SI ≈ 1 → neuron fires only for one class (fractured).
                   SI ≈ 0 → neuron fires uniformly (entangled / polysemantic).

    Returns mean, std, and fraction of neurons with SI > 0.5 (highly selective).
    """
    h = torch.relu(model.fc1(X)).cpu().numpy()   # (N, H)
    y_np = y.cpu().numpy()

    H = h.shape[1]
    class_means = np.zeros((N_CLASSES, H))
    for c in range(N_CLASSES):
        mask = y_np == c
        if mask.sum() > 0:
            class_means[c] = h[mask].mean(axis=0)

    mu_max    = class_means.max(axis=0)          # (H,)
    mu_others = (class_means.sum(axis=0) - mu_max) / (N_CLASSES - 1)
    si        = (mu_max - mu_others) / (mu_max + mu_others + 1e-8)

    return {
        "si_mean":       float(si.mean()),
        "si_std":        float(si.std()),
        "si_frac_high":  float((si > 0.5).mean()),   # highly class-selective
        "si_per_neuron": si,                          # (H,) for distribution plots
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. CKA — representational similarity ES vs GD
# ─────────────────────────────────────────────────────────────────────────────

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA (Kornblith et al. 2019).
    X, Y : (N, D1), (N, D2) — centred hidden representations.
    Returns scalar in [0, 1].  1 = identical geometry.
    """
    def centre(M):
        return M - M.mean(axis=0, keepdims=True)

    X = centre(X)
    Y = centre(Y)

    hsic_xy = np.linalg.norm(Y.T @ X, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2

    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / (denom + 1e-10))


@torch.no_grad()
def get_hidden(model: FFN, X: torch.Tensor) -> np.ndarray:
    return torch.relu(model.fc1(X)).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print("Loading test data …")
    X_test, y_test = load_test_data()

    rows        = []                         # flat metric table
    rob_curves  = {}                         # (method, hidden, seed) → [acc]
    si_dists    = {}                         # (method, hidden, seed) → SI array
    cka_matrix  = {}                         # (hidden, seed) → scalar CKA

    # ── Scan available checkpoints ───────────────────────────────────────────
    all_paths = {}
    for method in ("es", "gd"):
        for h in HIDDEN_SIZES:
            for s in SEEDS:
                path = f"{method}_h{h}_s{s}.pt"
                if os.path.exists(path):
                    all_paths[(method.upper(), h, s)] = path

    if not all_paths:
        print("No checkpoint files found (es_h*_s*.pt / gd_h*_s*.pt).")
        print("Run train.py first.")
        return

    total = len(all_paths)
    print(f"Found {total} checkpoint(s). Running analyses …\n")

    for idx, ((method, hidden, seed), path) in enumerate(sorted(all_paths.items())):
        print(f"[{idx+1}/{total}]  {method}  H={hidden}  seed={seed}  {path}")
        model = load_model(path, hidden)

        # 1. Modularity
        Q = modularity_q(model)

        # 2. Regularity
        reg = regularity(model)

        # 3. Effective rank
        er = effective_ranks(model)

        # 4. Robustness curve — both absolute and relative (scale-controlled)
        wrms = weight_rms(model)
        abs_rob, rel_rob = robustness_curve(model, X_test, y_test)
        rob_curves[(method, hidden, seed)] = (abs_rob, rel_rob)
        _trapz       = getattr(np, "trapezoid", None) or np.trapz
        span         = NOISE_SIGMAS[-1] - NOISE_SIGMAS[0]
        rob_auc_abs  = float(_trapz(abs_rob, NOISE_SIGMAS) / span)
        rob_auc_rel  = float(_trapz(rel_rob, NOISE_SIGMAS) / span)

        # 5. Dead neurons
        dead = dead_neuron_fraction(model, X_test)

        # 6. Selectivity
        si = selectivity_index(model, X_test, y_test)
        si_dists[(method, hidden, seed)] = si["si_per_neuron"]

        # Accuracy at σ=0 (clean)
        with torch.no_grad():
            acc = (model(X_test).argmax(1) == y_test).float().mean().item()

        rows.append(dict(
            method=method, hidden=hidden, seed=seed,
            accuracy=acc,
            modularity_Q=Q,
            regularity_W1=reg["W1"], regularity_W2=reg["W2"],
            regularity_mean=reg["mean"],
            erank_W1=er["erank_W1"], erank_W2=er["erank_W2"],
            erank_W1_norm=er["erank_W1_norm"], erank_W2_norm=er["erank_W2_norm"],
            robustness_auc_abs=rob_auc_abs,
            robustness_auc_rel=rob_auc_rel,
            weight_rms=wrms,
            dead_neuron_frac=dead,
            si_mean=si["si_mean"], si_std=si["si_std"],
            si_frac_high=si["si_frac_high"],
        ))

        print(f"  acc={acc:.4f}  Q={Q:.3f}  reg={reg['mean']:.3f}  "
              f"erank_W1={er['erank_W1']:.1f}  dead={dead:.3f}  "
              f"SI={si['si_mean']:.3f}  wrms={wrms:.4f}  "
              f"rob_abs={rob_auc_abs:.3f}  rob_rel={rob_auc_rel:.3f}")

    # ── CKA: compare ES vs GD representations ───────────────────────────────
    print("\nComputing CKA (ES vs GD) …")
    for h in HIDDEN_SIZES:
        for s in SEEDS:
            p_es = f"es_h{h}_s{s}.pt"
            p_gd = f"gd_h{h}_s{s}.pt"
            if os.path.exists(p_es) and os.path.exists(p_gd):
                m_es = load_model(p_es, h)
                m_gd = load_model(p_gd, h)
                H_es = get_hidden(m_es, X_test)
                H_gd = get_hidden(m_gd, X_test)
                cka  = linear_cka(H_es, H_gd)
                cka_matrix[(h, s)] = cka
                print(f"  H={h}  seed={s}  CKA(ES, GD) = {cka:.4f}")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv("analysis_results.csv", index=False)
    print("\nSaved → analysis_results.csv")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("Generating plots …")
    plot_analysis(df, rob_curves, si_dists, cka_matrix)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def summary_stat(df, metric, method, hidden):
    """Mean and std of a metric for a given (method, hidden) across seeds."""
    sub = df[(df.method == method) & (df.hidden == hidden)][metric]
    return sub.mean(), sub.std()


def grouped_bars(ax, df, metric, ylabel, title, hidden_sizes=HIDDEN_SIZES):
    xs = np.arange(len(hidden_sizes))
    w  = 0.35
    for offset, method in zip([-w/2, w/2], ["ES", "GD"]):
        means, stds = zip(*[summary_stat(df, metric, method, h) for h in hidden_sizes])
        ax.bar(xs + offset, means, w, yerr=stds, capsize=4,
               label=method, color=COLORS_METHOD[method], alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"H={h}" for h in hidden_sizes])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def plot_analysis(df, rob_curves, si_dists, cka_matrix):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("Neuroevolution Analysis: OpenAI-ES vs Adam (GD) on MNIST",
                 fontsize=14, y=1.01)

    # ── 1. Modularity Q ──────────────────────────────────────────────────────
    ax = axes[0, 0]
    grouped_bars(ax, df, "modularity_Q", "Q (Clune modularity)",
                 "1. Modularity Q\n(↑ more class-specific sub-circuits)")
    ax.set_ylim(0, 1)

    # ── 2. Regularity (mean of W1 + W2) ──────────────────────────────────────
    ax = axes[0, 1]
    grouped_bars(ax, df, "regularity_mean", "zlib compressibility",
                 "2. Regularity\n(↑ more compressible / structured weights)")
    ax.set_ylim(0, 1)

    # ── 3. Effective rank W1 (normalised) ────────────────────────────────────
    ax = axes[0, 2]
    grouped_bars(ax, df, "erank_W1_norm", "erank(W1) / min(dim)",
                 "3. Effective rank W1 (normalised)\n(↑ uses more dimensions)")
    ax.set_ylim(0, 1)

    # ── 4. Robustness abs curves ─────────────────────────────────────────────
    ax = axes[1, 0]
    for method in ("ES", "GD"):
        ls = "--" if method == "ES" else "-"
        for h in HIDDEN_SIZES:
            pairs = [rob_curves[(method, h, s)]
                     for s in SEEDS if (method, h, s) in rob_curves]
            if not pairs:
                continue
            mean_abs = np.mean([p[0] for p in pairs], axis=0)
            ax.plot(NOISE_SIGMAS, mean_abs,
                    color=COLORS_HIDDEN[h], linestyle=ls, linewidth=1.8,
                    label=f"{method} H={h}")
    ax.set_xlabel("Weight noise σ (absolute)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("4. Robustness — Absolute noise\n(ES=dashed GD=solid; colour=H)")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)

    # ── 5. Robustness rel curves (scale-corrected) ───────────────────────────
    ax = axes[1, 1]
    for method in ("ES", "GD"):
        ls = "--" if method == "ES" else "-"
        for h in HIDDEN_SIZES:
            pairs = [rob_curves[(method, h, s)]
                     for s in SEEDS if (method, h, s) in rob_curves]
            if not pairs:
                continue
            mean_rel = np.mean([p[1] for p in pairs], axis=0)
            ax.plot(NOISE_SIGMAS, mean_rel,
                    color=COLORS_HIDDEN[h], linestyle=ls, linewidth=1.8,
                    label=f"{method} H={h}")
    ax.set_xlabel("Noise fraction of weight RMS (σ_rel)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("5. Robustness — Scale-normalised noise\n(controls for weight magnitude)")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)

    # ── 6. Weight RMS — checks whether scale confounds robustness ────────────
    ax = axes[1, 2]
    grouped_bars(ax, df, "weight_rms", "RMS weight magnitude",
                 "6. Weight RMS\n(confound check: if ES << GD, abs noise is unfair)")
    ax.set_title("6. Weight RMS\n(confound: abs noise unfair if ES≠GD scale)")

    # ── 7. Dead neurons ───────────────────────────────────────────────────────
    ax = axes[2, 0]
    grouped_bars(ax, df, "dead_neuron_frac", "Fraction of dead neurons",
                 "7. Dead Neurons\n(↑ more ReLU neurons always silent)")
    ax.set_ylim(0, 0.15)

    # ── 8. Selectivity distributions (violin) ────────────────────────────────
    ax = axes[2, 1]
    positions = []
    data_vals = []
    colors_v  = []
    tick_labels = []
    pos = 1
    for method in ("ES", "GD"):
        for h in HIDDEN_SIZES:
            si_all = np.concatenate([
                si_dists[(method, h, s)]
                for s in SEEDS if (method, h, s) in si_dists
            ]) if any((method, h, s) in si_dists for s in SEEDS) else np.array([0.0])
            positions.append(pos)
            data_vals.append(si_all)
            colors_v.append(COLORS_METHOD[method])
            tick_labels.append(f"{method}\nH={h}")
            pos += 1
        pos += 0.5  # gap between methods

    parts = ax.violinplot(data_vals, positions=positions,
                          showmedians=True, showextrema=False)
    for pc, c in zip(parts["bodies"], colors_v):
        pc.set_facecolor(c)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=6)
    ax.set_ylabel("Selectivity index")
    ax.set_title("8. SI Distribution\n(fractured ↑ vs entangled ↓)")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.7, label="SI=0.5")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # ── 9. CKA similarity (ES vs GD) ─────────────────────────────────────────
    ax = axes[2, 2]
    if cka_matrix:
        cka_by_hidden = {h: [] for h in HIDDEN_SIZES}
        for (h, s), val in cka_matrix.items():
            cka_by_hidden[h].append(val)
        means = [np.mean(cka_by_hidden[h]) if cka_by_hidden[h] else np.nan
                 for h in HIDDEN_SIZES]
        stds  = [np.std(cka_by_hidden[h])  if cka_by_hidden[h] else 0.0
                 for h in HIDDEN_SIZES]
        xs = np.arange(len(HIDDEN_SIZES))
        ax.bar(xs, means, yerr=stds, capsize=4,
               color="#6A4C93", alpha=0.85, width=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"H={h}" for h in HIDDEN_SIZES])
        ax.set_ylabel("Linear CKA")
        ax.set_ylim(0, 1)
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_title("9. CKA Similarity\n(ES vs GD hidden representations)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("analysis_plot.png", dpi=150, bbox_inches="tight")
    print("Saved → analysis_plot.png")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  ANALYSIS SUMMARY — mean ± std across seeds")
    print("═" * 80)
    metrics = [
        ("accuracy",             "Accuracy"),
        ("weight_rms",           "Weight RMS"),
        ("modularity_Q",         "Modularity Q"),
        ("regularity_mean",      "Regularity"),
        ("erank_W1_norm",        "Eff.rank W1 (norm)"),
        ("robustness_auc_abs",   "Robustness AUC (abs)"),
        ("robustness_auc_rel",   "Robustness AUC (rel)"),
        ("dead_neuron_frac",     "Dead neurons"),
        ("si_mean",              "SI mean"),
        ("si_frac_high",         "SI frac > 0.5"),
    ]
    hdr = f"{'Metric':<26}" + "".join(
        f"  ES H={h:<4d}   GD H={h:<4d}" for h in HIDDEN_SIZES
    )
    print(hdr)
    print("─" * len(hdr))
    for col, label in metrics:
        row = f"{label:<26}"
        for h in HIDDEN_SIZES:
            for method in ("ES", "GD"):
                m, s = summary_stat(df, col, method, h)
                row += f"  {m:.3f}±{s:.3f}"
        print(row)
    print("═" * 80)


if __name__ == "__main__":
    main()