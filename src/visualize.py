# -*- coding: utf-8 -*-
"""
visualize.py -- Visualisations pour le projet ludus-nn.

4 types de graphiques :
  1. Courbes d'entrainement (loss batch + loss epoch)
  2. Diagramme d'architecture reseau (MLP layers)
  3. Distribution des poids par couche (histogramme)
  4. Scores des coups pour une position donnee (bar chart)

Usage direct :
  python src/visualize.py --history models/training_history.json
  python src/visualize.py --arch --checkpoint models/chess_eval_best.pt
  python src/visualize.py --weights --checkpoint models/chess_eval_best.pt
"""


import json
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # pas besoin de display X11 / Windows

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent))

# ─── Palette Ludus ───────────────────────────────────────────────────────────

COLORS = {
    "bg":       "#0f1117",
    "surface":  "#1a1d27",
    "primary":  "#6c63ff",
    "accent":   "#00d4aa",
    "warning":  "#ffb347",
    "danger":   "#ff6b6b",
    "text":     "#e0e0e0",
    "subtext":  "#888888",
    "grid":     "#2a2d3a",
}


def _apply_theme(fig,ax_list):
    """Applique le theme sombre Ludus a une figure."""
    fig.patch.set_facecolor(COLORS["bg"])
    for ax in (ax_list if hasattr(ax_list, "__iter__") else [ax_list]):
        ax.set_facecolor(COLORS["surface"])
        ax.tick_params(colors=COLORS["subtext"])
        ax.xaxis.label.set_color(COLORS["text"])
        ax.yaxis.label.set_color(COLORS["text"])
        ax.title.set_color(COLORS["text"])
        ax.spines["bottom"].set_color(COLORS["grid"])
        ax.spines["left"].set_color(COLORS["grid"])
        ax.spines["top"].set_color(COLORS["grid"])
        ax.spines["right"].set_color(COLORS["grid"])
        ax.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.7)



# ─── 1. Courbes d'entrainement ────────────────────────────────────────────────

def plot_training_history(history: dict, output_path: str = "models/training_curves.png") -> None:
    """
    Trace les courbes de loss batch et epoch depuis un historique.

    history = {
        "batch_losses":  [[epoch1_batches], [epoch2_batches], ...],
        "epoch_train":   [loss_ep1, loss_ep2, ...],
        "epoch_val":     [loss_ep1, loss_ep2, ...],   # peut etre vide
        "lr_history":    [lr_ep1, lr_ep2, ...]
    }
    """
    has_val   = bool(history.get("epoch_val"))
    has_batch = bool(history.get("batch_losses"))
    has_lr    = bool(history.get("lr_history"))

    n_plots = 2 + (1 if has_lr else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    _apply_theme(fig, axes)

    # ── Courbe batch (loss en temps reel) ─────────────────────────
    ax = axes[0]
    all_batches = []
    for ep_losses in history.get("batch_losses", []):
        all_batches.extend(ep_losses)
    if all_batches:
        x = range(len(all_batches))
        ax.plot(x, all_batches, color=COLORS["primary"], linewidth=0.6, alpha=0.5, label="Batch loss")
        # Moyenne mobile
        window = max(1, len(all_batches) // 50)
        smoothed = np.convolve(all_batches, np.ones(window) / window, mode="valid")
        ax.plot(range(len(smoothed)), smoothed, color=COLORS["accent"], linewidth=1.5, label=f"Moyenne ({window})")
        ax.set_xlabel("Batch global")
        ax.set_ylabel("Loss (BCE)")
        ax.set_title("Loss par batch")
        ax.legend(facecolor=COLORS["surface"], labelcolor=COLORS["text"])

    # ── Courbe epoch ──────────────────────────────────────────────
    ax = axes[1]
    epochs = history.get("epoch_train", [])
    if epochs:
        x = range(1, len(epochs) + 1)
        ax.plot(x, epochs, "o-", color=COLORS["primary"], linewidth=2, markersize=5, label="Train")
        if has_val:
            val = history["epoch_val"]
            ax.plot(x, val, "s--", color=COLORS["accent"], linewidth=2, markersize=5, label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (BCE)")
        ax.set_title("Loss par epoch")
        ax.legend(facecolor=COLORS["surface"], labelcolor=COLORS["text"])
        # Marquer le meilleur
        best_i = int(np.argmin(epochs))
        ax.axvline(best_i + 1, color=COLORS["warning"], linestyle=":", linewidth=1.2, alpha=0.8, label="Meilleur")

    # ── Learning rate ─────────────────────────────────────────────
    if has_lr:
        ax = axes[2]
        lr_hist = history["lr_history"]
        ax.plot(range(1, len(lr_hist) + 1), lr_hist, "o-", color=COLORS["warning"], linewidth=2, markersize=5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")

    fig.suptitle("Ludus Neural Bot — Historique d'entrainement",
                 color=COLORS["text"], fontsize=13, y=1.02)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"[ok] Courbes sauvegardees -> {output_path}")


# ─── 2. Architecture MLP ─────────────────────────────────────────────────────

def plot_architecture(layer_sizes: list[int], output_path: str = "models/architecture.png") -> None:
    """
    Dessine un diagramme de l'architecture MLP.

    layer_sizes : [68, 128, 64, 1]  (inclure input et output)
    """
    fig, ax = plt.subplots(figsize=(max(8, len(layer_sizes) * 2.5), 7))
    _apply_theme(fig, ax)
    ax.set_xlim(-0.5, len(layer_sizes) - 0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axis("off")

    MAX_NEURONS_SHOWN = 8   # au-dela, on dessine des ellipses

    cols = np.linspace(0, len(layer_sizes) - 1, len(layer_sizes))
    layer_names = ["Input"] + [f"Hidden {i}" for i in range(1, len(layer_sizes) - 1)] + ["Output"]
    activations = [""] + ["ReLU"] * (len(layer_sizes) - 2) + ["Linear"]
    colors_layers = (
        [COLORS["accent"]]
        + [COLORS["primary"]] * (len(layer_sizes) - 2)
        + [COLORS["warning"]]
    )

    node_positions = []  # [(x, [y1, y2, ...]), ...]

    for i, (col, size) in enumerate(zip(cols, layer_sizes)):
        shown = min(size, MAX_NEURONS_SHOWN)
        ys = np.linspace(0.1, 0.9, shown)
        node_positions.append((col, ys.tolist()))

        for j, y in enumerate(ys):
            circle = plt.Circle(
                (col, y), 0.04,
                color=colors_layers[i], zorder=3, linewidth=0
            )
            ax.add_patch(circle)

        # Ellipsis si trop de neurons
        if size > MAX_NEURONS_SHOWN:
            ax.text(col, 0.5, "...", color=COLORS["subtext"],
                    ha="center", va="center", fontsize=14, zorder=4)

        # Label
        ax.text(col, -0.1, f"{size}", ha="center", va="top",
                color=COLORS["text"], fontsize=11, fontweight="bold")
        ax.text(col, -0.22, layer_names[i], ha="center", va="top",
                color=COLORS["subtext"], fontsize=8)
        if activations[i]:
            ax.text(col, 1.1, activations[i], ha="center", va="bottom",
                    color=colors_layers[i], fontsize=8, style="italic")

    # Connexions (entre les colonnes adjacentes)
    for i in range(len(layer_sizes) - 1):
        x0, ys0 = node_positions[i]
        x1, ys1 = node_positions[i + 1]
        for y0 in ys0:
            for y1 in ys1:
                ax.plot([x0, x1], [y0, y1],
                        color=COLORS["grid"], linewidth=0.3, alpha=0.4, zorder=1)

    # Legende
    legend_items = [
        mpatches.Patch(color=COLORS["accent"],  label="Entree"),
        mpatches.Patch(color=COLORS["primary"], label="Couche cachee + ReLU"),
        mpatches.Patch(color=COLORS["warning"], label="Sortie (score)"),
    ]
    ax.legend(handles=legend_items, loc="upper right",
              facecolor=COLORS["surface"], labelcolor=COLORS["text"],
              framealpha=0.8, fontsize=8)

    n_params = sum(layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1]
                   for i in range(len(layer_sizes) - 1))
    ax.set_title(
        f"Architecture MLP — {len(layer_sizes)-1} couches  |  {n_params:,} parametres",
        color=COLORS["text"], fontsize=12, pad=20
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"[ok] Architecture sauvegardee -> {output_path}")


# ─── 3. Distribution des poids ────────────────────────────────────────────────

def plot_weight_distributions(model, output_path: str = "models/weight_distributions.png") -> None:
    """
    Histogramme des poids et biais de chaque couche lineaire.
    Utile pour diagnostiquer le gradient vanishing / exploding.
    """
    import torch.nn as nn
    linears = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    n = len(linears)
    if n == 0:
        print("[warn] Aucune couche nn.Linear trouvee.")
        return

    fig, axes = plt.subplots(2, n, figsize=(n * 3.5, 7))
    if n == 1:
        axes = axes.reshape(2, 1)
    _apply_theme(fig, axes.flat)

    for col, (name, layer) in enumerate(linears):
        W = layer.weight.detach().cpu().numpy().flatten()
        b = layer.bias.detach().cpu().numpy().flatten()

        # Poids
        ax_w = axes[0, col]
        ax_w.hist(W, bins=60, color=COLORS["primary"], alpha=0.8, edgecolor="none")
        ax_w.axvline(0, color=COLORS["warning"], linewidth=1, linestyle="--")
        ax_w.set_title(f"{name}\nPoids", color=COLORS["text"], fontsize=9)
        ax_w.set_xlabel(f"mu={W.mean():.3f}  sigma={W.std():.3f}",
                        color=COLORS["subtext"], fontsize=7)

        # Biais
        ax_b = axes[1, col]
        ax_b.hist(b, bins=30, color=COLORS["accent"], alpha=0.8, edgecolor="none")
        ax_b.axvline(0, color=COLORS["warning"], linewidth=1, linestyle="--")
        ax_b.set_title("Biais", color=COLORS["text"], fontsize=9)
        ax_b.set_xlabel(f"mu={b.mean():.3f}  sigma={b.std():.3f}",
                        color=COLORS["subtext"], fontsize=7)

    fig.suptitle("Distribution des poids par couche",
                 color=COLORS["text"], fontsize=12)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"[ok] Distributions sauvegardees -> {output_path}")



# ─── 4. Scores de coups pour une position ────────────────────────────────────

def plot_move_scores(
    scores: dict[str, float],
    title: str = "Scores des coups",
    output_path: str = "models/move_scores.png",
    top_k: int = 15,
) -> None:
    """
    Bar chart des scores de coups pour une position de test.

    scores : {"e2e4": 0.82, "d2d4": 0.75, ...}
    """
    sorted_moves = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    moves  = [m for m, _ in sorted_moves]
    values = [v for _, v in sorted_moves]

    fig, ax = plt.subplots(figsize=(max(7, len(moves) * 0.6), 5))
    _apply_theme(fig, ax)

    bar_colors = [
        COLORS["accent"] if i == 0 else
        COLORS["primary"] if v > 0.5 else
        COLORS["danger"]
        for i, (_, v) in enumerate(sorted_moves)
    ]
    bars = ax.bar(moves, values, color=bar_colors, edgecolor="none", width=0.6)

    # Valeur au dessus de chaque barre
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom",
                color=COLORS["text"], fontsize=8)

    ax.set_xlabel("Coup UCI", fontsize=10)
    ax.set_ylabel("Score du modele")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(min(0, min(values)) - 0.1, max(values) + 0.15)

    # Legende
    ax.bar(0, 0, color=COLORS["accent"],  label="Meilleur coup")
    ax.bar(0, 0, color=COLORS["primary"], label="Score > 0.5")
    ax.bar(0, 0, color=COLORS["danger"],  label="Score <= 0.5")
    ax.legend(facecolor=COLORS["surface"], labelcolor=COLORS["text"], fontsize=8)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"[ok] Scores sauvegardes -> {output_path}")


# ─── Fonction principale (genere tous les graphiques en une fois) ─────────────

def generate_all(
    history_path: str | None = None,
    checkpoint_path: str | None = None,
    model_variant: str = "simple",
    output_dir: str = "models",
) -> None:
    """Genere tous les graphiques disponibles selon les fichiers fournis."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if history_path and Path(history_path).exists():
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
        plot_training_history(history, f"{output_dir}/training_curves.png")

    if checkpoint_path and Path(checkpoint_path).exists():
        import torch
        from model import build_model
        import torch.nn as nn

        model = build_model(model_variant)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()

        # Architecture
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        sizes = [linears[0].in_features] + [l.out_features for l in linears]
        plot_architecture(sizes, f"{output_dir}/architecture.png")

        # Distribution des poids
        plot_weight_distributions(model, f"{output_dir}/weight_distributions.png")

    print(f"\n[ok] Tous les graphiques generes dans {output_dir}/")


# ─── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Visualisations d'entrainement ludus-nn")
    p.add_argument("--history",     help="JSON d'historique (models/training_history.json)")
    p.add_argument("--checkpoint",  help="Checkpoint PyTorch (.pt)")
    p.add_argument("--model",       dest="model_variant", default="simple", choices=["simple","bn"])
    p.add_argument("--output-dir",  default="models")
    p.add_argument("--arch",        action="store_true", help="Diagramme architecture seulement")
    p.add_argument("--weights",     action="store_true", help="Distributions poids seulement")
    args = p.parse_args()

    if not args.history and not args.checkpoint:
        p.print_help()
        sys.exit(0)

    generate_all(
        history_path    = args.history,
        checkpoint_path = args.checkpoint,
        model_variant   = args.model_variant,
        output_dir      = args.output_dir,
    )


