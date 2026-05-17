# -*- coding: utf-8 -*-
"""
train_nnue.py — Pipeline d'entraînement NNUE pour ludus-nn.

Architecture :
    Input : 768 features binaires (12 pièces × 64 cases)
    FT    : Linear(768, H)  + ClippedReLU → accumulateur [H]
    Head  : Linear(H+4, 64) + ReLU → Linear(64, 1)

Visualisations générées automatiquement :
    models/nnue_training_curves.png    — loss batch + epoch + learning rate
    models/nnue_architecture.png       — diagramme FT / Head
    models/nnue_ft_heatmap.png         — poids FT par type de pièce
    models/nnue_weight_distributions.png
    models/nnue_accumulator.png        — accumulateur de la position initiale
    models/nnue_move_scores.png        — scores des coups (position de test)

Usage :
    python src/train_nnue.py --pgn data/lichess.pgn.zst

Options :
    python src/train_nnue.py --help
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).parent))

from dataset import LichessDatasetInMemory

# Réutilise les fonctions génériques de visualize.py
from visualize import (
    plot_training_history,
    plot_weight_distributions,
    plot_move_scores,
    COLORS,
    _apply_theme,
)

# ─── NNUE Architecture ────────────────────────────────────────────────────────

PIECE_ORDER = ["wK", "wQ", "wR", "wB", "wN", "wP", "bK", "bQ", "bR", "bB", "bN", "bP"]
PIECE_PLANES = {p: i for i, p in enumerate(PIECE_ORDER)}
N_FEATURES = 768  # 12 pièces × 64 cases


def board_to_features(board: list) -> np.ndarray:
    """Encode un plateau en 768 features binaires."""
    out = np.zeros(N_FEATURES, dtype=np.float32)
    for row in range(8):
        for col in range(8):
            try:
                piece = board[row][col]
            except IndexError:
                continue
            if piece in PIECE_PLANES:
                out[PIECE_PLANES[piece] * 64 + row * 8 + col] = 1.0
    return out


def uci_to_enc(uci: str) -> np.ndarray:
    """Encode un coup UCI en 4 floats ∈ [0, 1]."""
    b = uci.encode()
    if len(b) < 4:
        return np.zeros(4, dtype=np.float32)
    return np.array([
        (b[1] - ord('1')) / 7.0,
        (b[0] - ord('a')) / 7.0,
        (b[3] - ord('1')) / 7.0,
        (b[2] - ord('a')) / 7.0,
    ], dtype=np.float32)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class NnueDataset(Dataset):
    def __init__(self, samples):
        self.data = samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board, move, label = self.data[idx]
        return (
            torch.from_numpy(board_to_features(board)),
            torch.from_numpy(uci_to_enc(move)),
            torch.tensor([label], dtype=torch.float32),
        )


# ─── Model ────────────────────────────────────────────────────────────────────

class NnueModel(nn.Module):
    """
    NNUE : FeatureTransformer (768→H ClippedReLU) + Head ((H+4)→64→1).
    """

    def __init__(self, H: int = 64):
        super().__init__()
        self.H = H
        self.ft = nn.Linear(N_FEATURES, H)
        self.head = nn.Sequential(
            nn.Linear(H + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, board_feat: torch.Tensor, move_enc: torch.Tensor) -> torch.Tensor:
        acc = torch.clamp(self.ft(board_feat), 0.0, 1.0)
        return self.head(torch.cat([acc, move_enc], dim=-1))


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATION NNUE — 5 graphiques spécialisés
# ═══════════════════════════════════════════════════════════════════════════════

PIECE_LABELS = ["wK♔", "wQ♕", "wR♖", "wB♗", "wN♘", "wP♙",
                "bK♚", "bQ♛", "bR♜", "bB♝", "bN♞", "bP♟"]


def plot_nnue_architecture(H: int, output_path: str = "models/nnue_architecture.png") -> None:
    """
    Diagramme de l'architecture NNUE en deux blocs :
      Bloc FT  : [768 binary] → ClippedReLU → [H accumulateur]
      Bloc Head: [H + 4]      → ReLU         → [64] → [1]

    Affiche la connexion de move_enc (4 floats) injectée dans le head.
    """
    fig = plt.figure(figsize=(14, 6))
    _apply_theme(fig, fig.get_axes())
    fig.patch.set_facecolor(COLORS["bg"])
    ax = fig.add_subplot(111)
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def _box(x, y, w, h, color, label, sublabel="", alpha=0.9):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, alpha=alpha,
                             edgecolor=COLORS["grid"], linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + 0.15, label, ha="center", va="center",
                color=COLORS["text"], fontsize=10, fontweight="bold", zorder=3)
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.35, sublabel, ha="center", va="center",
                    color=COLORS["subtext"], fontsize=8, zorder=3, style="italic")

    def _arrow(x0, y0, x1, y1, color=None):
        color = color or COLORS["subtext"]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5), zorder=4)

    # ── Colonne 1 : Entrée binaire ────────────────────────────────
    _box(0.2, 1.5, 1.8, 3.0, COLORS["surface"], "768", "Features binaires\n12 pièces × 64 cases",
         alpha=0.7)

    # ── Flèche → FT ───────────────────────────────────────────────
    _arrow(2.0, 3.0, 2.8, 3.0, COLORS["accent"])

    # ── Feature Transformer ───────────────────────────────────────
    _box(2.8, 0.8, 3.0, 4.4, "#1e2a45",
         f"Feature Transformer", f"Linear(768 → {H})\nClippedReLU ∈ [0, 1]")

    # ── Accumulateur (résultat FT) ────────────────────────────────
    acc_color = COLORS["accent"]
    _box(6.2, 1.8, 1.6, 2.4, acc_color,
         f"[{H}]", "Accumulateur", alpha=0.75)
    _arrow(5.8, 3.0, 6.2, 3.0, COLORS["accent"])

    # ── Move encoding (injection latérale) ───────────────────────
    _box(5.5, 0.1, 2.4, 0.9, "#321e45",
         "[4]", "Move encoding (UCI)", alpha=0.75)
    _arrow(6.5, 1.0, 7.0, 1.8, "#9b59b6")

    # ── Concaténation ─────────────────────────────────────────────
    _box(8.1, 1.5, 2.0, 3.0, "#1a2718",
         f"[{H + 4}]", "Concat(acc, move)", alpha=0.7)
    _arrow(7.8, 3.0, 8.1, 3.0, COLORS["accent"])

    # ── Head L1 ──────────────────────────────────────────────────
    _box(10.4, 1.8, 1.4, 2.4, COLORS["primary"],
         "[64]", "Linear + ReLU", alpha=0.8)
    _arrow(10.1, 3.0, 10.4, 3.0, COLORS["subtext"])

    # ── Head L2 / Output ─────────────────────────────────────────
    _box(12.1, 2.3, 1.5, 1.4, COLORS["warning"],
         "[1]", "Score\n(logit)", alpha=0.85)
    _arrow(11.8, 3.0, 12.1, 3.0, COLORS["warning"])

    # ── Paramètres ────────────────────────────────────────────────
    n_ft = 768 * H + H
    n_head = (H + 4) * 64 + 64 + 64 * 1 + 1
    n_total = n_ft + n_head

    ax.text(7.0, 5.5,
            f"Paramètres totaux : {n_total:,}  "
            f"(FT : {n_ft:,}  |  Head : {n_head:,})",
            ha="center", va="center", color=COLORS["text"], fontsize=10)

    ax.set_title(f"Architecture NNUE  (H = {H})",
                 color=COLORS["text"], fontsize=13, pad=12)

    legend_items = [
        mpatches.Patch(color=COLORS["accent"], label="Feature Transformer (FT)"),
        mpatches.Patch(color=COLORS["primary"], label="Head (hidden)"),
        mpatches.Patch(color=COLORS["warning"], label="Sortie (score)"),
        mpatches.Patch(color="#9b59b6", label="Move encoding (injecté)"),
    ]
    ax.legend(handles=legend_items, loc="lower right",
              facecolor=COLORS["surface"], labelcolor=COLORS["text"],
              framealpha=0.9, fontsize=8)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"[ok] Architecture NNUE → {output_path}")


def plot_ft_heatmap(model: nn.Module, output_path: str = "models/nnue_ft_heatmap.png") -> None:
    """
    Visualise les poids du FeatureTransformer (FT) par type de pièce.

    Deux sous-graphiques :
      • Heatmap 12×H : poids moyens de chaque type de pièce dans l'accumulateur
      • Bar chart 12 : norme L2 des poids par type de pièce (importance)
    """
    H = model.H
    W = model.ft.weight.detach().cpu().numpy()  # [H, 768]
    W_T = W.T  # [768, H]

    # Réorganiser par pièce : [12, 64, H]
    W_pieces = W_T.reshape(12, 64, H)

    # Moyenne sur les 64 cases → [12, H]
    W_mean = W_pieces.mean(axis=1)
    # Norme L2 par type de pièce → [12]
    W_norm = np.linalg.norm(W_pieces.reshape(12, -1), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _apply_theme(fig, axes)

    # ── Heatmap 12 × H ───────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(W_mean, aspect="auto", cmap="RdYlGn",
                   vmin=-abs(W_mean).max(), vmax=abs(W_mean).max())
    ax.set_yticks(range(12))
    ax.set_yticklabels(PIECE_LABELS, color=COLORS["text"], fontsize=9)
    ax.set_xlabel(f"Dimension accumulateur (0 → {H - 1})", color=COLORS["text"])
    ax.set_title(f"Poids FT par pièce — moyenne sur 64 cases\n(vert = positif, rouge = négatif)",
                 color=COLORS["text"], fontsize=10)
    # Séparateur blanc ↔ noir
    ax.axhline(5.5, color=COLORS["grid"], linewidth=2, linestyle="--", alpha=0.6)
    ax.text(-1.5, 2.5, "Blancs", color=COLORS["text"], fontsize=8,
            va="center", ha="right", rotation=90)
    ax.text(-1.5, 8.5, "Noirs", color=COLORS["text"], fontsize=8,
            va="center", ha="right", rotation=90)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.yaxis.set_tick_params(color=COLORS["subtext"])

    # ── Importance (norme L2) ─────────────────────────────────────
    ax = axes[1]
    bar_colors = [COLORS["primary"]] * 6 + [COLORS["danger"]] * 6
    bars = ax.barh(range(12), W_norm, color=bar_colors, edgecolor="none")
    ax.set_yticks(range(12))
    ax.set_yticklabels(PIECE_LABELS, color=COLORS["text"], fontsize=9)
    ax.set_xlabel("Norme L2 des poids FT (importance)", color=COLORS["text"])
    ax.set_title("Importance par type de pièce\n(plus haut = plus influent)",
                 color=COLORS["text"], fontsize=10)
    # Valeur sur chaque barre
    for bar, val in zip(bars, W_norm):
        ax.text(val + W_norm.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", color=COLORS["text"], fontsize=8)
    legend_items = [
        mpatches.Patch(color=COLORS["primary"], label="Pièces blanches"),
        mpatches.Patch(color=COLORS["danger"], label="Pièces noires"),
    ]
    ax.legend(handles=legend_items, facecolor=COLORS["surface"],
              labelcolor=COLORS["text"], fontsize=8)

    fig.suptitle("Feature Transformer — Analyse des poids", color=COLORS["text"], fontsize=12)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"[ok] FT heatmap → {output_path}")


def plot_accumulator_analysis(
        model: nn.Module,
        board: list | None = None,
        legal_moves: list | None = None,
        output_path: str = "models/nnue_accumulator.png",
) -> None:
    """
    Visualise l'accumulateur et les scores de coups pour une position donnée.

    Si `board` est None, utilise la position initiale des Blancs.
    Si `legal_moves` est None, utilise quelques coups d'ouverture typiques.
    """
    # Position par défaut : position initiale
    if board is None:
        board = [
            ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"],  # rang 1
            ["wP", "wP", "wP", "wP", "wP", "wP", "wP", "wP"],  # rang 2
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["bP", "bP", "bP", "bP", "bP", "bP", "bP", "bP"],  # rang 7
            ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],  # rang 8
        ]
    if legal_moves is None:
        legal_moves = ["e2e4", "d2d4", "c2c4", "g1f3", "b1c3",
                       "e2e3", "d2d3", "g2g3", "f2f4", "h2h4"]

    model.eval()
    with torch.no_grad():
        board_feat = torch.from_numpy(board_to_features(board)).unsqueeze(0)

        # Accumulateur (H floats après ClippedReLU)
        acc = torch.clamp(model.ft(board_feat), 0.0, 1.0)[0].cpu().numpy()

        # Score de chaque coup légal
        scores = {}
        for mv in legal_moves:
            mv_enc = torch.from_numpy(uci_to_enc(mv)).unsqueeze(0)
            inp = torch.cat([torch.from_numpy(acc).unsqueeze(0), mv_enc], dim=-1)
            scores[mv] = model.head(inp).item()

    H = len(acc)
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(COLORS["bg"])
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Accumulateur (bar chart des H features) ────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(COLORS["surface"])
    _apply_theme(fig, [ax1])
    bar_colors = [COLORS["accent"] if v > 0.5 else COLORS["primary"] for v in acc]
    ax1.bar(range(H), acc, color=bar_colors, edgecolor="none", width=1.0)
    ax1.axhline(0.5, color=COLORS["warning"], linewidth=1, linestyle="--", alpha=0.7, label="Seuil 0.5")
    ax1.set_xlim(-1, H)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel(f"Dimension de l'accumulateur (0 → {H - 1})", color=COLORS["text"])
    ax1.set_ylabel("Valeur (ClippedReLU ∈ [0, 1])", color=COLORS["text"])
    ax1.set_title(f"Accumulateur NNUE — Position initiale  ({int(acc.sum())} features actives / {H})",
                  color=COLORS["text"], fontsize=11)
    legend_items = [
        mpatches.Patch(color=COLORS["accent"], label="Haute activation (> 0.5)"),
        mpatches.Patch(color=COLORS["primary"], label="Basse activation"),
    ]
    ax1.legend(handles=legend_items, facecolor=COLORS["surface"],
               labelcolor=COLORS["text"], fontsize=8)
    n_high = int((acc > 0.5).sum())
    ax1.text(H * 0.98, 0.97,
             f"Max: {acc.max():.3f}  Min: {acc.min():.3f}\n"
             f"Moy: {acc.mean():.3f}  Haute: {n_high}/{H}",
             ha="right", va="top", transform=ax1.transAxes,
             color=COLORS["subtext"], fontsize=8,
             bbox=dict(facecolor=COLORS["bg"], alpha=0.7, edgecolor="none"))

    # ── 2. Scores des coups (bar chart) ──────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    _apply_theme(fig, [ax2])
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    moves = [m for m, _ in sorted_scores]
    values = [v for _, v in sorted_scores]
    bar_c = [COLORS["accent"] if i == 0 else COLORS["primary"] if v > 0 else COLORS["danger"]
             for i, (_, v) in enumerate(sorted_scores)]
    bars = ax2.bar(moves, values, color=bar_c, edgecolor="none", width=0.6)
    ax2.axhline(0, color=COLORS["grid"], linewidth=1)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + abs(max(values) - min(values)) * 0.02,
                 f"{val:.3f}", ha="center", va="bottom",
                 color=COLORS["text"], fontsize=8)
    ax2.set_xlabel("Coup UCI", color=COLORS["text"])
    ax2.set_ylabel("Score (logit)", color=COLORS["text"])
    ax2.set_title("Scores NNUE — Position initiale (coups d'ouverture)",
                  color=COLORS["text"], fontsize=11)

    fig.suptitle("Analyse de l'accumulateur et des scores NNUE",
                 color=COLORS["text"], fontsize=13, y=1.01)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"[ok] Accumulateur → {output_path}")


def generate_nnue_visuals(
        model: nn.Module,
        history: dict,
        save_dir: str,
        epoch: int | None = None,
) -> None:
    """
    Génère tous les graphiques de visualisation NNUE.

    Appelé à la fin de chaque epoch (curves uniquement) et à la fin du
    training complet (tous les graphiques).

    Args:
        model    : NnueModel (sur CPU)
        history  : dict {"batch_losses", "epoch_train", "epoch_val", "lr_history"}
        save_dir : répertoire de sortie
        epoch    : si None → génère tous les graphiques (fin de training)
                   si int  → seulement les courbes (fin d'epoch)
    """
    d = Path(save_dir)
    model.eval()
    model_cpu = model.cpu()

    # Courbes d'entraînement — générées à chaque epoch
    suffix = f"_ep{epoch}" if epoch is not None else ""
    plot_training_history(
        history,
        output_path=str(d / f"nnue_training_curves{suffix}.png"),
    )

    # Graphiques complets uniquement en fin de training
    if epoch is None:
        plot_nnue_architecture(
            H=model_cpu.H,
            output_path=str(d / "nnue_architecture.png"),
        )
        plot_ft_heatmap(
            model_cpu,
            output_path=str(d / "nnue_ft_heatmap.png"),
        )
        plot_weight_distributions(
            model_cpu,
            output_path=str(d / "nnue_weight_distributions.png"),
        )
        plot_accumulator_analysis(
            model_cpu,
            output_path=str(d / "nnue_accumulator.png"),
        )
        print(f"\n[viz] Tous les graphiques NNUE générés dans {save_dir}/")


# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULTS = {
    "H": 64,
    "min_elo": 1800,
    "neg_per_pos": 3,
    "max_games": None,
    "skip_opening": 10,
    "max_ply": 120,
    "batch_size": 512,
    "lr": 1e-3,
    "epochs": 10,
    "val_split": 0.05,
    "save_dir": "../models",
    "log_every": 200,
    "viz_every": 1,  # générer les courbes toutes les N epochs
}


# ─── Dataset loading ─────────────────────────────────────────────────────────

def load_nnue_dataset(args):
    print(f"[dataset] Chargement de {args.pgn} ...")
    ds = LichessDatasetInMemory(
        args.pgn,
        min_elo=args.min_elo,
        neg_per_pos=args.neg_per_pos,
        max_games=args.max_games,
        skip_opening=args.skip_opening,
        max_ply=args.max_ply,
    )
    if not hasattr(ds, '_raw_samples'):
        raise RuntimeError(
            "LichessDatasetInMemory doit exposer '_raw_samples' "
            "(list of (board, move, label)). Voir dataset.py."
        )
    nnue_ds = NnueDataset(ds._raw_samples)
    print(f"[dataset] {len(nnue_ds):,} échantillons NNUE chargés.")
    return nnue_ds


# ─── Training ────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Dataset
    dataset = load_nnue_dataset(args)
    val_size = max(1, int(len(dataset) * args.val_split))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Model
    model = NnueModel(H=args.H).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    ft_params = sum(p.numel() for p in model.ft.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    print(f"[model] NnueModel H={args.H} — {n_params:,} params "
          f"(FT: {ft_params:,}  Head: {head_params:,})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5, min_lr=1e-5
    )
    criterion = nn.BCEWithLogitsLoss()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    history = {
        "batch_losses": [],
        "epoch_train": [],
        "epoch_val": [],
        "lr_history": [],
    }

    # Génère l'architecture avant même d'entraîner
    print("[viz] Génération du diagramme d'architecture …")
    plot_nnue_architecture(args.H, str(save_dir / "nnue_architecture.png"))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        batch_log = []
        t0 = time.time()

        for batch_idx, (board_feat, move_enc, y) in enumerate(train_loader):
            board_feat = board_feat.to(device)
            move_enc = move_enc.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(board_feat, move_enc)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_log.append(loss.item())
            n_batches += 1

            if batch_idx % args.log_every == 0 and batch_idx > 0:
                avg = total_loss / n_batches
                elapsed = time.time() - t0
                print(f"  Epoch {epoch}/{args.epochs}  batch {batch_idx}  "
                      f"loss={avg:.4f}  ({elapsed:.0f}s)")

        avg_train = total_loss / max(n_batches, 1)
        history["batch_losses"].append(batch_log)
        history["epoch_train"].append(avg_train)
        history["lr_history"].append(optimizer.param_groups[0]["lr"])

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for board_feat, move_enc, y in val_loader:
                board_feat = board_feat.to(device)
                move_enc = move_enc.to(device)
                y = y.to(device)
                val_loss += criterion(model(board_feat, move_enc), y).item()
                n_val += 1
        avg_val = val_loss / max(n_val, 1)
        history["epoch_val"].append(avg_val)
        scheduler.step(avg_val)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs}  train={avg_train:.4f}  "
              f"val={avg_val:.4f}  ({elapsed:.1f}s)")

        # Checkpoint
        torch.save(model.state_dict(),
                   save_dir / f"chess_nnue_epoch{epoch}.pt")
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), save_dir / "chess_nnue_best.pt")
            print(f"  ★ Meilleur modèle → chess_nnue_best.pt")

        # Sauvegarde historique JSON
        with open(save_dir / "nnue_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f)

        # ── Visualisation de fin d'epoch ──────────────────────────
        if epoch % args.viz_every == 0:
            print(f"  [viz] Courbes epoch {epoch} …")
            try:
                generate_nnue_visuals(
                    model.cpu(), history,
                    save_dir=str(save_dir),
                    epoch=epoch,
                )
                model.to(device)  # remettre sur GPU si besoin
            except Exception as e:
                print(f"  [viz] Avertissement : {e}")

    # ── Visualisation finale (tous les graphiques) ────────────────
    print("\n[viz] Génération des visualisations finales …")
    model.load_state_dict(
        torch.load(save_dir / "chess_nnue_best.pt", map_location="cpu")
    )
    try:
        generate_nnue_visuals(model.cpu(), history, save_dir=str(save_dir), epoch=None)
    except Exception as e:
        print(f"[viz] Avertissement : {e}")

    # ── Export .ludus-nnue ────────────────────────────────────────
    print("\n[export] Export .ludus-nnue …")
    nnue_path = save_dir / "chess_eval.ludus-nnue"
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from export_nnue import export_nnue
        export_nnue(model.cpu(), str(nnue_path))
    except Exception as e:
        print(f"[export] Erreur: {e}")

    print(f"\n[done] Entraînement NNUE terminé !"
          f"\n  Modèle  : {save_dir}/chess_nnue_best.pt"
          f"\n  Weights : {nnue_path}"
          f"\n  Graphs  : {save_dir}/nnue_*.png")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entraîne un modèle NNUE pour ChessEval")
    p.add_argument("--pgn", required=True, help="Fichier PGN (.pgn, .bz2, .zst)")
    p.add_argument("--H", type=int,
                   default=DEFAULTS["H"],
                   help=f"Taille accumulateur FT (défaut: {DEFAULTS['H']})")
    p.add_argument("--min-elo", type=int, default=DEFAULTS["min_elo"])
    p.add_argument("--neg-per-pos", type=int, default=DEFAULTS["neg_per_pos"])
    p.add_argument("--max-games", type=int, default=DEFAULTS["max_games"])
    p.add_argument("--skip-opening", type=int, default=DEFAULTS["skip_opening"])
    p.add_argument("--max-ply", type=int, default=DEFAULTS["max_ply"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--val-split", type=float, default=DEFAULTS["val_split"])
    p.add_argument("--save-dir", default=DEFAULTS["save_dir"])
    p.add_argument("--log-every", type=int, default=DEFAULTS["log_every"])
    p.add_argument("--viz-every", type=int, default=DEFAULTS["viz_every"],
                   help="Générer les courbes d'entraînement toutes les N epochs")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
