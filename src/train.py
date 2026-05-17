# -*- coding: utf-8 -*-
"""
train.py — Boucle d'entraînement du modèle ChessEval.

Usage :
    python src/train.py --pgn data/lichess_db_2024-01.pgn.zst

Options complètes :
    python src/train.py --help
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent))

from dataset import LichessDataset, LichessDatasetInMemory
from model import build_model
from export import export_onnx
from visualize import generate_all


# ─── Configuration par défaut ────────────────────────────────────────────────

DEFAULTS = {
    "model_variant": "simple",     # "simple" | "bn"
    "min_elo":       1800,         # ELO minimum pour filtrer les parties
    "neg_per_pos":   3,            # Coups négatifs par position
    "max_games":     None,         # None = toutes les parties
    "skip_opening":  10,           # Ignorer les N premiers demi-coups
    "max_ply":       120,          # Ignorer après N demi-coups
    "batch_size":    512,
    "lr":            1e-3,
    "epochs":        10,
    "val_split":     0.05,         # 5% validation
    "in_memory":     False,        # Charger tout en RAM (petit dataset)
    "save_dir":      "../models",
    "log_every":     200,          # log tous les N batches
}

