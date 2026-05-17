# -*- coding: utf-8 -*-
"""
model.py — Architecture du réseau de neurones.

Input  : [batch, 68]  float32  (board_flat + move)
Output : [batch, 1]   float32  (quality score)

Deux variantes :
  - ChessEval   : simple (3 couches) — rapide, ~50KB ONNX ← recommandé pour débuter
  - ChessEvalBN : avec BatchNorm + Dropout — meilleure généralisation, plus lourd

une Troisième variante  est l'intruduction de reseau de neurone NNUE(Efficiently Updatable Neural Network)
qui est une architecture optimisée pour les moteurs d'échecs, offrant une évaluation rapide et précise des positions.
et surtout produit une fiche wasm tres legere:
Architecture :
    Input : 768 features binaires (12 pièces × 64 cases)
    FT    : Linear(768, H)  + ClippedReLU → accumulateur [H]
    Head  : Linear(H+4, 64) + ReLU → Linear(64, 1)

l'implementation est deja implememente dans train_nnue.py et export_nnue.py, mais n'est pas encore integré dans le pipeline de training et export classique.

by @YmC
"""

import torch
import torch.nn as nn
from typing import Literal

class ChessEval(nn.Module):
    """
        Réseau simple 3 couches.

        Input  [68] → Linear(128) → ReLU → Linear(64) → ReLU → Linear(1)

        ~50KB en ONNX. Recommandé pour le premier entraînement.
        """

    def __init__(self,hidden1:int=128,hidden2:int =64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(68, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChessEvalBN(nn.Module):
    """
    Réseau avec BatchNorm et Dropout pour meilleure généralisation.

    Input  [68] → BN → Linear(256) → BN → ReLU → Dropout
                      → Linear(128) → BN → ReLU → Dropout
                      → Linear(64)  → ReLU
                      → Linear(1)

    ~200KB en ONNX. Meilleur pour de grands datasets (> 1M positions).
    """

    def __init__(self,hidden: tuple[int, ...] = (256, 128, 64),dropout: float = 0.15,):
        super().__init__()
        layers: list[nn.Module] = [nn.BatchNorm1d(68)]
        in_size = 68
        for h in hidden:
            layers += [
                nn.Linear(in_size, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Factory ─────────────────────────────────────────────────────────────────

ModelVariant = Literal["simple", "bn"]

def build_model(variant: ModelVariant = "simple", **kwargs) -> nn.Module:
    """Construit le modèle selon la variante choisie dans la config."""
    if variant == "simple":
        return ChessEval(**kwargs)
    elif variant == "bn":
        return ChessEvalBN(**kwargs)
    raise ValueError(f"Variante inconnue: {variant!r}. Choisir 'simple' ou 'bn'.")

# ─── Test rapide ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for name,model in [("ChessEval", ChessEval()), ("ChessEvalBN", ChessEvalBN())]:
        x = torch.randn(4, 68)
        y = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}: input={tuple(x.shape)} output={tuple(y.shape)} params={params:,}")





