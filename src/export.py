# -*- coding: utf-8 -*-
"""
export.py — Exporte le modèle PyTorch vers ONNX et vérifie l'inférence.

Usage direct :
    python src/export.py --checkpoint models/chess_eval_best.pt

Le fichier ONNX produit (chess_eval.onnx) peut être copié vers :
    ludus-sdk/models/chess_eval.onnx
puis inclus dans le bot via include_bytes!("../models/chess_eval.onnx").
"""


import sys
import argparse
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import build_model


def export_onnx(model: torch.nn.Module, output_path: str) -> None:
    """
    Exporte le modèle en ONNX.

    Format :
      Input  name="input"  shape=[1, 68]  dtype=float32
      Output name="output" shape=[1, 1]   dtype=float32

    Compatible avec tract-onnx (opset 13).
    """
    model.eval()
    dummy = torch.zeros(1, 68)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"[ok] ONNX exporte : {output_path}")


def validate_onnx(onnx_path: str) -> None:
    """
    Valide l'ONNX avec onnxruntime et vérifie la cohérence PyTorch / ONNX.
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("[warn] onnx/onnxruntime non installes -- validation ignoree")
        return

    # Verifie la structure du modele
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    print(f"[ok] Structure ONNX valide")

    # Test inference
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 68).astype(np.float32)
    result = sess.run(["output"], {"input": x})
    score = result[0][0, 0]
    print(f"[ok] Inference ONNX ok -- score test: {score:.4f}")

    # Vérification entrées / sorties
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(f"   Input  : name={inp.name!r} shape={inp.shape} dtype={inp.type}")
    print(f"   Output : name={out.name!r} shape={out.shape} dtype={out.type}")


def export_and_validate(
    checkpoint_path: str,
    output_path: str,
    model_variant: str = "simple",
) -> None:
    """Pipeline complet : charge le checkpoint → exporte ONNX → valide."""
    model = build_model(model_variant)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    print(f"✅ Checkpoint chargé : {checkpoint_path}")

    export_onnx(model, output_path)
    validate_onnx(output_path)

    onnx_size = Path(output_path).stat().st_size / 1024
    print(f"\n[size] Taille ONNX : {onnx_size:.1f} KB")
    print(f"\n[next] Copier vers le SDK :")
    print(f"   cp {output_path} ../ludus-sdk/models/chess_eval.onnx")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export PyTorch → ONNX")
    p.add_argument("--checkpoint", required=True,        help="Chemin vers le .pt")
    p.add_argument("--output",     default="models/chess_eval.onnx")
    p.add_argument("--model",      dest="model_variant", default="simple", choices=["simple","bn"])
    args = p.parse_args()
    export_and_validate(args.checkpoint, args.output, args.model_variant)
