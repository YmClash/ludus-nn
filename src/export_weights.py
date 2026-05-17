# -*- coding: utf-8 -*-
"""
export_weights.py -- Exporte les poids PyTorch vers le format binaire Ludus (.bin).

Le format .bin est beaucoup plus leger que l'ONNX complet car il n'inclut pas
le runtime d'inference (tract-onnx = ~15MB; ce format = quelques KB de poids).

Usage :
    python src/export_weights.py --checkpoint models/chess_eval_best.pt
    # => models/chess_eval.bin (~50-200 KB selon le modele)

Puis copier vers le bot :
    cp models/chess_eval.bin ../path/to/chessbot/models/chess_eval.bin

Dans le bot Rust :
    const WEIGHTS: &[u8] = include_bytes!("../models/chess_eval.bin");

    fn next_move(state: &GameState) -> String {
        micro_neural::best_move(WEIGHTS, state)
            .unwrap_or_else(|| ludus_sdk::random_move(state))
    }

Format binaire :
    [n_layers: u32 LE]
    for each layer:
        [in_size: u32 LE]
        [out_size: u32 LE]
        [weights: in_size * out_size f32 LE, row-major (W[i,j] = weights[i*out+j])]
        [biases:  out_size f32 LE]
"""
import argparse
import struct
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from model import build_model


def export_bin(model: nn.Module, output_path: str) -> int:
    """
    Exporte les couches lineaires du modele au format binaire Ludus.
    Retourne la taille en octets.
    """
    # Extraire toutes les couches Linear dans l'ordre
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]

    if not linear_layers:
        raise ValueError("Aucune couche nn.Linear trouvee dans le modele.")

    buf = bytearray()

    # Nombre de couches
    buf += struct.pack("<I", len(linear_layers))

    for layer in linear_layers:
        W = layer.weight.detach().cpu().float()   # [out, in]
        b = layer.bias.detach().cpu().float()     # [out]

        in_size  = W.shape[1]
        out_size = W.shape[0]

        buf += struct.pack("<I", in_size)
        buf += struct.pack("<I", out_size)

        # Poids en row-major [in x out] : W[k, j] = layer.weight.T[k, j]
        # torch W est [out, in], on transpose pour [in, out]
        W_transposed = W.T.contiguous()  # [in, out]
        buf += W_transposed.numpy().tobytes()

        # Biais
        buf += b.numpy().tobytes()

    Path(output_path).write_bytes(buf)
    return len(buf)


def validate_bin(weights_path: str) -> None:
    """Verifie que le fichier .bin peut etre parse correctement."""
    data = Path(weights_path).read_bytes()
    off = 0

    def read_u32():
        nonlocal off
        v = struct.unpack_from("<I", data, off)[0]
        off += 4
        return v

    def read_f32(n):
        nonlocal off
        v = struct.unpack_from(f"<{n}f", data, off)
        off += 4 * n
        return v

    n_layers = read_u32()
    print(f"[ok] {n_layers} couche(s) trouvee(s)")
    for i in range(n_layers):
        in_s  = read_u32()
        out_s = read_u32()
        _weights = read_f32(in_s * out_s)
        _biases  = read_f32(out_s)
        total = in_s * out_s + out_s
        print(f"  Layer {i+1}: [{in_s} -> {out_s}]  {total} parametres")

    print(f"[ok] Parse complet ({len(data)} octets, {len(data)/1024:.1f} KB)")


def test_forward(model: nn.Module, weights_path: str) -> None:
    """Verifie coherence entre inference PyTorch et parsing du .bin."""
    import numpy as np

    data = Path(weights_path).read_bytes()
    off = 0

    def ru32():
        nonlocal off
        v = struct.unpack_from("<I", data, off)[0]; off += 4; return v
    def rf32s(n):
        nonlocal off
        v = struct.unpack_from(f"<{n}f", data, off); off += 4*n; return list(v)

    n_layers = ru32()
    layers = []
    for _ in range(n_layers):
        in_s = ru32(); out_s = ru32()
        W = np.array(rf32s(in_s * out_s), dtype=np.float32).reshape(in_s, out_s)
        b = np.array(rf32s(out_s), dtype=np.float32)
        layers.append((W, b))

    # Forward pass manuel
    x_np = np.random.randn(68).astype(np.float32)
    h = x_np.copy()
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)   # ReLU

    # Forward PyTorch
    model.eval()
    with torch.no_grad():
        x_th = torch.from_numpy(x_np).unsqueeze(0)
        y_th = model(x_th).item()

    diff = abs(h[0] - y_th)
    ok = diff < 1e-4
    print(f"[{'ok' if ok else 'ERREUR'}] Forward pass: numpy={h[0]:.6f}  pytorch={y_th:.6f}  diff={diff:.2e}")
    if not ok:
        print("  ATTENTION: incoherence detectee -- verifier l'architecture du modele")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Exporte les poids PyTorch vers le format .bin Ludus")
    p.add_argument("--checkpoint", required=True,   help="Chemin vers le .pt")
    p.add_argument("--output",     default="../models/chess_eval.bin")
    p.add_argument("--model",      dest="model_variant", default="simple", choices=["simple","bn"])
    p.add_argument("--validate",   action="store_true", help="Valider + tester le .bin apres export")
    args = p.parse_args()

    model = build_model(args.model_variant)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    print(f"[ok] Checkpoint charge : {args.checkpoint}")

    n_bytes = export_bin(model, args.output)
    print(f"[ok] Poids exportes -> {args.output}  ({n_bytes/1024:.1f} KB)")

    if args.validate:
        print("\nValidation du .bin :")
        validate_bin(args.output)
        print("\nTest forward pass :")
        test_forward(model, args.output)

    print(f"\n[next] Copier vers votre bot :")
    print(f"   cp {args.output} ../path/to/chessbot/models/chess_eval.bin")
    print(f"\n[next] Dans lib.rs de votre bot :")
    print(f"   const WEIGHTS: &[u8] = include_bytes!(\"../models/chess_eval.bin\");")
    print(f"   // Puis: micro_neural::best_move(WEIGHTS, state)")
