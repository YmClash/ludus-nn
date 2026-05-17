# ludus-nn — Chess Neural Network

Pipeline d'entraînement d'un évaluateur de coups d'échecs pour le bot Ludus.

```
Lichess PGN  →  encode.py  →  dataset.py  →  train.py  →  chess_eval.onnx
                                                               ↓
                                              ludus-sdk/models/chess_eval.onnx
                                                               ↓
                                                    NeuralBot WASM
```

## Structure

```
ludus-nn/
├── src/
│   ├── encode.py    # Encodage board/move → float (miroir de neural.rs)
│   ├── dataset.py   # Pipeline Lichess PGN → PyTorch Dataset
│   ├── model.py     # Architectures ChessEval / ChessEvalBN
│   ├── train.py     # Boucle d'entraînement + CLI
│   └── export.py    # Export ONNX + validation onnxruntime
├── data/
│   └── README.md    # Comment télécharger les données Lichess
├── models/          # Checkpoints .pt + chess_eval.onnx (généré)
└── requirements.txt
```

## Installation

```bash
cd ludus-nn
pip install -r requirements.txt
```

## Workflow complet

### 1. Télécharger les données
```bash
# Petit fichier pour tester (Lichess 2013-01, ~100MB)
curl -O https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst
mv lichess_db_standard_rated_2013-01.pgn.zst data/
```

### 2. Entraîner

```bash
# Entraînement rapide (5k parties, 3 epochs)
python src/train.py \
    --pgn data/lichess_db_standard_rated_2013-01.pgn.zst \
    --max-games 5000 \
    --min-elo 1800 \
    --epochs 3

# Produit : models/chess_eval_best.pt + models/chess_eval.onnx
```

### 3. Valider le modèle ONNX

```bash
python src/export.py \
    --checkpoint models/chess_eval_best.pt \
    --output models/chess_eval.onnx
```

### 4. Déployer dans le bot Rust

```bash
cp models/chess_eval.onnx ../ludus-sdk/models/chess_eval.onnx

cd ../ludus-sdk
cargo build --release --target wasm32-unknown-unknown --features neural
# → target/wasm32-unknown-unknown/release/ludus_sdk.wasm

#exporter les weights en binaire pour micro_neural
cd ludus-nn
python src/export_weights.py \
    --checkpoint models/chess_eval_best.pt \
    --output ../ludus-sdk/models/chess_eval.bin \
    --validate


# Depuis un checkpoint existant (pas besoin de réentraîner)
python src/visualize.py `
    --history ..\models\training_history.json `
    --checkpoint ..\models\chess_eval_best.pt

# Architecture seulement :
python src/visualize.py --arch --checkpoint ..\models\chess_eval_best.pt

# Distribution des poids seulement :
python src/visualize.py --weights --checkpoint ..\models\chess_eval_best.pt

```


---

## Modèle

- **Input**  : `[1, 68]` float32 = board (64) + move (4)
- **Output** : `[1, 1]`  float32 = quality score
- **Format** : ONNX opset 13 (compatible `tract-onnx`)

## Stratégie d'apprentissage

**Imitation Learning pondérée par l'outcome** :
- Coup joué par un joueur ≥ 1800 ELO → label `1.0` (victoire) / `0.5` (nul) / `0.0` (défaite)
- Coups aléatoires parmi les légaux → label `0.0`
- Loss : `BCEWithLogitsLoss`
