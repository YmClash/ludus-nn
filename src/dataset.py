# -*- coding: utf-8 -*-
"""
dataset.py — Pipeline de données Lichess PGN → PyTorch Dataset

Stratégie d'apprentissage (Imitation Learning pondérée par outcome) :
  - Pour chaque position dans une partie de joueur fort (ELO > min_elo) :
    - Le coup joué = label +1.0 si le joueur a gagné, +0.5 si draw, 0.0 si perdu
    - N coups aléatoires parmi les légaux = label 0.0 (ou -1.0 selon le mode)
  - Résultat : le modèle apprend à imiter les bons joueurs pondérés par le résultat

Formats supportés : .pgn, .pgn.bz2, .pgn.zst (téléchargements Lichess)
"""


import bz2
import io
import random
import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset,IterableDataset
from pathlib import Path
from typing import Iterator,Optional
from tqdm import tqdm

from encode import encode_sample

# ─── Helpers ─────────────────────────────────────────────────────────────────

def open_pgn(path: str|Path):
    """Ouvre un fichier PGN (plain, .bz2, ou .zst)."""
    path = Path(path)
    suffix = "".join(path.suffixes).lower()

    if ".zst" in suffix:
        try:
            import zstandard as zstd
        except ImportError:
            raise ImportError("pip install zstandard pour lire les .pgn.zst de Lichess")
        ctx = zstd.ZstdDecompressor()
        raw = open(path, "rb")
        return io.TextIOWrapper(ctx.stream_reader(raw),encoding="utf-8",errors="replace")
    elif ".bz2" in suffix:
        return open(path,encoding="utf-8",errors="replace")


def game_outcome_for_color(game: chess.pgn.Game,color:chess.Color) -> float:
    """Retourne le score du joueur de couleur `color` : 1.0 / 0.5 / 0.0."""
    result = game.headers.get("Result","*")
    if result == "1-0":
        return 1.0 if color == chess.WHITE else 0.0
    elif result == "0-1":
        return 0.0 if color == chess.WHITE else 1.0
    elif result == "1/2-1/2":
        return 0.5
    return -1.0  # Partie inachève -> exclure


def get_elo(game: chess.pgn.Game, color: chess.Color) -> Optional[int]:
    """Retourne l'ELO du joueur ou None si absent."""
    key = "WhiteElo" if color == chess.WHITE else "BlackElo"
    try:
        return int(game.headers.get(key,"0") or "0")
    except ValueError:
        return None


def _chess_board_to_list(board:chess.Board) -> list:
    """
        Convertit un chess.Board en liste Ludus : list[8][8] de 'wP', 'bK', '' etc.

        Format identique a celui envoye par le serveur dans GameState.board :
          - board[0] = rang 1 (rangee des blancs)
          - board[7] = rang 8 (rangee des noirs)
          - board[row][col] = '' si vide, 'wP' 'bK' etc. sinon
        """
    result =[]
    for rank in range(8):
        row = []
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece is None:
                row.append('')
            else:
                color = "w" if piece.color == chess.WHITE else "b"
                row.append(f"{color}{piece.symbol().upper()}")
            result.append(row)
    return result

# ─── Iterable Dataset (streaming, pas de RAM excessive) ──────────────────────



class LichessDataset(IterableDataset):
    """
    Dataset en streaming depuis un fichier PGN Lichess.

    Génère des (X, y) tensors de forme ([68], [1]).

    X = board (64) + move (4)
    y = label in [0.0, 1.0]

    Args:
        pgn_path      : chemin vers le fichier PGN (plain, bz2, ou zst)
        min_elo       : ELO minimum du joueur qui joue le coup (défaut 1800)
        neg_per_pos   : coups négatifs par position (défaut 3)
        max_games     : nombre max de parties à lire (None = tout)
        max_ply       : ignorer les coups après N demi-coups (évite les fins techniques)
        skip_opening  : ignorer les N premiers demi-coups (ouvertures pas intéressantes)
    """

    def __init__(
        self,
        pgn_path: str | Path,
        min_elo: int = 1800,
        neg_per_pos: int = 3,
        max_games: Optional[int] = None,
        max_ply: int = 120,
        skip_opening: int = 10,
        _collect_raw: bool = False,   # si True, __iter__ yielde 5-tuples (x,y,board_list,uci,label)
    ):
        self.pgn_path     = Path(pgn_path)
        self.min_elo      = min_elo
        self.neg_per_pos  = neg_per_pos
        self.max_games    = max_games
        self.max_ply      = max_ply
        self.skip_opening = skip_opening
        self._collect_raw = _collect_raw

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        games_read = 0
        with open_pgn(self.pgn_path) as f:
            while True:
                if self.max_games and games_read >= self.max_games:
                    break
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                # Filtrer les parties incomplètes
                result_str = game.headers.get("Result", "*")
                if result_str == "*":
                    continue

                games_read += 1
                yield from self._extract_samples(game)

    def _extract_samples(self, game: chess.pgn.Game) -> Iterator:
        board = game.board()
        ply   = 0

        for node in game.mainline():
            move  = node.move
            color = board.turn

            if ply < self.skip_opening:
                board.push(move)
                ply += 1
                continue
            if ply > self.max_ply:
                break

            elo = get_elo(game, color)
            if elo is not None and elo < self.min_elo:
                board.push(move)
                ply += 1
                continue

            outcome = game_outcome_for_color(game, color)
            if outcome < 0:
                break

            legal = list(board.legal_moves)
            if not legal or move not in legal:
                board.push(move)
                ply += 1
                continue

            # Snapshot du plateau AVANT le coup (pour NNUE, si demande)
            board_list = _chess_board_to_list(board) if self._collect_raw else None

            # ── Echantillon positif ──────────────────────────────────────────
            x_pos = torch.from_numpy(encode_sample(board, move))
            y_pos = torch.tensor([outcome], dtype=torch.float32)
            if self._collect_raw:
                yield x_pos, y_pos, board_list, move.uci(), outcome
            else:
                yield x_pos, y_pos

            # ── Echantillons negatifs ────────────────────────────────────────
            others      = [m for m in legal if m != move]
            neg_samples = random.sample(others, min(self.neg_per_pos, len(others)))
            for neg_move in neg_samples:
                x_neg = torch.from_numpy(encode_sample(board, neg_move))
                y_neg = torch.tensor([0.0], dtype=torch.float32)
                if self._collect_raw:
                    yield x_neg, y_neg, board_list, neg_move.uci(), 0.0
                else:
                    yield x_neg, y_neg

            board.push(move)
            ply += 1



# ─── Version en mémoire (pour petits datasets / debug) ───────────────────────

class LichessDatasetInMemory(Dataset):
    """
    Charge toutes les parties en RAM.
    Utile pour tester ou pour de petits fichiers (~10k parties max).

    Attributs :
        samples       : list[(x_tensor, y_tensor)]          — pour train.py
        _raw_samples  : list[(board_list, uci_str, label)]  — pour train_nnue.py
    """

    def __init__(self, pgn_path: str | Path, **kwargs):
        super().__init__()
        print(f"Chargement du dataset depuis {pgn_path} …")
        self.samples:      list[tuple[torch.Tensor, torch.Tensor]] = []
        self._raw_samples: list[tuple[list, str, float]]            = []

        # Passe unique avec _collect_raw=True : on recupere les deux formats
        streaming = LichessDataset(pgn_path, _collect_raw=True, **kwargs)
        for item in tqdm(streaming, desc="Positions extraites"):
            x, y, board_list, uci_str, label = item
            self.samples.append((x, y))
            self._raw_samples.append((board_list, uci_str, float(label)))

        print(f"  → {len(self.samples)} echantillons charges")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


# ─── Utilitaire CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <fichier.pgn[.bz2|.zst]> [max_games]")
        sys.exit(1)

    path = sys.argv[1]
    max_g = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    ds = LichessDataset(path,max_games=max_g,min_elo=1800)
    count = 0
    for x,y in ds:
        count +=1
    print(f"✅ {count} échantillons extraits de {max_g} parties")








