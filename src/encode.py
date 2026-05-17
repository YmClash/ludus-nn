import chess
import numpy as np
from typing import Optional


# ─── Valeurs des pièces (range [-1.0, 1.0]) ─────────────────────────────────
# Blancs positifs, Noirs négatifs, vide = 0.0
# Doit correspondre à piece_to_value() dans neural.rs

PIECE_VALUES: dict[Optional[chess.PieceType], float] = {
    chess.KING:   1.000,
    chess.QUEEN:  0.900,
    chess.ROOK:   0.500,
    chess.BISHOP: 0.333,
    chess.KNIGHT: 0.320,
    chess.PAWN:   0.100,
    None: 0.0,
}


def piece_to_value(piece: Optional[chess.Piece]) -> float:
    """Encode une pièce en float normalisé. Blanc = positif, Noir = négatif."""
    if piece is None:
        return 0.0
    val = PIECE_VALUES.get(piece.piece_type, 0.0)
    return val if piece.color == chess.WHITE else -val


def board_to_tensor_flat(board: chess.Board) -> np.ndarray:
    """
    Encode l'échiquier en [64] float32, row-major.

    Row 0 = rank 1 (rangée blanche), col 0 = file a.
    Correspond à board_to_tensor_flat() dans neural.rs.
    """
    out = np.zeros(64, dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        rank = chess.square_rank(sq)   # 0 = rank 1
        file = chess.square_file(sq)   # 0 = file a
        out[rank * 8 + file] = piece_to_value(piece)
    return out


def board_to_tensor_planes(board: chess.Board) -> np.ndarray:
    """
    Encodage AlphaZero : 12 plans binaires [768] float32.

    Ordre : wK wQ wR wB wN wP bK bQ bR bB bN bP
    Correspond à board_to_tensor_planes() dans neural.rs.
    """
    piece_order = [
        (chess.KING,   chess.WHITE), (chess.QUEEN,  chess.WHITE),
        (chess.ROOK,   chess.WHITE), (chess.BISHOP, chess.WHITE),
        (chess.KNIGHT, chess.WHITE), (chess.PAWN,   chess.WHITE),
        (chess.KING,   chess.BLACK), (chess.QUEEN,  chess.BLACK),
        (chess.ROOK,   chess.BLACK), (chess.BISHOP, chess.BLACK),
        (chess.KNIGHT, chess.BLACK), (chess.PAWN,   chess.BLACK),
    ]
    out = np.zeros(768, dtype=np.float32)
    for plane, (pt, color) in enumerate(piece_order):
        for sq in board.pieces(pt, color):
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            out[plane * 64 + rank * 8 + file] = 1.0
    return out


def move_to_tensor(move: chess.Move) -> np.ndarray:
    """
    Encode un coup en [4] float32 : [from_row, from_col, to_row, to_col].
    Valeurs dans [0.0, 1.0] (divisé par 7).

    Correspond à uci_to_tensor() dans neural.rs.
    """
    return np.array([
        chess.square_rank(move.from_square) / 7.0,
        chess.square_file(move.from_square) / 7.0,
        chess.square_rank(move.to_square)   / 7.0,
        chess.square_file(move.to_square)   / 7.0,
    ], dtype=np.float32)


def encode_sample(board: chess.Board, move: chess.Move) -> np.ndarray:
    """
    Encode (board, move) → [68] float32.

    C'est le vecteur d'entrée du modèle :
      [0:64]  = board_to_tensor_flat
      [64:68] = move_to_tensor
    """
    return np.concatenate([board_to_tensor_flat(board), move_to_tensor(move)])


board_tensor = board_to_tensor_planes(chess.Board())
print("Tensor de l'échiquier (64 éléments):{}",format(board_tensor))