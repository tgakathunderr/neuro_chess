"""Sensory chessboard representation and feature extraction for BIB-2."""

from typing import Dict, Optional, Tuple
import numpy as np
import chess


# Standard Piece Base Values
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 100.0,
}

# Simplified Centralization & Positional Bonuses (Normalized)
KNIGHT_PST = np.array([
    -0.5, -0.4, -0.3, -0.3, -0.3, -0.3, -0.4, -0.5,
    -0.4, -0.2,  0.0,  0.0,  0.0,  0.0, -0.2, -0.4,
    -0.3,  0.0,  0.2,  0.3,  0.3,  0.2,  0.0, -0.3,
    -0.3,  0.1,  0.3,  0.4,  0.4,  0.3,  0.1, -0.3,
    -0.3,  0.0,  0.3,  0.4,  0.4,  0.3,  0.0, -0.3,
    -0.3,  0.1,  0.2,  0.3,  0.3,  0.2,  0.1, -0.3,
    -0.4, -0.2,  0.0,  0.1,  0.1,  0.0, -0.2, -0.4,
    -0.5, -0.4, -0.3, -0.3, -0.3, -0.3, -0.4, -0.5,
], dtype=np.float32)

PAWN_PST = np.array([
     0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
     0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,
     0.1,  0.1,  0.2,  0.3,  0.3,  0.2,  0.1,  0.1,
     0.0,  0.0,  0.2,  0.4,  0.4,  0.2,  0.0,  0.0,
     0.0,  0.0,  0.1,  0.3,  0.3,  0.1,  0.0,  0.0,
     0.1, -0.1, -0.1,  0.0,  0.0, -0.1, -0.1,  0.1,
     0.1,  0.2,  0.2, -0.2, -0.2,  0.2,  0.2,  0.1,
     0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
], dtype=np.float32)


class ChessBoardSensory:
    """
    Translates an 8x8 chessboard into biological nerve vectors:
    - Optic Nerve CN II: 64-dimensional retinotopic piece, control, and positional grid.
    - Spinal Segment C5: Interoceptive material balance (DCML) and King distress/pain (STT).
    """

    def __init__(self, board: Optional[chess.Board] = None):
        self.board = board if board is not None else chess.Board()

    def set_board(self, board: chess.Board) -> None:
        self.board = board

    def get_material_balance(self, perspective_color: chess.Color = chess.WHITE) -> float:
        """Returns net material balance in pawn units from perspective of specified color."""
        white_mat = sum(
            len(self.board.pieces(pt, chess.WHITE)) * val
            for pt, val in PIECE_VALUES.items()
            if pt != chess.KING
        )
        black_mat = sum(
            len(self.board.pieces(pt, chess.BLACK)) * val
            for pt, val in PIECE_VALUES.items()
            if pt != chess.KING
        )
        net = white_mat - black_mat
        return float(net if perspective_color == chess.WHITE else -net)

    def is_king_threatened(self, perspective_color: chess.Color = chess.WHITE) -> float:
        """
        Computes somatic King danger score:
        - 1.0 if King is currently in check.
        - +0.2 for each enemy piece attacking adjacent King squares.
        """
        threat = 0.0
        # Check status
        if self.board.is_check() and self.board.turn == perspective_color:
            threat += 1.0

        king_sq = self.board.king(perspective_color)
        if king_sq is not None:
            enemy_color = not perspective_color
            adj_squares = [
                sq for sq in chess.SQUARES
                if chess.square_distance(king_sq, sq) == 1
            ]
            for sq in adj_squares:
                attackers = self.board.attackers(enemy_color, sq)
                threat += len(attackers) * 0.15

        return float(np.clip(threat, 0.0, 3.0))

    def encode_retina(self, perspective_color: chess.Color = chess.WHITE) -> np.ndarray:
        """
        Constructs 64-dimensional retinotopic visual vector for Optic Nerve (CN II).
        Each square (0 to 63) maps 1:1 to a feature channel:
        - Piece identity & value
        - Positional square bonus
        - Square attack / control gradient
        """
        retina = np.zeros(64, dtype=np.float32)

        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            sq_val = 0.0

            if piece is not None:
                base_val = PIECE_VALUES[piece.piece_type]
                # Scale king representation for normalization
                if piece.piece_type == chess.KING:
                    base_val = 15.0

                # Positional table bonus
                pst_bonus = 0.0
                if piece.piece_type == chess.KNIGHT:
                    pst_idx = sq if piece.color == chess.WHITE else chess.square_mirror(sq)
                    pst_bonus = KNIGHT_PST[pst_idx]
                elif piece.piece_type == chess.PAWN:
                    pst_idx = sq if piece.color == chess.WHITE else chess.square_mirror(sq)
                    pst_bonus = PAWN_PST[pst_idx]

                piece_score = base_val + pst_bonus
                sq_val = piece_score if piece.color == chess.WHITE else -piece_score

            # Square control / attack gradient
            w_attacks = len(self.board.attackers(chess.WHITE, sq))
            b_attacks = len(self.board.attackers(chess.BLACK, sq))
            control = (w_attacks - b_attacks) * 0.15
            sq_val += control

            # Orient from perspective of current player
            retina[sq] = sq_val if perspective_color == chess.WHITE else -sq_val

        # Normalize to reasonable range [-3.0, 3.0]
        return np.clip(retina * 0.3, -3.0, 3.0).astype(np.float32)

    def encode_somatic(self, perspective_color: chess.Color = chess.WHITE) -> np.ndarray:
        """
        Constructs 64-dimensional interoceptive vector for Spinal Segment C5:
        - Index 0: Net material balance (DCML proprioceptive tension)
        - Index 1: King danger score (Spinothalamic distress & pain)
        - Index 2: Mobility ratio
        - Index 3: Center control
        """
        somatic = np.zeros(64, dtype=np.float32)

        # 1. Material balance normalized by max piece material (39 pawns)
        mat = self.get_material_balance(perspective_color)
        somatic[0] = float(np.clip(mat / 39.0, -1.0, 1.0))

        # 2. King threat / check pain
        danger = self.is_king_threatened(perspective_color)
        somatic[1] = float(np.clip(danger, 0.0, 2.0))

        # 3. Legal move mobility ratio
        num_moves = self.board.legal_moves.count()
        somatic[2] = float(np.clip(num_moves / 40.0, 0.0, 2.0))

        # 4. Central squares occupancy (e4, d4, e5, d5)
        center_sqs = [chess.E4, chess.D4, chess.E5, chess.D5]
        center_control = sum(
            1.0 if self.board.piece_at(sq) and self.board.piece_at(sq).color == perspective_color else 0.0
            for sq in center_sqs
        )
        somatic[3] = float(center_control / 4.0)

        return somatic
