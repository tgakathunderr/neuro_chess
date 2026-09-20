import sys
from pathlib import Path

# Ensure root directory is on sys.path
_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pytest
import chess
from neuro_chess.board import ChessBoardSensory


def test_initial_board_retina_encoding():
    board = chess.Board()
    sensory = ChessBoardSensory(board)
    retina = sensory.encode_retina(chess.WHITE)
    assert isinstance(retina, np.ndarray)
    assert retina.shape == (64,)
    assert np.all(np.isfinite(retina))

    # E4 and D4 should be empty initially
    assert retina[chess.E4] == 0.0

    # E1 has white King, E8 has black King
    assert retina[chess.E1] > 0.0
    assert retina[chess.E8] < 0.0


def test_somatic_interoception_and_king_threat():
    board = chess.Board()
    sensory = ChessBoardSensory(board)
    somatic = sensory.encode_somatic(chess.WHITE)
    assert somatic.shape == (64,)
    # Material starts even
    assert somatic[0] == 0.0
    assert somatic[1] == 0.0  # Not in check

    # Put white in Scholar's Mate / Fool's mate check
    board.set_fen("rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    somatic_check = sensory.encode_somatic(chess.WHITE)
    assert somatic_check[1] >= 1.0  # King in check!


def test_material_and_mobility():
    board = chess.Board()
    sensory = ChessBoardSensory(board)
    mat = sensory.get_material_balance(chess.WHITE)
    assert mat == 0.0

    # Remove black queen
    board.remove_piece_at(chess.D8)
    mat_up = sensory.get_material_balance(chess.WHITE)
    assert mat_up == 9.0
