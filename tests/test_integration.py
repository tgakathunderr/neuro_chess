import sys
import os
from pathlib import Path

# Ensure root directory and BIB-2 are on sys.path
_ROOT = str(Path(__file__).resolve().parent.parent.parent)
_BIB2_DIR = os.path.join(_ROOT, "BIB-2")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _BIB2_DIR not in sys.path:
    sys.path.insert(0, _BIB2_DIR)

import pytest
import chess
from bib2.brain import BIB2NervousSystem
from neuro_chess.adapter import ChessAdapter
from neuro_chess.training import GrandmasterTrainer


def test_full_game_simulation_15_moves():
    """Simulates a 15-move game between two players/brain to verify stability."""
    brain = BIB2NervousSystem(seed=42)
    adapter = ChessAdapter(brain)
    trainer = GrandmasterTrainer(brain, adapter)

    # Imprint a few puzzles first
    trainer.imprint_grandmaster_motifs(limit=5)

    board = chess.Board()
    for move_num in range(15):
        if board.is_game_over():
            break
        color = board.turn
        move = adapter.select_move(board, color)
        assert move in board.legal_moves
        board.push(move)

    assert len(board.move_stack) == 15
    activations = adapter.get_brain_activations()
    assert all(0.0 <= v <= 1.0 for v in activations.values())
