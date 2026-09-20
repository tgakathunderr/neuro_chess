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
import numpy as np
from bib2.brain import BIB2NervousSystem
from neuro_chess.adapter import ChessAdapter


def test_adapter_selects_legal_move():
    brain = BIB2NervousSystem(seed=42)
    adapter = ChessAdapter(brain)
    board = chess.Board()

    move = adapter.select_move(board, chess.WHITE)
    assert move in board.legal_moves
    assert isinstance(move, chess.Move)


def test_adapter_brain_activations_and_telemetry():
    brain = BIB2NervousSystem(seed=42)
    adapter = ChessAdapter(brain)
    board = chess.Board()
    adapter.select_move(board, chess.WHITE)

    acts = adapter.get_brain_activations()
    assert "v1" in acts and "dlpfc" in acts and "m1" in acts
    assert "hippocampus" in acts and "basal_ganglia" in acts
    assert "cerebellum" in acts and "amygdala" in acts and "thalamus" in acts
    assert all(0.0 <= v <= 1.0 for v in acts.values())

    telemetry = adapter.get_telemetry()
    assert "heart_rate" in telemetry
    assert "dopamine" in telemetry
    assert "cortisol" in telemetry
    assert 50 <= telemetry["heart_rate"] <= 160


def test_adapter_plasticity_on_reward():
    brain = BIB2NervousSystem(seed=42)
    adapter = ChessAdapter(brain)
    board = chess.Board()
    initial_da = brain.chemistry.matrix.state.dopamine

    # Reinforce a move
    snapshot = adapter.sensory.encode_retina(chess.WHITE)
    adapter.reinforce_move(reward_rpe=0.6, board_snapshot=snapshot)

    assert brain.chemistry.matrix.state.dopamine >= initial_da
    assert len(brain.limbic.hippocampus.ca3.stored_patterns) > 0

    # Negative punishment
    adapter.reinforce_move(reward_rpe=-0.6, board_snapshot=snapshot)
    assert brain.peripheral.autonomic.sympathetic_tone > 0.2


def test_adapter_hippocampal_ablation_queen_sacrifice():
    """
    Verifies that Hippocampal CA3 attractor actively overrules heuristic loss aversion:
    In a back-rank queen sacrifice position (e2e8# after Rxe8 Rxe8#), pure heuristics avoids
    hanging the Queen (-9.0 penalty), while CA3 pattern completion (+15.0) executes the sacrifice.
    """
    brain = BIB2NervousSystem(seed=42)
    adapter = ChessAdapter(brain)

    # Position: White to move, e2e8 is the back-rank queen sac
    fen = "3r2k1/5ppp/8/8/8/8/4QPPP/6K1 w - - 0 1"
    board = chess.Board(fen)
    adapter.sensory.set_board(board)

    # Imprint winning pattern into CA3 attractor
    board.push_uci("e2e8")
    sac_snapshot = adapter.sensory.encode_retina(chess.WHITE)
    adapter.reinforce_move(reward_rpe=1.0, board_snapshot=sac_snapshot)
    board.pop()

    # With CA3 active: CA3 attractor completion triggers the queen sacrifice
    move_with_ca3 = adapter.select_move(board, chess.WHITE, use_hippocampus=True)
    assert move_with_ca3 == chess.Move.from_uci("e2e8")

    # With CA3 ablated: pure heuristics refuses to sacrifice the Queen
    move_without_ca3 = adapter.select_move(board, chess.WHITE, use_hippocampus=False)
    assert move_without_ca3 != chess.Move.from_uci("e2e8")
