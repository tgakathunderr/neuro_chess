"""Tests for ChessHUD rendering and pacing mechanics."""

import os
import sys
from pathlib import Path
import pytest
import chess

# Ensure root directory and BIB-2 are on path
_ROOT = str(Path(__file__).resolve().parent.parent.parent)
_BIB2_DIR = os.path.join(_ROOT, "BIB-2")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _BIB2_DIR not in sys.path:
    sys.path.insert(0, _BIB2_DIR)

# Set headless SDL dummy video driver for test environments
os.environ["SDL_VIDEODRIVER"] = "dummy"

from bib2.brain import BIB2NervousSystem
from neuro_chess.adapter import ChessAdapter
from neuro_chess.hud import ChessHUD


def test_hud_render_self_play_parameters():
    """Verify ChessHUD renders cleanly with self-play and pacing flags."""
    brain = BIB2NervousSystem(seed=42)
    adapter = ChessAdapter(brain)
    board = chess.Board()

    hud = ChessHUD(width=1360, height=820)
    acts = adapter.get_brain_activations()
    telemetry = adapter.get_telemetry()

    # Test standard self-play render
    ok = hud.render(
        board=board,
        human_color=chess.WHITE,
        brain_activations=acts,
        telemetry=telemetry,
        last_move=chess.Move.from_uci("e2e4"),
        status_msg="Move #1: White played e4 (e2e4)",
        self_play=True,
        is_paused=False,
        move_delay=1.8,
        time_until_next=1.2,
    )
    assert ok is True

    # Test paused self-play render
    ok_paused = hud.render(
        board=board,
        human_color=chess.WHITE,
        brain_activations=acts,
        telemetry=telemetry,
        last_move=chess.Move.from_uci("e7e5"),
        status_msg="PAUSED. Press [SPACE] to resume.",
        self_play=True,
        is_paused=True,
        move_delay=2.5,
        time_until_next=0.0,
    )
    assert ok_paused is True


def test_san_notation_generation():
    """Verify Standard Algebraic Notation generation matches expectations."""
    board = chess.Board()
    m1 = chess.Move.from_uci("e2e4")
    san1 = board.san(m1)
    assert san1 == "e4"
    board.push(m1)

    m2 = chess.Move.from_uci("e7e5")
    san2 = board.san(m2)
    assert san2 == "e5"
    board.push(m2)

    m3 = chess.Move.from_uci("g1f3")
    san3 = board.san(m3)
    assert san3 == "Nf3"
    board.push(m3)


def test_game_logger_pgn_and_file_export():
    """Verify GameLogger records moves, exports valid PGN, and writes logs."""
    import shutil
    from neuro_chess.logger import GameLogger

    tmp_dir = os.path.join(os.path.dirname(__file__), "test_tmp_logs")
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        logger = GameLogger(white_name="BIB-2 White", black_name="BIB-2 Black")
        board = chess.Board()

        m1 = chess.Move.from_uci("e2e4")
        logger.record_move(board, m1, player_name="White", activations={"v1": 0.8}, telemetry={"heart_rate": 72})
        board.push(m1)

        m2 = chess.Move.from_uci("e7e5")
        logger.record_move(board, m2, player_name="Black", activations={"dlpfc": 0.7}, telemetry={"heart_rate": 75})
        board.push(m2)

        logger.set_result("1/2-1/2")

        # Check PGN
        pgn_str = logger.to_pgn()
        assert "BIB-2 White" in pgn_str
        assert "BIB-2 Black" in pgn_str
        assert "1. e4 e5" in pgn_str

        # Check text log
        txt_log = logger.to_text_log()
        assert "GAME LOG" in txt_log
        assert "1.       White" in txt_log

        # Check recent moves string
        recent = logger.get_recent_moves_san(4)
        assert "1. e4" in recent
        assert "e5" in recent

        # Check file persistence
        pgn_path, txt_path = logger.save_to_file(base_dir=tmp_dir)
        assert os.path.exists(pgn_path)
        assert os.path.exists(txt_path)
        with open(pgn_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "1. e4 e5" in content

        # Check clipboard copy method executes without crash
        ok, msg = logger.copy_to_clipboard()
        assert isinstance(ok, bool)
        assert isinstance(msg, str)
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def test_hud_render_fast_play_and_copy_feedback():
    """Verify ChessHUD renders fast turbo mode and copy feedback banner."""
    brain = BIB2NervousSystem(seed=42)
    adapter = ChessAdapter(brain)
    board = chess.Board()

    hud = ChessHUD(width=1360, height=820)
    acts = adapter.get_brain_activations()
    telemetry = adapter.get_telemetry()

    ok = hud.render(
        board=board,
        human_color=chess.WHITE,
        brain_activations=acts,
        telemetry=telemetry,
        last_move=chess.Move.from_uci("e2e4"),
        status_msg="Move #1: White played e4 (e2e4)",
        self_play=True,
        is_paused=False,
        move_delay=0.08,
        time_until_next=0.02,
        recent_moves="1. e4 e5  2. Nf3",
        copied_feedback="✓ COPIED! Game PGN & Move Log copied to clipboard!",
    )
    assert ok is True


def test_elo_benchmark_computation():
    """Verify Elo benchmark runs gauntlet and produces valid statistical rating metrics."""
    from neuro_chess.elo_benchmark import EloBenchmark

    brain = BIB2NervousSystem(seed=42)
    adapter = ChessAdapter(brain)
    bench = EloBenchmark(brain, adapter)

    # Fast test with 1 game per opponent
    res = bench.run_full_benchmark(games_per_opp=1)
    assert "composite_elo" in res
    assert "performance_rating" in res
    assert "tactical_elo" in res
    assert 400 <= res["composite_elo"] <= 2800
    assert 0.0 <= res["tactical_accuracy"] <= 100.0
