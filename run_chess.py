"""Main CLI entrypoint for BIB-2 Neuro-Chess Grandmaster."""

import argparse
import os
import sys
import time
from typing import Dict, Any, Optional
import numpy as np
import chess

# Ensure root directory and BIB-2 are on path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BIB2_PATH = os.path.join(_ROOT, "BIB-2")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _BIB2_PATH not in sys.path:
    sys.path.insert(0, _BIB2_PATH)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from bib2.brain import BIB2NervousSystem
from neuro_chess.adapter import ChessAdapter
from neuro_chess.training import GrandmasterTrainer
from neuro_chess.logger import GameLogger


def run_training_curriculum(puzzles: int = 24, spar_games: int = 5, save_path: str = None) -> None:
    """Trains BIB-2 on grandmaster tactical motifs, sparring matches, and SWS sleep consolidation."""
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), "models", "master_brain.npz")

    print("=" * 80)
    print("[CHESS] BIB-2 GRANDMASTER CURRICULUM TRAINING")
    print(f"Puzzles: {puzzles}  |  Sparring Games: {spar_games}  |  Target: {save_path}")
    print("=" * 80)

    brain = BIB2NervousSystem(seed=42)
    adapter = ChessAdapter(brain)
    trainer = GrandmasterTrainer(brain, adapter)

    t0 = time.time()

    # 1. Imprint Grandmaster Puzzles & Openings
    print(f"[STAGE 1] Imprinting {puzzles} Grandmaster tactical puzzle motifs into Hippocampal CA3...")
    n_imprinted = trainer.imprint_grandmaster_motifs(limit=puzzles)
    print(f"          Successfully imprinted {n_imprinted} tactical patterns.")

    # 2. High-Speed Autonomous Sparring
    print(f"[STAGE 2] Running {spar_games} autonomous sparring matches against evaluator bot...")
    for g_idx in range(spar_games):
        result = trainer.spar_match(max_moves=40)
        print(f"          Game {g_idx + 1}/{spar_games}: {result['moves_played']} moves -> Winner: {result['winner']}")

    # 3. Nocturnal Slow-Wave Sleep Consolidation
    print("[STAGE 3] Entering Slow-Wave Sleep (SWS): Sharp-Wave Ripples & Tononi SHY Downscaling...")
    sleep_stats = trainer.consolidate_sleep(n_sleep_ticks=15)
    print(f"          Consolidated {sleep_stats['replayed_count']} winning episodes; Synaptic Downscaling Applied.")

    # 4. Save Master Brain Checkpoint
    trainer.save_checkpoint(save_path)
    dt = time.time() - t0
    print(f"[SAVED] Master Brain Checkpoint saved to: {save_path} ({dt:.2f}s total)")
    print("=" * 80 + "\n")


def play_interactive_game(
    player_color: chess.Color = chess.WHITE,
    model_path: str = None,
    self_play: bool = False,
    move_delay: Optional[float] = None,
    fast_mode: bool = False,
) -> None:
    """Launches interactive Pygame Chessboard HUD with 2D Brain Activation Heatmap."""
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "models", "master_brain.npz")

    # Default pacing: 0.08s for fast mode (like original), 1.8s for readable self-play, 0.4s for vs human
    if fast_mode:
        move_delay = 0.08
    elif move_delay is None:
        move_delay = 1.8 if self_play else 0.4

    # Auto-train master brain if model doesn't exist yet
    if not os.path.exists(model_path):
        print("[NOTICE] No pre-trained master brain found. Running initial Grandmaster Curriculum training...")
        run_training_curriculum(puzzles=24, spar_games=3, save_path=model_path)

    import pygame
    from neuro_chess.hud import ChessHUD

    brain = BIB2NervousSystem(seed=42)
    adapter = ChessAdapter(brain)
    trainer = GrandmasterTrainer(brain, adapter)

    # Load master brain checkpoint
    loaded = trainer.load_checkpoint(model_path)
    if loaded:
        print(f"[LOADED] Master Grandmaster Brain loaded from: {model_path}")
        print(f"         Hippocampal CA3 memory banks: {len(brain.limbic.hippocampus.ca3.stored_patterns)} tactical chunks.")

    hud = ChessHUD(width=1360, height=820)
    board = chess.Board()
    human_color = player_color
    last_move = None

    white_title = "BIB-2 Grandmaster (White)" if self_play else ("Human Challenger" if human_color == chess.WHITE else "BIB-2 Grandmaster")
    black_title = "BIB-2 Grandmaster (Black)" if self_play else ("BIB-2 Grandmaster" if human_color == chess.WHITE else "Human Challenger")
    logger = GameLogger(white_name=white_title, black_name=black_title)

    if self_play:
        pace_label = "TURBO FAST (0.08s)" if move_delay <= 0.15 else f"{move_delay:.1f}s/move"
        status_msg = f"Self-Play Started. Pacing: {pace_label}. [TAB] Fast/Slow | [C] Copy PGN"
    else:
        status_msg = "Game Started. Your move! Press [C] anytime to copy PGN."

    is_paused = False
    step_once = False
    last_move_time = time.time()
    last_slow_delay = 1.8 if move_delay <= 0.15 else move_delay
    copied_feedback = ""
    copied_feedback_time = 0.0

    running = True
    while running:
        # Clear copy confirmation badge after 3.5 seconds
        if copied_feedback and (time.time() - copied_feedback_time > 3.5):
            copied_feedback = ""

        # 1. Event Handling (keyboard & mouse)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    running = False
                    break
                elif event.key == pygame.K_SPACE:
                    is_paused = not is_paused
                    if is_paused:
                        status_msg = "PAUSED. Press [SPACE] to resume or [->/N] to step 1 move."
                    else:
                        status_msg = f"Resumed. Speed: {move_delay:.2f}s/move."
                elif event.key in [pygame.K_RIGHT, pygame.K_n]:
                    if self_play:
                        step_once = True
                        is_paused = True
                        status_msg = "Stepping 1 move forward..."
                elif event.key == pygame.K_TAB:
                    # Toggle between Fast Mode (0.08s - original speed) and Readable Pace (1.8s)
                    if move_delay > 0.2:
                        last_slow_delay = move_delay
                        move_delay = 0.08
                        status_msg = "TURBO FAST PLAY enabled (0.08s/move) [Press TAB to slow down]."
                    else:
                        move_delay = last_slow_delay if last_slow_delay > 0.2 else 1.8
                        status_msg = f"READABLE PACE restored ({move_delay:.1f}s/move) [Press TAB for fast]."
                elif event.key == pygame.K_c:
                    # Copy Game PGN & Move Log to Clipboard
                    ok, msg = logger.copy_to_clipboard()
                    pgn_file, txt_file = logger.save_to_file()
                    copied_feedback = "Game PGN & Move Log copied to clipboard! (Ctrl+V to paste)"
                    copied_feedback_time = time.time()
                    status_msg = copied_feedback
                    print("\n" + "=" * 70)
                    print(f"[COPIED] {msg}")
                    print(f"[SAVED] PGN: {pgn_file}")
                    print(f"        TXT: {txt_file}")
                    print("=" * 70 + "\n")
                elif event.key == pygame.K_1:
                    move_delay = 2.5
                    status_msg = "Pacing set to RELAXED (2.5s / move)."
                elif event.key == pygame.K_2:
                    move_delay = 1.8
                    status_msg = "Pacing set to STANDARD (1.8s / move)."
                elif event.key == pygame.K_3:
                    move_delay = 1.0
                    status_msg = "Pacing set to BRISK (1.0s / move)."
                elif event.key == pygame.K_4:
                    move_delay = 0.4
                    status_msg = "Pacing set to FAST (0.4s / move)."
                elif event.key in [pygame.K_5, pygame.K_0]:
                    move_delay = 0.08
                    status_msg = "Pacing set to TURBO FAST (0.08s / move - original speed)."
                elif event.key in [pygame.K_UP, pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS]:
                    move_delay = min(5.0, round(move_delay + (0.05 if move_delay < 0.3 else 0.25), 2))
                    status_msg = f"Delay increased to {move_delay:.2f}s / move."
                elif event.key in [pygame.K_DOWN, pygame.K_MINUS, pygame.K_KP_MINUS]:
                    move_delay = max(0.02, round(move_delay - (0.05 if move_delay <= 0.3 else 0.25), 2))
                    status_msg = f"Delay decreased to {move_delay:.2f}s / move."
                elif event.key == pygame.K_r:
                    logger = GameLogger(white_name=white_title, black_name=black_title)
                    board.reset()
                    last_move = None
                    last_move_time = time.time()
                    status_msg = "Board reset. Fresh game!"
                elif event.key == pygame.K_f:
                    if not self_play:
                        human_color = not human_color
                        status_msg = f"Flipped color. You are now {'WHITE' if human_color == chess.WHITE else 'BLACK'}."
                    else:
                        # In self-play, F acts as shortcut for fast play toggle
                        if move_delay > 0.2:
                            last_slow_delay = move_delay
                            move_delay = 0.08
                            status_msg = "TURBO FAST PLAY enabled (0.08s/move) [Press F/TAB to slow down]."
                        else:
                            move_delay = last_slow_delay if last_slow_delay > 0.2 else 1.8
                            status_msg = f"READABLE PACE restored ({move_delay:.1f}s/move)."
                elif event.key == pygame.K_t:
                    status_msg = "Running quick sparring epoch..."
                    trainer.spar_match(max_moves=20)
                    trainer.consolidate_sleep()
                    status_msg = "Sparring & SWS sleep complete!"

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Human piece interaction
                if not self_play and board.turn == human_color and not board.is_game_over():
                    chosen_move = hud.handle_click(event.pos, board, human_color)
                    if chosen_move and chosen_move in board.legal_moves:
                        san_move = logger.record_move(
                            board_before=board,
                            move=chosen_move,
                            player_name="You",
                            activations=adapter.get_brain_activations(),
                            telemetry=adapter.get_telemetry(),
                        )
                        board.push(chosen_move)
                        last_move = chosen_move
                        last_move_time = time.time()
                        logger.save_to_file()
                        status_msg = f"You played: {san_move} ({chosen_move.uci()})"

        if not running:
            break

        # 2. Handle Non-Blocking Bot Move / Self-Play Timing
        now = time.time()
        elapsed = now - last_move_time
        time_until_next = 0.0

        if not board.is_game_over():
            if self_play:
                if not is_paused:
                    time_until_next = max(0.0, move_delay - elapsed)
                    if elapsed >= move_delay:
                        color_name = "White" if board.turn == chess.WHITE else "Black"
                        bot_move = adapter.select_move(board, board.turn)
                        if bot_move and bot_move in board.legal_moves:
                            san_move = logger.record_move(
                                board_before=board,
                                move=bot_move,
                                player_name=color_name,
                                activations=adapter.get_brain_activations(),
                                telemetry=adapter.get_telemetry(),
                            )
                            board.push(bot_move)
                            last_move = bot_move
                            last_move_time = time.time()
                            time_until_next = move_delay
                            logger.save_to_file()
                            status_msg = f"Move #{board.fullmove_number}: {color_name} played {san_move} ({bot_move.uci()})"
                elif step_once:
                    color_name = "White" if board.turn == chess.WHITE else "Black"
                    bot_move = adapter.select_move(board, board.turn)
                    if bot_move and bot_move in board.legal_moves:
                        san_move = logger.record_move(
                            board_before=board,
                            move=bot_move,
                            player_name=color_name,
                            activations=adapter.get_brain_activations(),
                            telemetry=adapter.get_telemetry(),
                        )
                        board.push(bot_move)
                        last_move = bot_move
                        last_move_time = time.time()
                        logger.save_to_file()
                        status_msg = f"Step #{board.fullmove_number}: {color_name} played {san_move} ({bot_move.uci()})"
                    step_once = False
            else:
                # Playing against human
                if board.turn != human_color:
                    time_until_next = max(0.0, move_delay - elapsed)
                    if elapsed >= move_delay:
                        bot_move = adapter.select_move(board, board.turn)
                        if bot_move and bot_move in board.legal_moves:
                            san_move = logger.record_move(
                                board_before=board,
                                move=bot_move,
                                player_name="BIB-2",
                                activations=adapter.get_brain_activations(),
                                telemetry=adapter.get_telemetry(),
                            )
                            board.push(bot_move)
                            last_move = bot_move
                            last_move_time = time.time()
                            logger.save_to_file()
                            status_msg = f"BIB-2 played: {san_move} ({bot_move.uci()})"

        # 3. Check Game Over
        if board.is_game_over():
            if board.is_checkmate():
                winner = "White" if board.turn == chess.BLACK else "Black"
                logger.set_result("1-0" if board.turn == chess.BLACK else "0-1")
                status_msg = f"CHECKMATE! {winner} Wins! Press [C] to copy PGN | [R] to reset."
            elif board.is_stalemate():
                logger.set_result("1/2-1/2")
                status_msg = "Draw by Stalemate! Press [C] to copy PGN | [R] to reset."
            elif board.is_insufficient_material():
                logger.set_result("1/2-1/2")
                status_msg = "Draw by Insufficient Material! Press [C] to copy PGN | [R] to reset."
            else:
                logger.set_result("1/2-1/2")
                status_msg = "Game Over! Press [C] to copy PGN | [R] to reset."
            logger.save_to_file()

        # 4. Render HUD with Real-Time Brain Heatmap & ECG
        acts = adapter.get_brain_activations()
        telemetry = adapter.get_telemetry()

        alive = hud.render(
            board=board,
            human_color=human_color,
            brain_activations=acts,
            telemetry=telemetry,
            last_move=last_move,
            status_msg=status_msg,
            self_play=self_play,
            is_paused=is_paused,
            move_delay=move_delay,
            time_until_next=time_until_next,
            recent_moves=logger.get_recent_moves_san(6),
            copied_feedback=copied_feedback,
        )
        if not alive:
            break

    pygame.quit()
    print("[CHESS] Session cleanly closed.")


def run_elo_benchmark(model_path: Optional[str] = None, games_per_opp: int = 4) -> Dict[str, Any]:
    """Runs complete empirical Elo benchmark across tactical suite and gauntlet matches."""
    from neuro_chess.elo_benchmark import compute_bib2_elo

    print("=" * 78)
    print("            BIB-2 EMPIRICAL CHESS ELO RATING BENCHMARK EVALUATION")
    print("=" * 78)
    print("[1] Standardized Tactical Benchmark Suite (1000 - 1750 Elo positions)")
    print("[2] Tournament Gauntlet vs Calibrated Opponents (400, 750, 1150, 1400 Elo)")
    print("[3] FIDE Maximum Likelihood Performance Rating Calculation")
    print("-" * 78)
    print("Running tournament games & tactical evaluations...")

    report = compute_bib2_elo(model_path=model_path, games_per_opp=games_per_opp)

    comp_elo = report["composite_elo"]
    ci = report["ci_95"]
    perf = report["performance_rating"]
    tact_elo = report["tactical_elo"]
    tact_acc = report["tactical_accuracy"]
    win_rate = report["match_win_rate"]
    dt = report["elapsed_seconds"]

    print("\n" + "=" * 78)
    print(f"  OFFICIAL BIB-2 ELO RATING:  {comp_elo} +/- {ci} Elo (95% CI)")
    print(f"  FIDE Performance Rating:    {perf} Elo")
    print(f"  Tactical Rating:            {tact_elo} Elo ({tact_acc:.1f}% solve rate)")
    print(f"  Gauntlet Win Rate:          {win_rate:.1f}% ({report['total_games']} games in {dt:.2f}s)")
    print("=" * 78)

    print("\n[TOURNAMENT OPPONENT BREAKDOWN]")
    print(f"{'Opponent Engine':<28} {'Base Elo':<10} {'Record (W-D-L)':<18} {'Score':<10} {'Score %'}")
    print("-" * 78)
    for name, stat in report["gauntlet_details"]["opponents"].items():
        rec = f"{stat['wins']}W - {stat['draws']}D - {stat['losses']}L"
        sc = f"{stat['score']}/{games_per_opp}"
        print(f"{name:<28} {stat['elo']:<10} {rec:<18} {sc:<10} {stat['win_pct']:.1f}%")

    print("\n[TACTICAL SOLVE RATE BY ELO TIER]")
    print(f"{'Rating Tier':<18} {'Solved / Total':<18} {'Accuracy'}")
    print("-" * 78)
    for tier, tstat in report["tactical_details"]["tier_breakdown"].items():
        ratio = f"{tstat['correct']} / {tstat['total']}"
        pct = (tstat['correct'] / tstat['total'] * 100.0) if tstat['total'] > 0 else 0.0
        print(f"{tier:<18} {ratio:<18} {pct:.1f}%")

    if "ablation_details" in report:
        abl = report["ablation_details"]
        act = abl["ca3_active"]
        dis = abl["ca3_ablated"]
        print("\n" + "=" * 78)
        print("          HIPPOCAMPAL CA3 TACTICAL ABLATION EXPERIMENT")
        print("=" * 78)
        print(f"Condition:               Solved / Total      Accuracy      Tactical Elo")
        print("-" * 78)
        print(f"Pure Heuristics (No CA3): {dis['solved']:>2} / {dis['total']:<2}           {dis['accuracy_pct']:5.1f}%       {dis['tactical_elo']} Elo")
        print(f"BIB-2 Full (CA3 Active):  {act['solved']:>2} / {act['total']:<2}           {act['accuracy_pct']:5.1f}%       {act['tactical_elo']} Elo")
        print("-" * 78)
        print(f"CA3 Attribution Delta:    +{abl['accuracy_delta']:.1f}% Accuracy  |  +{abl['elo_delta']} Tactical Elo")
        if abl.get("puzzles_solved_by_ca3_only"):
            print("Tactical Puzzles Enabled Exclusively by CA3 Attractor:")
            for motif in abl["puzzles_solved_by_ca3_only"]:
                print(f"  * {motif} (Overruled heuristic material penalty / saw checkmate)")
        print("=" * 78)

    print("=" * 78 + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="BIB-2 Neuro-Chess Grandmaster")
    parser.add_argument("--train", action="store_true", help="Run Grandmaster curriculum training (puzzles + sparring + sleep)")
    parser.add_argument("--play", action="store_true", help="Launch interactive Cybernetic Chessboard HUD vs BIB-2")
    parser.add_argument("--self-play", action="store_true", help="Watch two BIB-2 brains play each other visually")
    parser.add_argument("--fast", action="store_true", help="Launch in fast play mode (0.08s / move, like original)")
    parser.add_argument("--elo", "--benchmark", action="store_true", help="Run empirical Elo rating benchmark tournament")
    parser.add_argument("--games", type=int, default=4, help="Number of games per opponent in Elo benchmark (default: 4)")
    parser.add_argument("--speed", "--delay", type=float, default=None, help="Move delay in seconds (default: 1.8s for self-play, 0.4s for interactive)")
    parser.add_argument("--color", type=str, default="white", choices=["white", "black"], help="Player color (default: white)")
    parser.add_argument("--puzzles", type=int, default=24, help="Number of grandmaster puzzles to imprint")
    parser.add_argument("--spar", type=int, default=5, help="Number of sparring matches")
    parser.add_argument("--model", type=str, default=None, help="Custom brain checkpoint path")
    args = parser.parse_args()

    if args.elo:
        run_elo_benchmark(model_path=args.model, games_per_opp=args.games)
    elif args.train:
        run_training_curriculum(puzzles=args.puzzles, spar_games=args.spar, save_path=args.model)
    elif args.self_play:
        play_interactive_game(self_play=True, model_path=args.model, move_delay=args.speed, fast_mode=args.fast)
    else:
        # Default to play interactive
        c = chess.WHITE if args.color == "white" else chess.BLACK
        play_interactive_game(player_color=c, model_path=args.model, move_delay=args.speed, fast_mode=args.fast)


if __name__ == "__main__":
    main()
