"""Grandmaster Training Curriculum: Puzzle imprinting, sparring, and SWS sleep consolidation."""

import os
import sys
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import chess

# Ensure BIB-2 library is available
_BIM2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BIB2_PATH = os.path.join(_BIM2_ROOT, "BIB-2")
if _BIB2_PATH not in sys.path:
    sys.path.insert(0, _BIB2_PATH)

from bib2.brain import BIB2NervousSystem
from bib2.sleep.orchestrator import SleepStage
from .adapter import ChessAdapter
from .board import ChessBoardSensory, PIECE_VALUES


# Classic Grandmaster Tactical Motifs & Puzzles (FEN, Best Move, Motif Name)
GRANDMASTER_PUZZLES: List[Tuple[str, str, str]] = [
    # 1. Back-Rank Mates
    ("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "e1e8", "Back-Rank Mate"),
    ("3r2k1/5ppp/8/8/8/8/4QPPP/6K1 w - - 0 1", "e2e8", "Back-Rank Queen Sac"),
    ("6k1/3R1ppp/8/8/8/8/5PPP/6K1 w - - 0 1", "d7d8", "Rook Infiltration Mate"),

    # 2. Scholar's & Opening Traps
    ("r1bqkb1r/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4", "f3f7", "Scholar's Mate Strike"),
    ("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "c2c3", "Italian Game Center Prep"),
    ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5", "d2d3", "Italian Game Development"),

    # 3. Royal Knight Forks
    ("r1b1k2r/pppp1ppp/8/2b1N3/4n2q/8/PPPP1PPP/RNBQ1RK1 w kq - 0 7", "d2d4", "Center Counter Fork"),
    ("r1bqk2r/pppp1ppp/8/4n3/2B1P3/8/PPP2PPP/RNBQK2R w KQkq - 0 6", "c4b3", "Bishop Retreat"),
    ("r1b1k2r/ppppqppp/2n5/4N3/2B1P3/8/PPP2PPP/RNBQK2R w KQkq - 0 6", "e5c6", "Knight Capture Swap"),

    # 4. Greek Gift Sacrifice & Kingside Attacks
    ("r1bq1rk1/ppp2ppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 6", "c4d5", "Center Tension Release"),
    ("r1b2rk1/pp1nqppp/2p1pn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQK2R w KQ - 0 8", "e1g1", "Kingside Castling Safety"),
    ("r1bq1rk1/1pp2ppp/p1np1n2/2b1p3/2B1P3/2PP1N2/PP1N1PPP/R1BQ1RK1 w - - 0 8", "f1e1", "Rook Centralization"),

    # 5. Queen's Gambit Lines
    ("rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 1 3", "c4d5", "Queen's Gambit Exchange"),
    ("r1bqkb1r/pp1npppp/2p2n2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 2 5", "e2e3", "Solidify Pawn Center"),
    ("rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 1", "e7e6", "Nimzo-Indian Setup"),

    # 6. Sicilian Defense Motifs
    ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "g1f3", "Open Sicilian Prep"),
    ("r1bqkbnr/pp1ppppp/2n5/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq - 0 4", "g8f6", "Knight Counter-Attack"),
    ("rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 2 5", "a7a6", "Najdorf Defense Hook"),

    # 7. Discovered Attacks & Pins
    ("r1b1k2r/pp2bppp/2n1pn2/q1pp4/2PP4/2NBPN2/PP3PPP/R1BQK2R w KQkq - 3 7", "e1g1", "Castle Away from Pin"),
    ("r1bqk2r/pppp1ppp/2n5/4P3/1bB1n3/2N2N2/PPP2PPP/R1BQK2R w KQkq - 0 7", "c1d2", "Unpin Knight"),
    ("r2q1rk1/ppp2ppp/2n1bn2/3p4/3P4/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 9", "d1c2", "Battery on h7"),

    # 8. Endgame Fundamentals
    ("8/8/8/4k3/8/8/4K3/4R3 w - - 0 1", "e2d3", "King Opposition Cutoff"),
    ("8/5k2/8/8/4KP2/8/8/8 w - - 0 1", "e4f5", "King Takes Opposition"),
    ("8/8/8/8/5kp1/8/6PK/8 w - - 0 1", "h2g1", "Hold Pawn Line"),
]


class GrandmasterTrainer:
    """
    Coordinates Grandmaster imprinting, autonomous sparring, and SWS sleep consolidation:
    1. Imprinting: Master motifs stamp attractor memories into Hippocampus CA3 and reinforce Basal Ganglia D1.
    2. Sparring: High-speed autonomous sparring against an internal evaluation bot.
    3. Sleep Consolidation: Nightly SWS replay and Tononi SHY downscaling.
    4. Persistence: Saves and loads master brain checkpoints (master_brain.npz).
    """

    def __init__(self, brain: BIB2NervousSystem, adapter: ChessAdapter):
        self.brain = brain
        self.adapter = adapter
        self.sensory = ChessBoardSensory()
        self.total_puzzles_trained = 0
        self.total_games_played = 0

    def imprint_grandmaster_motifs(self, limit: int = 30) -> int:
        """
        Imprints classic tactical puzzles and master games into Hippocampal CA3 attractor
        and reinforces Basal Ganglia D1 (Go) weights.
        """
        count = 0
        puzzles = GRANDMASTER_PUZZLES[:limit]

        for fen, best_uci, name in puzzles:
            board = chess.Board(fen)
            self.sensory.set_board(board)

            # 1. Ingest sensory state into Optic Nerve
            retina = self.sensory.encode_retina(board.turn)
            self.brain.peripheral.cranial.set_sensory("CN_II", retina)
            somatic = self.sensory.encode_somatic(board.turn)
            self.brain.peripheral.spinal.set_dermatome("C5", somatic)

            # Step clock
            self.brain.tick()

            # 2. Execute and reinforce best master move
            try:
                best_move = chess.Move.from_uci(best_uci)
                if best_move in board.legal_moves:
                    board.push(best_move)
                    retina_after = self.sensory.encode_retina(board.turn)
                    # Reinforce into Hippocampus CA3 and Basal Ganglia
                    self.adapter.reinforce_move(reward_rpe=0.8, board_snapshot=retina_after)
                    count += 1
            except Exception:
                continue

        self.total_puzzles_trained += count
        return count

    def spar_match(self, max_moves: int = 40) -> Dict[str, Any]:
        """
        Executes a rapid autonomous game between BIB-2 (White) and an internal heuristic opponent (Black).
        """
        board = chess.Board()
        moves_played = 0

        while moves_played < max_moves and not board.is_game_over():
            if board.turn == chess.WHITE:
                # BIB-2 plays
                move = self.adapter.select_move(board, chess.WHITE)
                if move is None or move not in board.legal_moves:
                    move = list(board.legal_moves)[0]
                board.push(move)
            else:
                # Sparring opponent heuristic
                opp_moves = list(board.legal_moves)
                # Prioritize captures or checks
                captures = [m for m in opp_moves if board.is_capture(m)]
                if captures:
                    move = captures[0]
                else:
                    move = opp_moves[0]
                board.push(move)

            moves_played += 1

        self.total_games_played += 1
        winner = "Draw"
        if board.is_checkmate():
            winner = "Black" if board.turn == chess.WHITE else "White (BIB-2)"

        # Reinforce outcome
        final_retina = self.sensory.encode_retina(chess.WHITE)
        if winner == "White (BIB-2)":
            self.adapter.reinforce_move(reward_rpe=1.0, board_snapshot=final_retina)
        elif winner == "Black":
            self.adapter.reinforce_move(reward_rpe=-1.0, board_snapshot=final_retina)

        return {
            "moves_played": moves_played,
            "winner": winner,
            "is_checkmate": board.is_checkmate(),
            "final_fen": board.fen()
        }

    def consolidate_sleep(self, n_sleep_ticks: int = 10) -> Dict[str, Any]:
        """
        Executes nocturnal Slow-Wave Sleep (SWS) consolidation:
        - Sharp-Wave Ripples (SWRs) replay top memories into Neocortex.
        - Tononi Synaptic Homeostasis (SHY) applies -5% downscaling to clear noise.
        - Glymphatics clear adenosine.
        """
        initial_adenosine = float(self.brain.chemistry.matrix.state.adenosine)
        self.brain.sleep.transition_to(SleepStage.NREM_SWS)

        replayed_count = min(5, len(self.brain.episodic_memory))
        for _ in range(n_sleep_ticks):
            self.brain.tick_sleep()

        self.brain.chemistry.matrix.clear_adenosine(amount=initial_adenosine)
        self.brain.chemistry.matrix.state.adenosine = float(
            max(0.0, self.brain.chemistry.matrix.state.adenosine - 0.5)
        )
        self.brain.chemistry.matrix.state.cortisol = 0.15
        self.brain.sleep.transition_to(SleepStage.WAKE)

        return {
            "replayed_count": replayed_count,
            "downscaled": True,
            "adenosine_cleared": initial_adenosine
        }

    def save_checkpoint(self, filepath: str) -> None:
        """Saves mastered brain weights to persistent numpy .npz format."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        ca3 = self.brain.limbic.hippocampus.ca3
        bg = self.brain.basal_ganglia
        chem = self.brain.chemistry.matrix.state

        patterns = np.array(ca3.stored_patterns, dtype=np.float32) if ca3.stored_patterns else np.zeros((0, 64), dtype=np.float32)

        np.savez(
            filepath,
            ca3_matrix=ca3.recurrent_matrix,
            ca3_patterns=patterns,
            d1_weights=bg.d1_weights,
            d2_weights=bg.d2_weights,
            dopamine=np.array([chem.dopamine], dtype=np.float32),
            cortisol=np.array([chem.cortisol], dtype=np.float32),
        )

    def load_checkpoint(self, filepath: str) -> bool:
        """Loads mastered brain weights from persistent .npz file."""
        if not os.path.exists(filepath):
            return False

        data = np.load(filepath)
        ca3 = self.brain.limbic.hippocampus.ca3
        bg = self.brain.basal_ganglia
        chem = self.brain.chemistry.matrix.state

        if "ca3_matrix" in data:
            ca3.recurrent_matrix = data["ca3_matrix"].copy()
        if "ca3_patterns" in data:
            ca3.stored_patterns = [p.copy() for p in data["ca3_patterns"]]
        if "d1_weights" in data:
            bg.d1_weights = data["d1_weights"].copy()
        if "d2_weights" in data:
            bg.d2_weights = data["d2_weights"].copy()
        if "dopamine" in data:
            chem.dopamine = float(data["dopamine"][0])
        if "cortisol" in data:
            chem.cortisol = float(data["cortisol"][0])

        return True
