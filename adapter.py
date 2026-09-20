"""Neurobiological Chess Adapter: Bridges python-chess to BIB-2's 14 anatomical subsystems."""

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
from bib2.adapters.base import BaseNeuralAdapter
from .board import ChessBoardSensory, PIECE_VALUES


class ChessAdapter(BaseNeuralAdapter):
    """
    Biological chess adapter connecting chessboard perception to the human nervous system:
    1. Retinotopic Optic Nerve CN II: 64-dimensional board geometry.
    2. Somatosensory Spinal C5: Material balance (DCML) and King distress/pain (STT).
    3. Cerebellar Forward Model: Lookahead simulation of candidate moves & counter-replies.
    4. Hippocampal CA3 Attractor: Grandmaster tactical chunk recall and pattern completion.
    5. Basal Ganglia Gating: Candidate move disinhibition across Direct (D1) vs Indirect (D2) pathways.
    6. Macro-Brain Activation Telemetry: Real-time energy levels across all 9 major brain regions.
    """

    def __init__(self, brain: BIB2NervousSystem):
        super().__init__(brain)
        self.sensory = ChessBoardSensory()
        self.last_action_idx = 0
        self.last_hippocampal_match = 0.0
        self.cerebellar_error = 0.0
        self.last_predicted_reply = ""
        self.last_move = None

    def select_move(
        self,
        board: chess.Board,
        color: chess.Color = chess.WHITE,
        use_hippocampus: bool = True,
    ) -> Optional[chess.Move]:
        """
        Evaluate candidate legal moves through BIB-2's biological stack and disinhibit winning move.
        When use_hippocampus=True, Hippocampal CA3 associative memory attractor drives tactical
        chunk recall (+15.0 weight), enabling master moves such as sacrifices.
        When use_hippocampus=False, operates in pure heuristic ablation mode.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        self.sensory.set_board(board)

        # 1. Ingest current sensory state into peripheral bus
        cn2_visual = self.sensory.encode_retina(color)
        self.brain.peripheral.cranial.set_sensory("CN_II", cn2_visual)

        somatic = self.sensory.encode_somatic(color)
        self.brain.peripheral.spinal.set_dermatome("C5", somatic)

        # If King is in check, trigger sympathetic threat surge and amygdala fear drive
        if board.is_check():
            self.brain.peripheral.autonomic.trigger_sympathetic_surge(intensity=0.8)
            self.brain.chemistry.matrix.state.cortisol = float(
                np.clip(self.brain.chemistry.matrix.state.cortisol + 0.15, 0.0, 1.0)
            )

        # 2. Step 19-Stage Biological Clock Cycle
        self.brain.tick()

        # 3. Evaluate Candidate Moves
        candidate_scores: List[Tuple[chess.Move, float, float]] = []

        for move in legal_moves:
            # Tactical heuristic baseline (normalized)
            h_score = self._evaluate_move_heuristics(board, move, color)

            # Lookahead via Cerebellar simulation & Hippocampal pattern recall
            board.push(move)
            retina_fwd = self.sensory.encode_retina(color)

            # Cerebellar Veto: Check if move allows opponent an immediate checkmate
            has_mate_reply = False
            for opp_m in board.legal_moves:
                if board.gives_check(opp_m):
                    board.push(opp_m)
                    if board.is_checkmate():
                        has_mate_reply = True
                    board.pop()
                    if has_mate_reply:
                        break

            # Hippocampal CA3 associative pattern match against stored attractor chunks
            hip_match = 0.0
            stored = self.brain.limbic.hippocampus.ca3.stored_patterns
            if use_hippocampus and len(stored) > 0:
                rf_norm = retina_fwd / (np.linalg.norm(retina_fwd) + 1e-6)
                best_dot = max(float(np.dot(rf_norm, p / (np.linalg.norm(p) + 1e-6))) for p in stored)
                hip_match = float(np.clip(best_dot, 0.0, 1.0))

            board.pop()

            # CA3 pattern completion heavily weights master tactical attractor configurations (+15.0)
            if use_hippocampus:
                total_score = h_score + (hip_match * 15.0)
            else:
                total_score = h_score

            if has_mate_reply:
                total_score -= 50.0  # Cerebellar Veto for blunders allowing immediate checkmate

            candidate_scores.append((move, total_score, hip_match))

        # Sort candidate moves by preliminary evaluation
        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        # 4. Basal Ganglia Action Selection across top candidate proposals
        top_candidates = candidate_scores[:8]
        proposals = np.array([c[1] for c in top_candidates], dtype=np.float32)

        winner_idx, _ = self.brain.basal_ganglia.select_action(
            proposals, dopamine_level=self.brain.chemistry.matrix.state.dopamine
        )
        selected_cand = top_candidates[winner_idx % len(top_candidates)]
        winning_move = selected_cand[0]
        self.last_hippocampal_match = selected_cand[2]
        self.last_move = winning_move
        self.last_action_idx = int(winner_idx)

        # Cerebellar anticipation of opponent reply
        board.push(winning_move)
        opponent_legal = list(board.legal_moves)
        if opponent_legal:
            # Predict top response
            self.last_predicted_reply = str(opponent_legal[0])
        else:
            self.last_predicted_reply = "None (Mate/Draw)"
        board.pop()

        return winning_move

    def _evaluate_move_heuristics(self, board: chess.Board, move: chess.Move, color: chess.Color) -> float:
        """Computes positional and tactical heuristic score for candidate move."""
        score = 0.0
        opp_color = not color
        piece = board.piece_at(move.from_square)
        p_val = PIECE_VALUES.get(piece.piece_type, 1.0) if piece else 1.0

        # 1. Captures (Most Valuable Victim - Least Valuable Attacker)
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = piece
            v_val = PIECE_VALUES[victim.piece_type] if victim else 1.0
            a_val = PIECE_VALUES[attacker.piece_type] if attacker else 1.0
            score += 2.5 + (v_val - a_val * 0.1)

            # If capturing into defended square with more valuable attacker, apply trade penalty
            if board.is_attacked_by(opp_color, move.to_square) and a_val > v_val:
                score -= (a_val - v_val) * 1.5

        # 2. Checks & Checkmates
        if board.gives_check(move):
            score += 1.5

        # 3. Promotions
        if move.promotion:
            score += 8.0

        # 4. Castling (King Safety & Rook Activation)
        if board.is_castling(move):
            score += 1.5

        # 5. Center Control (Pawn/Knight development to central squares)
        center_sqs = {chess.E4, chess.D4, chess.E5, chess.D5, chess.C4, chess.F4, chess.C5, chess.F5}
        if move.to_square in center_sqs:
            score += 0.4

        # 6. Piece Development (moving pieces off rank 1 for White / rank 8 for Black in opening)
        if piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            if (color == chess.WHITE and chess.square_rank(move.from_square) == 0) or \
               (color == chess.BLACK and chess.square_rank(move.from_square) == 7):
                score += 0.5

        # 7. Amygdala Threat Perception: Hanging piece avoidance
        if not board.is_capture(move):
            is_attacked_dest = board.is_attacked_by(opp_color, move.to_square)
            if is_attacked_dest:
                is_defended_dest = board.is_attacked_by(color, move.to_square)
                if not is_defended_dest:
                    # Penalize moving into undefended enemy fire (bounded so CA3 tactical attractors can overrule)
                    score -= min(p_val * 1.2, 9.0)
                elif p_val > 3.0:
                    # Major piece moving to attacked square even if defended
                    score -= min(p_val * 0.4, 4.0)

        # 8. Escape Threat: Reward moving piece away from current attack
        if board.is_attacked_by(opp_color, move.from_square):
            # Moving out of fire to an unattacked square
            if not board.is_attacked_by(opp_color, move.to_square):
                score += p_val * 0.8

        return float(score)

    def reinforce_move(self, reward_rpe: float, board_snapshot: np.ndarray) -> None:
        """
        Dopaminergic corticostriatal plasticity and hippocampal episodic pattern storage.
        """
        if reward_rpe > 0.0:
            # Reinforce D1 Go pathway for winning combination
            self.brain.basal_ganglia.reinforce_action(self.last_action_idx, reward_rpe=reward_rpe)
            self.brain.chemistry.matrix.state.dopamine = float(
                np.clip(self.brain.chemistry.matrix.state.dopamine + reward_rpe * 0.2, 0.05, 1.0)
            )
            # Store winning tactical snapshot in Hippocampal CA3 attractor
            snapshot = np.asarray(board_snapshot, dtype=np.float32).flatten()
            if snapshot.size >= 64:
                self.brain.limbic.hippocampus.ca3.store(snapshot[:64], lr=0.5)

        elif reward_rpe < 0.0:
            # Negative punishment: reinforce D2 NoGo & trigger sympathetic surge
            self.brain.basal_ganglia.reinforce_action(self.last_action_idx, reward_rpe=reward_rpe)
            self.brain.chemistry.matrix.state.dopamine = float(
                np.clip(self.brain.chemistry.matrix.state.dopamine - abs(reward_rpe) * 0.2, 0.05, 1.0)
            )
            self.brain.habenula.compute_anti_reward(expected_reward=0.5, received_reward=reward_rpe)
            self.brain.peripheral.autonomic.trigger_sympathetic_surge(intensity=float(min(1.0, abs(reward_rpe))))
            self.brain.chemistry.matrix.state.cortisol = float(
                np.clip(self.brain.chemistry.matrix.state.cortisol + abs(reward_rpe) * 0.15, 0.0, 1.0)
            )

    def get_brain_activations(self) -> Dict[str, float]:
        """
        Extracts real-time scalar energy levels (0.0 to 1.0) across all 9 major brain regions
        for the 2D anatomical brain activation heatmap.
        """
        v1_val = float(np.mean(np.abs(self.brain.neocortex.registry.get_area("V1_Visual").l5_output)))
        dlpfc_val = float(np.mean(np.abs(self.brain.neocortex.registry.get_area("DLPFC_WorkingMemory").l5_output)))
        m1_val = float(np.mean(np.abs(self.brain.neocortex.registry.get_area("M1_PrimaryMotor").l5_output)))
        hip_val = float(np.mean(np.abs(self.brain.limbic.hippocampus.subiculum)))
        amy_val = float(np.clip(np.mean(self.brain.amygdala.bla_weights) * 2.0, 0.0, 1.0))
        bg_val = float(np.clip(np.mean(self.brain.basal_ganglia.d1_weights[:3]) / 2.0, 0.0, 1.0))
        thal_val = float(np.mean(np.abs(self.brain.thalamus.read_nucleus("LGN"))))
        cereb_val = float(np.clip(self.cerebellar_error * 0.5, 0.0, 1.0))
        stem_val = float(np.clip(self.brain.peripheral.autonomic.sympathetic_tone, 0.0, 1.0))

        return {
            "v1": float(np.clip(v1_val * 2.0, 0.05, 1.0)),
            "dlpfc": float(np.clip(dlpfc_val * 2.5, 0.05, 1.0)),
            "m1": float(np.clip(m1_val * 2.0, 0.05, 1.0)),
            "hippocampus": float(np.clip(max(hip_val * 1.5, self.last_hippocampal_match), 0.05, 1.0)),
            "amygdala": float(np.clip(amy_val, 0.05, 1.0)),
            "basal_ganglia": float(np.clip(bg_val, 0.05, 1.0)),
            "thalamus": float(np.clip(thal_val * 2.0, 0.05, 1.0)),
            "cerebellum": float(np.clip(cereb_val + 0.1, 0.05, 1.0)),
            "brainstem": float(np.clip(stem_val, 0.05, 1.0)),
        }

    def get_telemetry(self) -> Dict[str, Any]:
        """Collects real-time autonomic, neurochemical, and cognitive metrics."""
        matrix = self.brain.chemistry.matrix.state
        autonomic = self.brain.peripheral.autonomic
        hr = int(68.0 + autonomic.sympathetic_tone * 70.0)

        return {
            "heart_rate": hr,
            "dopamine": float(matrix.dopamine),
            "cortisol": float(matrix.cortisol),
            "serotonin": float(matrix.serotonin),
            "norepinephrine": float(matrix.norepinephrine),
            "adenosine": float(matrix.adenosine),
            "sympathetic_tone": float(autonomic.sympathetic_tone),
            "vagal_tone": float(autonomic.parasympathetic_tone),
            "hippocampal_match": self.last_hippocampal_match,
            "predicted_reply": self.last_predicted_reply,
            "last_move": str(self.last_move) if self.last_move else "None",
            "tick_count": self.brain.tick_count,
        }
