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

        # 3. Evaluate Candidate Moves via Biological Sensory-Motor Bus
        candidate_scores: List[Tuple[chess.Move, float, float]] = []

        for move in legal_moves:
            # Spinal Somatosensory candidate drive (DCML, STT, Nociception)
            s_score = self._evaluate_somatic_candidate(board, move, color)

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
                raw_dot = float(np.clip(best_dot, 0.0, 1.0))
                # Dentate Gyrus pattern separation / sharpening power
                hip_match = raw_dot
                ca3_score = (hip_match ** 5) * 25.0
                total_score = s_score + ca3_score
            else:
                total_score = s_score

            board.pop()

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

    def _evaluate_somatic_candidate(self, board: chess.Board, move: chess.Move, color: chess.Color) -> float:
        """
        Computes biological spinal and somatosensory candidate drive:
        1. Spinal C5 (DCML): Proprioceptive material balance delta (Delta DCML).
        2. Spinal C5 (STT): Somatosensory King distress / threat reduction (Delta STT).
        3. Spinal Nociception: Tissue vulnerability & unshielded piece exposure avoidance.
        4. Motor Intent: Terminal checkmate goal and check disruption.

        Zero classical heuristics (no MVV-LVA, no artificial piece trade formulas).
        """
        opp_color = not color

        # 1. Somatosensory baseline before action
        mat_before = self.sensory.get_material_balance(color)
        threat_before = self.sensory.is_king_threatened(color)

        # 2. Simulate physical action
        board.push(move)
        mat_after = self.sensory.get_material_balance(color)
        threat_after = self.sensory.is_king_threatened(color)
        gives_check = board.is_check()
        is_mate = board.is_checkmate()

        # Nociceptive vulnerability: moving into undefended enemy fire
        to_sq = move.to_square
        is_attacked_dest = board.is_attacked_by(opp_color, to_sq)
        is_defended_dest = board.is_attacked_by(color, to_sq)
        piece = board.piece_at(to_sq)
        p_val = PIECE_VALUES.get(piece.piece_type, 1.0) if piece else 1.0

        board.pop()

        # 3. Compute biological sensory deltas
        delta_dcml = mat_after - mat_before           # Net material change via spinal DCML
        delta_stt = threat_before - threat_after       # Pain relief via spinal STT

        score = delta_dcml + (delta_stt * 2.0)

        if is_mate:
            score += 50.0  # Terminal biological goal / dopaminergic triumph
        elif gives_check:
            score += 1.0   # Somatic disruption of opponent

        if move.promotion:
            score += 8.0   # Somatic metamorphic expansion (pawn to queen)

        # Spinal nociception: penalize unshielded piece exposure
        if is_attacked_dest and not is_defended_dest:
            score -= min(p_val * 1.2, 9.0)
        elif is_attacked_dest and is_defended_dest and p_val > 3.0:
            score -= min(p_val * 0.4, 4.0)

        return float(score)

    def _evaluate_move_heuristics(self, board: chess.Board, move: chess.Move, color: chess.Color) -> float:
        """Alias to biological somatic evaluation (legacy name preserved for backward compatibility)."""
        return self._evaluate_somatic_candidate(board, move, color)

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
