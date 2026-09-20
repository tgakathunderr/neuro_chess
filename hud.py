"""Interactive Cybernetic Chess HUD with Real-Time 2D Brain Activation Heatmap and ECG."""

import math
import time
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
import pygame
import chess

# Color Palette
BG_DARK = (10, 14, 22)
PANEL_BG = (14, 20, 32)
PANEL_BORDER = (28, 44, 68)
GRID_LINE = (20, 30, 48)

TEXT_WHITE = (235, 242, 255)
TEXT_MUTED = (140, 160, 185)
TEXT_GOLD = (250, 204, 21)
TEXT_CYAN = (56, 189, 248)

# Board Colors
SQ_LIGHT = (220, 228, 240)
SQ_DARK = (60, 85, 120)
SQ_SELECTED = (250, 204, 21, 140)
SQ_LAST_MOVE = (56, 189, 248, 80)
SQ_CHECK = (239, 68, 68, 180)
DOT_MOVE = (34, 197, 94, 200)

# Piece Colors
WHITE_PIECE_BG = (245, 248, 255)
WHITE_PIECE_BORDER = (250, 204, 21)
WHITE_PIECE_TEXT = (15, 23, 42)

BLACK_PIECE_BG = (15, 23, 42)
BLACK_PIECE_BORDER = (56, 189, 248)
BLACK_PIECE_TEXT = (245, 248, 255)

# Biometrics
ECG_LINE = (34, 211, 153)
ECG_BG = (8, 16, 24)

# Brain Anatomical Coordinates (Normalized within Brain Panel 0.0 to 1.0)
BRAIN_NODES = {
    "v1": {"pos": (0.80, 0.42), "name": "V1 (Visual)", "tracts": ["thalamus", "cerebellum"]},
    "dlpfc": {"pos": (0.24, 0.28), "name": "DLPFC (Logic)", "tracts": ["m1", "basal_ganglia", "thalamus"]},
    "m1": {"pos": (0.46, 0.20), "name": "M1 (Motor)", "tracts": ["basal_ganglia", "brainstem"]},
    "thalamus": {"pos": (0.50, 0.42), "name": "Thalamus", "tracts": ["v1", "basal_ganglia", "hippocampus"]},
    "basal_ganglia": {"pos": (0.38, 0.40), "name": "Basal Ganglia", "tracts": ["m1", "thalamus", "amygdala"]},
    "hippocampus": {"pos": (0.44, 0.58), "name": "Hippocampus", "tracts": ["thalamus", "amygdala"]},
    "amygdala": {"pos": (0.30, 0.58), "name": "Amygdala", "tracts": ["brainstem", "basal_ganglia"]},
    "cerebellum": {"pos": (0.74, 0.70), "name": "Cerebellum", "tracts": ["brainstem", "thalamus"]},
    "brainstem": {"pos": (0.50, 0.82), "name": "Brainstem", "tracts": []},
}


class ChessHUD:
    """
    1360x820 Cybernetic Grandmaster HUD:
    - Interactive 8x8 chessboard with click-to-move piece controls and legal move circles.
    - Real-time 2D Macro-Brain Activation Heatmap (V1, DLPFC, M1, Hippocampus, Amygdala, etc.).
    - Animated ECG oscilloscope (accelerates to 130+ BPM under check or tactical sharp lines).
    - Neurochemical meters, move notation log, and captured piece trays.
    """

    def __init__(self, width: int = 1360, height: int = 820):
        pygame.init()
        pygame.display.set_caption("BIB-2 Neuro-Chess: Biomimetic Grandmaster")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_body = pygame.font.SysFont("Consolas", 13)
        self.font_small = pygame.font.SysFont("Consolas", 11)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_piece = pygame.font.SysFont("Arial", 22, bold=True)

        # Board Geometry
        self.board_x = 30
        self.board_y = 70
        self.sq_size = 85
        self.board_size = self.sq_size * 8

        # Interactive State
        self.selected_sq: Optional[chess.Square] = None
        self.valid_targets: Set[chess.Square] = set()

        # ECG State
        self.ecg_points: List[float] = [0.0] * 120
        self.ecg_phase = 0.0

    def _generate_ecg_sample(self, phase: float) -> float:
        """P-Q-R-S-T cardiac voltage waveform."""
        p = phase % (2.0 * math.pi)
        val = 0.0
        if 0.5 <= p < 1.0:
            val += 0.15 * math.sin((p - 0.5) * 2.0 * math.pi)
        elif 1.4 <= p < 1.6:
            val -= 0.15 * math.sin((p - 1.4) * 5.0 * math.pi)
        elif 1.6 <= p < 1.9:
            val += 1.0 * math.sin((p - 1.6) * (math.pi / 0.3))
        elif 1.9 <= p < 2.1:
            val -= 0.25 * math.sin((p - 1.9) * 5.0 * math.pi)
        elif 2.6 <= p < 3.4:
            val += 0.25 * math.sin((p - 2.6) * (math.pi / 0.8))
        return float(val)

    def handle_click(self, mouse_pos: Tuple[int, int], board: chess.Board, human_color: chess.Color) -> Optional[chess.Move]:
        """
        Processes mouse click on chessboard. Returns chosen legal chess.Move if completed.
        """
        mx, my = mouse_pos
        if not (self.board_x <= mx < self.board_x + self.board_size and
                self.board_y <= my < self.board_y + self.board_size):
            self.selected_sq = None
            self.valid_targets.clear()
            return None

        file_idx = (mx - self.board_x) // self.sq_size
        rank_idx = 7 - ((my - self.board_y) // self.sq_size)
        clicked_sq = chess.square(file_idx, rank_idx)

        # If already selected a piece and clicked on a valid target square -> execute move!
        if self.selected_sq is not None and clicked_sq in self.valid_targets:
            # Check for pawn promotion to Queen
            move = chess.Move(self.selected_sq, clicked_sq)
            piece = board.piece_at(self.selected_sq)
            if piece and piece.piece_type == chess.PAWN and (rank_idx == 7 or rank_idx == 0):
                move = chess.Move(self.selected_sq, clicked_sq, promotion=chess.QUEEN)

            self.selected_sq = None
            self.valid_targets.clear()
            return move

        # Otherwise, select piece if it belongs to human
        piece = board.piece_at(clicked_sq)
        if piece and piece.color == human_color:
            self.selected_sq = clicked_sq
            self.valid_targets = {
                m.to_square for m in board.legal_moves if m.from_square == clicked_sq
            }
        else:
            self.selected_sq = None
            self.valid_targets.clear()

        return None

    def render(
        self,
        board: chess.Board,
        human_color: chess.Color,
        brain_activations: Dict[str, float],
        telemetry: Dict[str, Any],
        last_move: Optional[chess.Move] = None,
        status_msg: str = "",
        self_play: bool = False,
        is_paused: bool = False,
        move_delay: float = 1.8,
        time_until_next: float = 0.0,
        recent_moves: str = "",
        copied_feedback: str = "",
    ) -> bool:
        """
        Renders complete frame. Returns False if user closed window.
        """
        self.screen.fill(BG_DARK)

        # 1. Header Banner
        self._render_header(board, human_color, telemetry, self_play, is_paused, move_delay, time_until_next)

        # 2. Interactive Chessboard (Left 50%)
        self._render_board(board, last_move)

        # 3. 2D Macro-Brain Activation Heatmap (Upper Right)
        brain_rect = pygame.Rect(740, 70, 590, 360)
        self._render_brain_heatmap(brain_rect, brain_activations)

        # 4. Biometrics & ECG Panel (Lower Right)
        bio_rect = pygame.Rect(740, 445, 590, 305)
        self._render_biometrics(bio_rect, telemetry, status_msg, self_play, recent_moves, copied_feedback)

        # 5. Footer Keybinds Help
        self._render_footer(self_play, is_paused)

        pygame.display.flip()
        self.clock.tick(60)
        return True

    def _render_header(
        self,
        board: chess.Board,
        human_color: chess.Color,
        telemetry: Dict[str, Any],
        self_play: bool = False,
        is_paused: bool = False,
        move_delay: float = 1.8,
        time_until_next: float = 0.0,
    ) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, (25, 10, self.width - 50, 45), border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, (25, 10, self.width - 50, 45), 1, border_radius=6)

        title_str = "BIB-2 NEURO-CHESS  |  AUTONOMOUS SELF-PLAY" if self_play else "BIB-2 NEURO-CHESS  |  BIOMIMETIC GRANDMASTER"
        title = self.font_title.render(title_str, True, TEXT_CYAN)
        self.screen.blit(title, (40, 22))

        # Turn indicator
        if self_play:
            side_str = "WHITE" if board.turn == chess.WHITE else "BLACK"
            if is_paused:
                turn_str = f"{side_str} [PAUSED - PRESS SPACE OR ->]"
                turn_col = (251, 146, 60)
            elif move_delay <= 0.15:
                turn_str = f"{side_str} (TURBO {move_delay:.2f}s | TAB TO SLOW)"
                turn_col = (56, 189, 248) if board.turn == chess.WHITE else (250, 204, 21)
            else:
                turn_str = f"{side_str} (NEXT IN {time_until_next:.1f}s | {move_delay:.1f}s PACE)"
                turn_col = TEXT_GOLD if board.turn == chess.WHITE else TEXT_CYAN
        else:
            turn_str = "WHITE (YOU)" if board.turn == human_color else "BLACK (BIB-2 THINKING...)"
            turn_col = TEXT_GOLD if board.turn == human_color else TEXT_CYAN

        if board.is_check():
            turn_str += " [CHECK!]"
            turn_col = (239, 68, 68)

        t_surf = self.font_body.render(f"TURN: {turn_str}", True, turn_col)
        self.screen.blit(t_surf, (510, 24))

        # Move count
        m_count = f"MOVE: #{board.fullmove_number}"
        self.screen.blit(self.font_body.render(m_count, True, TEXT_WHITE), (990, 24))

        # Heart Rate
        hr = telemetry.get("heart_rate", 70)
        hr_col = (239, 68, 68) if hr > 105 else TEXT_GOLD
        hr_surf = self.font_body.render(f"HR: {hr} BPM", True, hr_col)
        self.screen.blit(hr_surf, (1180, 24))

    def _render_board(self, board: chess.Board, last_move: Optional[chess.Move]) -> None:
        # Outer Border
        pygame.draw.rect(self.screen, PANEL_BG, (self.board_x - 5, self.board_y - 5, self.board_size + 10, self.board_size + 10), border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, (self.board_x - 5, self.board_y - 5, self.board_size + 10, self.board_size + 10), 2, border_radius=6)

        # Check square
        check_sq = None
        if board.is_check():
            check_sq = board.king(board.turn)

        # Draw Squares & Coordinates
        for rank in range(8):
            for file in range(8):
                sq = chess.square(file, rank)
                x = self.board_x + file * self.sq_size
                y = self.board_y + (7 - rank) * self.sq_size

                is_light = (file + rank) % 2 != 0
                col = SQ_LIGHT if is_light else SQ_DARK
                pygame.draw.rect(self.screen, col, (x, y, self.sq_size, self.sq_size))

                # Highlight last move
                if last_move and sq in (last_move.from_square, last_move.to_square):
                    overlay = pygame.Surface((self.sq_size, self.sq_size), pygame.SRCALPHA)
                    overlay.fill(SQ_LAST_MOVE)
                    self.screen.blit(overlay, (x, y))

                # Highlight selected piece
                if sq == self.selected_sq:
                    overlay = pygame.Surface((self.sq_size, self.sq_size), pygame.SRCALPHA)
                    overlay.fill(SQ_SELECTED)
                    self.screen.blit(overlay, (x, y))

                # Highlight King in check
                if sq == check_sq:
                    overlay = pygame.Surface((self.sq_size, self.sq_size), pygame.SRCALPHA)
                    overlay.fill(SQ_CHECK)
                    self.screen.blit(overlay, (x, y))

                # Draw Rank/File labels on edges
                if file == 0:
                    r_lbl = self.font_small.render(str(rank + 1), True, TEXT_MUTED)
                    self.screen.blit(r_lbl, (x + 3, y + 3))
                if rank == 0:
                    f_lbl = self.font_small.render(chess.FILE_NAMES[file], True, TEXT_MUTED)
                    self.screen.blit(f_lbl, (x + self.sq_size - 12, y + self.sq_size - 14))

                # Draw Piece
                piece = board.piece_at(sq)
                if piece:
                    self._draw_piece(piece, x + self.sq_size // 2, y + self.sq_size // 2)

                # Draw Legal Move Destination Circle
                if sq in self.valid_targets:
                    overlay = pygame.Surface((self.sq_size, self.sq_size), pygame.SRCALPHA)
                    pygame.draw.circle(overlay, DOT_MOVE, (self.sq_size // 2, self.sq_size // 2), 12)
                    self.screen.blit(overlay, (x, y))

    def _draw_piece(self, piece: chess.Piece, cx: int, cy: int) -> None:
        """Renders stylized circular piece token with high contrast."""
        radius = int(self.sq_size * 0.38)
        is_white = (piece.color == chess.WHITE)

        bg_col = WHITE_PIECE_BG if is_white else BLACK_PIECE_BG
        border_col = WHITE_PIECE_BORDER if is_white else BLACK_PIECE_BORDER
        text_col = WHITE_PIECE_TEXT if is_white else BLACK_PIECE_TEXT

        # Shadow
        pygame.draw.circle(self.screen, (10, 15, 25), (cx + 2, cy + 2), radius)
        # Token body
        pygame.draw.circle(self.screen, bg_col, (cx, cy), radius)
        pygame.draw.circle(self.screen, border_col, (cx, cy), radius, 2)

        # Piece glyph representation
        symbols = {
            chess.PAWN: "P",
            chess.KNIGHT: "N",
            chess.BISHOP: "B",
            chess.ROOK: "R",
            chess.QUEEN: "Q",
            chess.KING: "K",
        }
        sym = symbols.get(piece.piece_type, "?")
        glyph = self.font_piece.render(sym, True, text_col)
        self.screen.blit(glyph, (cx - glyph.get_width() // 2, cy - glyph.get_height() // 2))

    def _render_brain_heatmap(self, rect: pygame.Rect, acts: Dict[str, float]) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, 1, border_radius=6)

        title = self.font_title.render("2D MACRO-BRAIN ACTIVATION HEATMAP", True, TEXT_WHITE)
        self.screen.blit(title, (rect.left + 15, rect.top + 10))

        sub = self.font_small.render("Real-Time Cortical & Subcortical Regional Energy Levels", True, TEXT_MUTED)
        self.screen.blit(sub, (rect.left + 15, rect.top + 30))

        # Helper to compute pixel coords of brain node
        def node_xy(norm_pos: Tuple[float, float]) -> Tuple[int, int]:
            nx, ny = norm_pos
            px = int(rect.left + 20 + nx * (rect.width - 40))
            py = int(rect.top + 50 + ny * (rect.height - 70))
            return px, py

        # 1. Draw connecting neural axons
        for key, info in BRAIN_NODES.items():
            p1 = node_xy(info["pos"])
            for target in info["tracts"]:
                if target in BRAIN_NODES:
                    p2 = node_xy(BRAIN_NODES[target]["pos"])
                    pygame.draw.line(self.screen, (28, 45, 70), p1, p2, 2)

        # 2. Draw glowing regional nodes
        for key, info in BRAIN_NODES.items():
            cx, cy = node_xy(info["pos"])
            val = acts.get(key, 0.2)

            # Color interpolation: Cyan (0.1) -> Gold (0.5) -> Crimson (1.0)
            if val < 0.4:
                col = (40, 180, 240)
            elif val < 0.75:
                col = (250, 204, 21)
            else:
                col = (245, 70, 70)

            # Glowing halo
            halo_r = int(12 + val * 16)
            glow_surf = pygame.Surface((halo_r * 2, halo_r * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*col, int(60 + val * 120)), (halo_r, halo_r), halo_r)
            self.screen.blit(glow_surf, (cx - halo_r, cy - halo_r))

            # Core Node
            pygame.draw.circle(self.screen, col, (cx, cy), 8)
            pygame.draw.circle(self.screen, (240, 245, 255), (cx, cy), 4)

            # Node Label
            lbl = self.font_small.render(f"{info['name']} {int(val * 100)}%", True, TEXT_WHITE)
            self.screen.blit(lbl, (cx - lbl.get_width() // 2, cy + 12))

    def _render_biometrics(
        self,
        rect: pygame.Rect,
        telemetry: Dict[str, Any],
        status_msg: str,
        self_play: bool = False,
        recent_moves: str = "",
        copied_feedback: str = "",
    ) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, 1, border_radius=6)

        lbl = self.font_title.render("AUTONOMIC BIOMETRICS & TACTICAL INTUITION", True, TEXT_WHITE)
        self.screen.blit(lbl, (rect.left + 15, rect.top + 10))

        # ECG Oscilloscope
        hr = telemetry.get("heart_rate", 70)
        ecg_rect = pygame.Rect(rect.left + 15, rect.top + 38, 250, 75)
        pygame.draw.rect(self.screen, ECG_BG, ecg_rect, border_radius=4)
        pygame.draw.rect(self.screen, PANEL_BORDER, ecg_rect, 1, border_radius=4)

        # Advance ECG
        self.ecg_phase += (hr / 60.0) * 0.15
        sample = self._generate_ecg_sample(self.ecg_phase)
        self.ecg_points.append(sample)
        if len(self.ecg_points) > ecg_rect.width:
            self.ecg_points.pop(0)

        mid_y = ecg_rect.centery
        pts = [(ecg_rect.left + i, int(mid_y - v * 28.0)) for i, v in enumerate(self.ecg_points)]
        if len(pts) > 1:
            pygame.draw.lines(self.screen, ECG_LINE, False, pts, 2)
            pygame.draw.circle(self.screen, (220, 255, 240), pts[-1], 3)

        hr_str = f"HEART: {hr} BPM"
        self.screen.blit(self.font_small.render(hr_str, True, TEXT_CYAN), (rect.left + 15, rect.top + 118))

        # Neurochemistry Meters
        y_m = rect.top + 40
        x_m = rect.left + 290
        meters = [
            ("DOPAMINE (ATTACK/REWARD)", telemetry.get("dopamine", 0.5), (250, 204, 21)),
            ("CORTISOL (CHECK/STRESS)", telemetry.get("cortisol", 0.2), (244, 63, 94)),
            ("SEROTONIN (PATIENCE)", telemetry.get("serotonin", 0.5), (168, 85, 247)),
            ("NOREPINEPHRINE (ALERT)", telemetry.get("norepinephrine", 0.4), (245, 158, 11)),
        ]
        for name, val, col in meters:
            self.screen.blit(self.font_small.render(f"{name}: {val:.2f}", True, TEXT_WHITE), (x_m, y_m))
            bar_bg = pygame.Rect(x_m + 160, y_m + 2, 110, 8)
            pygame.draw.rect(self.screen, (20, 30, 48), bar_bg, border_radius=2)
            fill_w = int(max(0.0, min(1.0, val)) * 110)
            if fill_w > 0:
                pygame.draw.rect(self.screen, col, (x_m + 160, y_m + 2, fill_w, 8), border_radius=2)
            y_m += 22

        # Tactical Intent & Opponent Prediction
        y_info = rect.top + 138
        pred_reply = telemetry.get("predicted_reply", "Analyzing...")
        hip_match = telemetry.get("hippocampal_match", 0.0) * 100.0

        anticipate_lbl = "CEREBELLAR LOOKAHEAD" if self_play else "CEREBELLUM ANTICIPATES"
        p1 = self.font_body.render(f"{anticipate_lbl}: {pred_reply}", True, TEXT_GOLD)
        p2 = self.font_body.render(f"HIPPOCAMPUS MOTIF RECALL: {hip_match:.1f}%", True, TEXT_CYAN)
        self.screen.blit(p1, (rect.left + 15, y_info))
        self.screen.blit(p2, (rect.left + 15, y_info + 22))

        # Recent moves line
        if recent_moves:
            m_surf = self.font_small.render(f"MOVES: {recent_moves}", True, TEXT_WHITE)
            self.screen.blit(m_surf, (rect.left + 15, y_info + 44))

        # Status / Action Log
        if status_msg:
            stat_surf = self.font_body.render(f">> {status_msg}", True, (52, 211, 153))
            self.screen.blit(stat_surf, (rect.left + 15, y_info + 64))

        # Copy feedback or clipboard hotkey tip
        if copied_feedback:
            tip_surf = self.font_small.render(f"CLIPBOARD: {copied_feedback}", True, (34, 197, 94))
        else:
            tip_surf = self.font_small.render("PGN LOGS: Press [C] to Copy PGN & Game Log to Clipboard (Auto-saved)", True, (125, 211, 252))
        self.screen.blit(tip_surf, (rect.left + 15, y_info + 86))

    def _render_footer(self, self_play: bool = False, is_paused: bool = False) -> None:
        y_bot = self.height - 35
        if self_play:
            pause_label = "Resume" if is_paused else "Pause"
            guide = f"HOTKEYS: [SPACE] {pause_label}  |  [TAB] Fast/Slow Toggle  |  [C] Copy Game Log/PGN  |  [1-5] Speed (2.5s-0.08s)  |  [->] Step  |  [R] Reset  |  [Q] Exit"
        else:
            guide = "HOTKEYS: [C] Copy Game Log/PGN  |  [R] Reset  |  [F] Flip Color  |  [T] Sparring Epoch  |  [Q] Exit"
        surf = self.font_body.render(guide, True, TEXT_MUTED)
        self.screen.blit(surf, (45, y_bot))
