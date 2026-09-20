"""Game logger and PGN exporter for BIB-2 Neuro-Chess."""

import os
import sys
import time
import datetime
import subprocess
from typing import List, Dict, Any, Optional, Tuple
import chess
import chess.pgn


def copy_text_to_clipboard(text: str) -> bool:
    """Copies text to the system clipboard across platforms with fallbacks."""
    # 1. Windows native clip.exe
    if sys.platform == "win32":
        try:
            p = subprocess.Popen(["clip.exe"], stdin=subprocess.PIPE, shell=True)
            p.communicate(text.encode("utf-8"))
            return True
        except Exception:
            pass

    # 2. Pygame scrap if initialized
    try:
        import pygame.scrap
        if pygame.scrap.get_init():
            pygame.scrap.put_text(text)
            return True
    except Exception:
        pass

    # 3. Tkinter fallback
    try:
        import tkinter as tk
        r = tk.Tk()
        r.withdraw()
        r.clipboard_clear()
        r.clipboard_append(text)
        r.update()
        r.destroy()
        return True
    except Exception:
        pass

    return False


class GameLogger:
    """
    Logs chess game moves, telemetry, and board states.
    Supports PGN formatting, detailed text logs, file persistence, and clipboard copying.
    """

    def __init__(self, white_name: str = "BIB-2 Grandmaster (White)", black_name: str = "BIB-2 Grandmaster (Black)"):
        self.white_name = white_name
        self.black_name = black_name
        self.start_time = datetime.datetime.now()
        self.history: List[Dict[str, Any]] = []
        self.result: str = "*"

    def record_move(
        self,
        board_before: chess.Board,
        move: chess.Move,
        player_name: str = "",
        activations: Optional[Dict[str, float]] = None,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Records a move before it is pushed to the board. Returns the SAN representation."""
        san = board_before.san(move)
        uci = move.uci()
        move_num = board_before.fullmove_number
        color_str = "White" if board_before.turn == chess.WHITE else "Black"
        piece = board_before.piece_at(move.from_square)
        piece_symbol = piece.symbol().upper() if piece else ""

        entry = {
            "move_number": move_num,
            "turn": board_before.turn,
            "color": color_str,
            "player": player_name or (self.white_name if board_before.turn == chess.WHITE else self.black_name),
            "san": san,
            "uci": uci,
            "piece": piece_symbol,
            "is_capture": board_before.is_capture(move),
            "is_check": board_before.gives_check(move),
            "timestamp": time.time(),
            "activations": dict(activations) if activations else {},
            "telemetry": dict(telemetry) if telemetry else {},
        }
        self.history.append(entry)
        return san

    def set_result(self, result: str) -> None:
        """Sets the game outcome (e.g., '1-0', '0-1', '1/2-1/2', '*')."""
        self.result = result

    def get_recent_moves_san(self, n: int = 6) -> str:
        """Returns the last n half-moves in clean standard notation (e.g., '1. e4 e5  2. Nf3 Nc6')."""
        if not self.history:
            return "No moves yet"

        recent = self.history[-n:]
        parts = []
        for h in recent:
            if h["turn"] == chess.WHITE:
                parts.append(f"{h['move_number']}. {h['san']}")
            else:
                parts.append(f"{h['san']}")
        return "  ".join(parts)

    def to_pgn(self, event: str = "BIB-2 Biomimetic Neuro-Chess Grandmaster") -> str:
        """Builds standard PGN compliant string representation of the game."""
        # Reconstruct game with chess.pgn
        game = chess.pgn.Game()
        game.headers["Event"] = event
        game.headers["Site"] = "Universal Neural Bus (BIB-2)"
        game.headers["Date"] = self.start_time.strftime("%Y.%m.%d")
        game.headers["Round"] = "1"
        game.headers["White"] = self.white_name
        game.headers["Black"] = self.black_name
        game.headers["Result"] = self.result

        node = game
        temp_board = chess.Board()
        for h in self.history:
            m = chess.Move.from_uci(h["uci"])
            if m in temp_board.legal_moves:
                node = node.add_variation(m)
                temp_board.push(m)

        exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=False)
        return game.accept(exporter)

    def to_text_log(self) -> str:
        """Generates a complete, beautifully structured human-readable text log."""
        lines = []
        lines.append("=" * 70)
        lines.append("BIB-2 NEURO-CHESS GRANDMASTER - GAME LOG")
        lines.append(f"Date: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"White: {self.white_name}")
        lines.append(f"Black: {self.black_name}")
        lines.append(f"Result: {self.result}")
        lines.append(f"Total Half-Moves: {len(self.history)}")
        lines.append("=" * 70)
        lines.append("")
        lines.append("[PORTABLE GAME NOTATION (PGN)]")
        lines.append(self.to_pgn())
        lines.append("")
        lines.append("[MOVE-BY-MOVE BREAKDOWN]")
        lines.append(f"{'Move':<8} {'Player':<22} {'SAN':<8} {'UCI':<8} {'Heart Rate':<12} {'Cerebellar Prediction'}")
        lines.append("-" * 70)

        for h in self.history:
            num_str = f"{h['move_number']}." if h['turn'] == chess.WHITE else f"{h['move_number']}..."
            hr = h['telemetry'].get('heart_rate', 70)
            pred = h['telemetry'].get('predicted_reply', '-')
            lines.append(
                f"{num_str:<8} {h['player'][:20]:<22} {h['san']:<8} {h['uci']:<8} {str(hr) + ' BPM':<12} {pred}"
            )

        lines.append("=" * 70)
        return "\n".join(lines)

    def copy_to_clipboard(self) -> Tuple[bool, str]:
        """Copies the PGN and game log to the OS clipboard. Returns (success, feedback_message)."""
        content = self.to_text_log()
        ok = copy_text_to_clipboard(content)
        if ok:
            return True, f"Game Log ({len(self.history)} moves) copied to clipboard! (Press Ctrl+V to paste)"
        return False, "Failed to copy to clipboard. Game saved to neuro_chess/game_logs/"

    def save_to_file(self, base_dir: Optional[str] = None) -> Tuple[str, str]:
        """Saves game logs to .pgn and .txt files. Returns (pgn_path, txt_path)."""
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(__file__), "game_logs")
        os.makedirs(base_dir, exist_ok=True)

        pgn_file = os.path.join(base_dir, "latest_game.pgn")
        txt_file = os.path.join(base_dir, "latest_game.txt")

        pgn_content = self.to_pgn()
        txt_content = self.to_text_log()

        with open(pgn_file, "w", encoding="utf-8") as f:
            f.write(pgn_content)
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(txt_content)

        return pgn_file, txt_file
