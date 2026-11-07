"""LLM proposer: versioned prompts + robust parsing of the LLM's JSON answer.

Same prompts are used for the raw LLM (C1/C2/C4) and the fine-tuned LLM (C3/C5).
"""
from __future__ import annotations

import json
import re

import chess

PROMPT_VERSION = "p1"
SYSTEM = "You are a strong chess player. Think about threats, captures and checks, then answer with JSON only."


def board_ascii(board: chess.Board) -> str:
    rows = []
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            p = board.piece_at(chess.square(file, rank))
            cells.append(p.symbol() if p else ".")
        rows.append(f"{rank + 1} " + " ".join(cells))
    rows.append("  a b c d e f g h")
    return "\n".join(rows)


def legal_sans(board: chess.Board) -> list[str]:
    return sorted(board.san(m) for m in board.legal_moves)


def _position_block(board: chess.Board) -> str:
    side = "White" if board.turn == chess.WHITE else "Black"
    return (f"Position (FEN): {board.fen()}\n"
            f"{side} to move.\n"
            f"Board (uppercase = White, lowercase = Black):\n{board_ascii(board)}\n"
            f"Legal moves (SAN): {', '.join(legal_sans(board))}")


def single_move_messages(board: chess.Board) -> list[dict]:
    user = (_position_block(board) +
            '\n\nChoose the best move. Reply with JSON only: {"move": "<SAN>"}')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def candidates_messages(board: chess.Board, k: int = 5) -> list[dict]:
    user = (_position_block(board) +
            f"\n\nPropose the {k} most promising candidate moves, best first, each with a short reason "
            f"(at most 12 words). Reply with JSON only: "
            '{"candidates": [{"move": "<SAN>", "why": "<reason>"}, ...]}')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# ------------------------------------------------------------------ parsing
_THINK = re.compile(r"<think>.*?</think>", re.S)


def _json_obj(text: str):
    text = _THINK.sub("", text or "")
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b <= a:
        return None
    try:
        return json.loads(text[a:b + 1])
    except json.JSONDecodeError:
        return None


def to_move(board: chess.Board, s: str) -> chess.Move | None:
    """SAN (lenient) or UCI -> legal Move, else None."""
    if not isinstance(s, str):
        return None
    s = s.strip().strip(".").replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    s = re.sub(r"^\d+\.+\s*", "", s)  # "12. Nf3" / "12...Nf3"
    for cand in (s, s.rstrip("+#!?")):
        try:
            return board.parse_san(cand)
        except ValueError:
            pass
    try:
        mv = chess.Move.from_uci(s.lower())
        return mv if mv in board.legal_moves else None
    except ValueError:
        return None


def parse_single(board: chess.Board, text: str) -> chess.Move | None:
    obj = _json_obj(text)
    if isinstance(obj, dict) and "move" in obj:
        return to_move(board, obj["move"])
    m = re.search(r'"move"\s*:\s*"([^"]+)"', text or "")
    return to_move(board, m.group(1)) if m else None


def pad_order(board: chess.Board) -> list[chess.Move]:
    """Deterministic, engine-free fallback order: checks, then captures, then the rest; SAN alphabetical."""
    def key(m):
        return (not board.gives_check(m), not board.is_capture(m), board.san(m))
    return sorted(board.legal_moves, key=key)


def parse_candidates(board: chess.Board, text: str, k: int = 5) -> tuple[list[dict], int, int]:
    """Returns (candidates, n_llm_valid, n_pad). candidates: [{uci, san, why, padded}] of length min(k, n_legal)."""
    obj = _json_obj(text)
    raw = obj.get("candidates", []) if isinstance(obj, dict) else []
    if not raw:  # fallback: any "move": "..." in the text
        raw = [{"move": m} for m in re.findall(r'"move"\s*:\s*"([^"]+)"', text or "")]
    out, seen = [], set()
    for c in raw if isinstance(raw, list) else []:
        mv = to_move(board, c.get("move") if isinstance(c, dict) else c)
        if mv is None or mv in seen:
            continue
        seen.add(mv)
        why = c.get("why", "") if isinstance(c, dict) else ""
        out.append(dict(uci=mv.uci(), san=board.san(mv), why=str(why)[:200], padded=False))
        if len(out) == k:
            break
    n_valid = len(out)
    for mv in pad_order(board):
        if len(out) >= min(k, board.legal_moves.count()):
            break
        if mv not in seen:
            seen.add(mv)
            out.append(dict(uci=mv.uci(), san=board.san(mv), why="", padded=True))
    return out, n_valid, len(out) - n_valid
