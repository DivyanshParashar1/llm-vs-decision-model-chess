"""SFT targets for LLM* (the expensive arm). Same prompts as the raw LLM (proposer.py).

Each train position yields ONE example, randomly either the single-move format or the
k-candidate format (Stockfish MultiPV order), so one LLM* serves both C3 (single) and C5 (proposer).
"""
from __future__ import annotations

import json
import random

import chess

from .proposer import candidates_messages, single_move_messages


def auto_why(board: chess.Board, mv: chess.Move) -> str:
    """Short, deterministic, engine-free description of a move (no eval leakage)."""
    bits = []
    piece = board.piece_at(mv.from_square)
    name = chess.piece_name(piece.piece_type) if piece else "piece"
    if board.is_castling(mv):
        bits.append("castles for king safety")
    elif board.is_capture(mv):
        cap = board.piece_at(mv.to_square)
        bits.append(f"{name} captures {chess.piece_name(cap.piece_type) if cap else 'pawn'}")
    else:
        bits.append(f"{name} to {chess.square_name(mv.to_square)}")
    if mv.promotion:
        bits.append("promotes")
    b2 = board.copy(stack=False)
    b2.push(mv)
    if b2.is_checkmate():
        bits.append("checkmate")
    elif b2.is_check():
        bits.append("gives check")
    return ", ".join(bits)


def make_example(fen: str, sf_multipv: str, k: int, rng: random.Random, p_single: float = 0.5) -> tuple[list[dict], str, str]:
    """Returns (messages, assistant_json, format)."""
    board = chess.Board(fen)
    lines = json.loads(sf_multipv)
    if rng.random() < p_single:
        return single_move_messages(board), json.dumps({"move": lines[0]["san"]}), "single"
    cands = []
    for d in lines[:k]:
        mv = chess.Move.from_uci(d["uci"])
        cands.append({"move": d["san"], "why": auto_why(board, mv)})
    return candidates_messages(board, k), json.dumps({"candidates": cands}), "cands"
