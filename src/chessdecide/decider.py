"""GLiNER2.5-Decide wrapper: versioned input rendering (shared by training and eval) + scoring.

API verified in docs/gliner_notes.md.
"""
from __future__ import annotations

import chess

from .proposer import board_ascii

RENDER_VERSION = "r1"
TASK = "best_move"


def render_text(board: chess.Board) -> str:
    """Decider text input. Kept well under 512 tokens (board ~130 tokens, FEN ~40)."""
    side = "White" if board.turn == chess.WHITE else "Black"
    return (f"Chess position. {side} to move. Which move is best?\n"
            f"FEN: {board.fen()}\n"
            f"Board (uppercase = White, lowercase = Black):\n{board_ascii(board)}")


def render_labels(cands: list[dict]) -> dict[str, str]:
    """{SAN: description}. Description = LLM rationale if present."""
    out = {}
    for c in cands:
        why = (c.get("why") or "").strip()
        out[c["san"]] = f"{c['san']}: {why}" if why else f"the move {c['san']}"
    return out


def _scores_from_raw(raw, labels: list[str]) -> dict[str, float]:
    """Pull {label: score} out of whatever structure gliner2 returns."""
    task = raw.get(TASK, raw) if isinstance(raw, dict) else raw
    scores: dict[str, float] = {}

    def visit(x):
        if isinstance(x, dict):
            lab = x.get("label") or x.get("name") or x.get("text")
            sc = x.get("confidence", x.get("score", x.get("probability")))
            if lab in labels and isinstance(sc, (int, float)):
                scores[lab] = max(scores.get(lab, 0.0), float(sc))
            for k, v in x.items():
                if k in labels and isinstance(v, (int, float)):
                    scores[k] = float(v)
                elif isinstance(v, (dict, list, tuple)):
                    visit(v)
        elif isinstance(x, (list, tuple)):
            if len(x) == 2 and x[0] in labels and isinstance(x[1], (int, float)):
                scores[x[0]] = float(x[1])
            else:
                for v in x:
                    visit(v)
        elif isinstance(x, str) and x in labels and not scores:
            scores[x] = 1.0

    visit(task)
    return scores


class GlinerDecider:
    def __init__(self, model_name_or_path: str = "fastino/GLiNER2.5-Decide", device: str | None = None,
                 use_descriptions: bool = True):
        from gliner2 import AutoExtractor
        self.model = AutoExtractor.from_pretrained(model_name_or_path)
        if device:
            self.model = self.model.to(device)
        self.model.eval()
        self.use_descriptions = use_descriptions
        self.full_scores = None  # decided on first call: does format_results=False give all labels?

    def _tasks(self, cands):
        labels = render_labels(cands)
        return {TASK: labels if self.use_descriptions else list(labels)}

    def choose_batch(self, boards: list[chess.Board], cand_lists: list[list[dict]], batch_size: int = 16):
        """Returns list of (chosen_uci, {san: score})."""
        out = []
        for board, cands in zip(boards, cand_lists):
            sans = [c["san"] for c in cands]
            text = render_text(board)
            scores = {}
            if self.full_scores is not False:
                try:
                    raw = self.model.classify_text(text, self._tasks(cands), format_results=False,
                                                   include_confidence=True, threshold=0.0)
                    scores = _scores_from_raw(raw, sans)
                    if self.full_scores is None:
                        self.full_scores = len(scores) == len(sans)
                        print(f"[decider] format_results=False gives all-label scores: {self.full_scores}; raw sample: {repr(raw)[:400]}")
                except Exception as e:  # noqa: BLE001
                    if self.full_scores is None:
                        print(f"[decider] format_results=False failed ({e!r}); using top-label mode")
                    self.full_scores = False
            if len(scores) < len(sans):
                raw = self.model.classify_text(text, self._tasks(cands), include_confidence=True)
                top = _scores_from_raw(raw, sans)
                if top and not scores:
                    scores = top
                elif top:
                    scores.update({k: v for k, v in top.items() if k not in scores})
            best_san = max(scores, key=scores.get) if scores else sans[0]
            uci = next(c["uci"] for c in cands if c["san"] == best_san)
            out.append((uci, scores))
        return out
