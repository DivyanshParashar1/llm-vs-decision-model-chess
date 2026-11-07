import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import chess
from chessdecide import decider as D

B = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
L = ["Qxf7#", "Nf3", "d3"]


def test_render_deterministic():
    assert D.render_text(B) == D.render_text(B.copy())
    assert "White to move" in D.render_text(B)


def test_render_labels():
    labs = D.render_labels([{"san": "Qxf7#", "why": "mate"}, {"san": "Nf3", "why": ""}])
    assert labs == {"Qxf7#": "Qxf7#: mate", "Nf3": "the move Nf3"}


def test_scores_formats():
    assert D._scores_from_raw({"best_move": {"label": "Nf3", "confidence": 0.3}}, L) == {"Nf3": 0.3}
    assert D._scores_from_raw({"best_move": [("Qxf7#", 0.7), ("Nf3", 0.2)]}, L) == {"Qxf7#": 0.7, "Nf3": 0.2}
    assert D._scores_from_raw({"best_move": [{"label": "d3", "score": 0.1}]}, L) == {"d3": 0.1}
    assert D._scores_from_raw({"best_move": {"Qxf7#": 0.9, "Nf3": 0.05}}, L) == {"Qxf7#": 0.9, "Nf3": 0.05}
    assert D._scores_from_raw({"best_move": "Nf3"}, L) == {"Nf3": 1.0}
