import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import chess
from chessdecide import proposer as P

B = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")


def test_prompt_contains_legal_moves_and_board():
    msg = P.candidates_messages(B, 5)[1]["content"]
    assert "Qxf7#" in msg and "White to move" in msg and "a b c d e f g h" in msg


def test_parse_single_variants():
    assert P.parse_single(B, '{"move": "Qxf7#"}').uci() == "h5f7"
    assert P.parse_single(B, '<think>hmm</think>\n```json\n{"move": "Qxf7"}\n```').uci() == "h5f7"
    assert P.parse_single(B, '{"move": "h5f7"}').uci() == "h5f7"
    assert P.parse_single(B, '{"move": "Qh8"}') is None
    assert P.parse_single(B, "garbage") is None


def test_castling_zero_notation():
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert P.to_move(b, "0-0").uci() == "e1g1"
    assert P.to_move(b, "0-0-0").uci() == "e1c1"


def test_parse_candidates_dedup_illegal_and_pad():
    txt = ('{"candidates": [{"move": "Qxf7#", "why": "mate"}, {"move": "Qxf7", "why": "dup"}, '
           '{"move": "Ke3", "why": "illegal"}, {"move": "Nf3", "why": "develop"}]}')
    cands, n_valid, n_pad = P.parse_candidates(B, txt, k=5)
    assert [c["san"] for c in cands[:2]] == ["Qxf7#", "Nf3"]
    assert n_valid == 2 and n_pad == 3 and len(cands) == 5
    assert len({c["uci"] for c in cands}) == 5
    assert cands[2]["padded"] and cands[0]["why"] == "mate"


def test_pad_when_few_legal_moves():
    b = chess.Board("7k/8/6Q1/8/8/8/8/K7 b - - 0 1")  # black king: only Kh8->? few moves
    cands, n_valid, n_pad = P.parse_candidates(b, "nothing", k=5)
    assert n_valid == 0 and len(cands) == b.legal_moves.count()
