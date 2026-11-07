import json, random, sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import chess
from chessdecide import sft
from chessdecide.proposer import parse_candidates, parse_single

FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
MPV = json.dumps([{"uci": "h5f7", "san": "Qxf7#", "cp": 100000, "mate": 1},
                  {"uci": "g1f3", "san": "Nf3", "cp": 50, "mate": None}])


def test_auto_why():
    b = chess.Board(FEN)
    assert sft.auto_why(b, chess.Move.from_uci("h5f7")) == "queen captures pawn, checkmate"
    assert sft.auto_why(b, chess.Move.from_uci("g1f3")) == "knight to f3"


def test_examples_roundtrip_through_parsers():
    b = chess.Board(FEN)
    _, tgt, fmt = sft.make_example(FEN, MPV, 5, random.Random(0), p_single=1.0)
    assert fmt == "single" and parse_single(b, tgt).uci() == "h5f7"
    _, tgt, fmt = sft.make_example(FEN, MPV, 5, random.Random(0), p_single=0.0)
    cands, n_valid, _ = parse_candidates(b, tgt, 5)
    assert fmt == "cands" and n_valid == 2 and cands[0]["uci"] == "h5f7"
