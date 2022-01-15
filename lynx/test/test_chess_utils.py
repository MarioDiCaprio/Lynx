import unittest
from lynx import chess_utils


class TestChessUtils(unittest.TestCase):

    def test_fen_to_board(self):
        fen = 'r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4'
        result = chess_utils.fen_to_board(fen)
        expected = [
            ['r', '.', 'b', 'q', 'k', 'b', '.', 'r'],
            ['p', 'p', 'p', 'p', '.', 'Q', 'p', 'p'],
            ['.', '.', 'n', '.', '.', 'n', '.', '.'],
            ['.', '.', '.', '.', 'p', '.', '.', '.'],
            ['.', '.', 'B', '.', 'P', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['P', 'P', 'P', 'P', '.', 'P', 'P', 'P'],
            ['R', 'N', 'B', '.', 'K', '.', 'N', 'R']
        ]
        self.assertEqual(expected, result)

    def test_fen_to_tuple(self):
        fen = 'r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4'
        result = chess_utils.fen_to_tuple(fen)
        expected = (
            [
                ['r', '.', 'b', 'q', 'k', 'b', '.', 'r'],
                ['p', 'p', 'p', 'p', '.', 'Q', 'p', 'p'],
                ['.', '.', 'n', '.', '.', 'n', '.', '.'],
                ['.', '.', '.', '.', 'p', '.', '.', '.'],
                ['.', '.', 'B', '.', 'P', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '.', '.', '.'],
                ['P', 'P', 'P', 'P', '.', 'P', 'P', 'P'],
                ['R', 'N', 'B', '.', 'K', '.', 'N', 'R']
            ],
            'b',
            ['K', 'Q', 'k', 'q'],
            '-',
            0,
            4
        )
        self.assertEqual(expected, result)
