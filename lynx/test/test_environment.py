import unittest
import numpy as np

from lynx import environment, chess_utils

from tf_agents.environments import utils


class TestEnvironment(unittest.TestCase):

    def test_fen_to_state(self):
        result = environment.fen_to_state(chess_utils.STARTING_FEN, 0)
        expected = {
            'board': np.asarray([
                [-4, -2, -3, -5, -6, -3, -2, -4],
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [ 0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0,  0],
                [ 1,  1,  1,  1,  1,  1,  1,  1],
                [ 4,  2,  3,  5,  6,  3,  2,  4]
            ], dtype=np.int32),
            'color': np.asarray(0, dtype=np.int32),
            'castling': np.asarray([1, 1, 1, 1], dtype=np.int32),
            'enPassant': np.array([0, 0], dtype=np.int32),
            'halfMoves': np.asarray(0, dtype=np.int32),
            'fullMoves': np.asarray(1, dtype=np.int32),
            'selfColor': np.asarray(0, dtype=np.int32)
        }
        np.testing.assert_array_equal(expected['board'], result['board'])
        np.testing.assert_array_equal(expected['color'], result['color'])
        np.testing.assert_array_equal(expected['castling'], result['castling'])
        np.testing.assert_array_equal(expected['enPassant'], result['enPassant'])
        np.testing.assert_array_equal(expected['halfMoves'], result['halfMoves'])
        np.testing.assert_array_equal(expected['fullMoves'], result['fullMoves'])
        np.testing.assert_array_equal(expected['selfColor'], result['selfColor'])

    def test_state_shapes(self):
        result = environment.fen_to_state(chess_utils.STARTING_FEN, 0)
        self.assertEqual(result['board'].shape, (8, 8))
        self.assertEqual(result['color'].shape, ())
        self.assertEqual(result['castling'].shape, (4,))
        self.assertEqual(result['enPassant'].shape, (2,))
        self.assertEqual(result['halfMoves'].shape, ())
        self.assertEqual(result['fullMoves'].shape, ())
        self.assertEqual(result['selfColor'].shape, ())

    def test_environment_is_valid(self):
        e = environment.ChessEnvironment()
        utils.validate_py_environment(e, episodes=10)
