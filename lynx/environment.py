from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from typing import Dict, List

from . import chess_utils
import chess

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


######################################################
# Encryption
######################################################

file_enc = {k: int(v) for (k, v) in zip('abcdefgh', '12345678')}
"""
Encrypts the files a-h. They are mapped to integers 1-8.
"""

file_dec = {val: key for (key, val) in file_enc.items()}
"""
Decrypts the values 1-8. They are mapped to the files a-h.
"""

rank_enc = {k: int(v) for (k, v) in zip('12345678', '12345678')}
"""
Encrypts the ranks 1-8. They are mapped to integers 1-8.
"""

rank_dec = {val: key for (key, val) in rank_enc.items()}
"""
Decrypts the values 1-8. They are mapped to the ranks 1-8.
"""

board_enc = {
    None: 0,  '.': 0,  # empty square

    'p': -1, 'P': 1,  # pawns

    'n': -2, 'N': 2,  # knights

    'b': -3, 'B': 3,  # bishops

    'r': -4, 'R': 4,  # rooks

    'q': -5, 'Q': 5,  # queens

    'k': -6, 'K': 6  # kings
}
"""
This is the board encryption. Each piece (None, "." = empty square) is mapped
to an integer, so values between -6 and 6 exist.
"""

board_dec = {val: key for (key, val) in board_enc.items()}
"""
Decrypts the values -6 to 6. They are mapped to board pieces 'pnbrqkPNBRQK.'.
"""


def encrypt_board(board: List[List[str]]) -> List[List[int]]:
    """
    Encrypts the given board. The argument must be a list of lists, each of which
    contains exclusively the characters "pnbrqk" (black), "PNBRQK" (white) and
    "." (or None, for an empty square). All values are mapped to integers given
    by 'board_enc' (-6 to 6 accordingly).

    :param board:
        The board to encrypt
    :return:
        The encrypted board
    """
    result = []
    for row in board:
        tmp = []
        for x in row:
            tmp.append(board_enc[x])
        result.append(tmp)
    return result


def fen_to_state(fen: str, color: int) -> Dict[str, np.ndarray]:
    """
    Returns a stateful value that represents the given FEN position.
    The stateful value has the structure given by `ChessEnvironment#observation_spec()`.

    :param fen:
        The FEN position of the board
    :return:
        A state that represents the given FEN
    """
    _board, _color, _castling, _en_passant, _half_moves, _full_moves = chess_utils.fen_to_tuple(fen)
    # board
    board = encrypt_board(_board)
    # color
    color = 0 if (_color == 'w') else 1
    # castling
    castling = []
    for x in 'KQkq':
        castling.append(1 if (x in _castling) else 0)
    # en-passant
    if len(_en_passant) > 1:
        en_passant = [file_enc[_en_passant[0]], rank_enc[_en_passant[1]]]
    else:
        en_passant = [0, 0]

    return {
        'board': np.asarray(board, dtype=np.int32),
        'color': np.asarray(color, dtype=np.int32),
        'castling': np.asarray(castling, dtype=np.int32),
        'enPassant': np.asarray(en_passant, dtype=np.int32),
        'halfMoves': np.asarray(_half_moves, dtype=np.int32),
        'fullMoves': np.asarray(_full_moves, dtype=np.int32),
        'selfColor': np.asarray(color, dtype=np.int32)
    }

######################################################
# Specs
######################################################

_square_spec = array_spec.BoundedArraySpec(
    shape=(2,), minimum=1, maximum=8, dtype=np.int32
    )
"""
The spec for one square on the board. This spec has two dimensions:
the rank and the file. Therefore, each dimension has a value between
1 and 8.
"""

_prom_spec = array_spec.BoundedArraySpec(
    shape=(1,), minimum=-6, maximum=6, dtype=np.int32
    )
"""
The spec for when a pawn promotes. It has only one dimension, which represents
the piece the pawn promotes to. The minimum and maximum values equal the pieces
of the board encryption (-6 to 6).
"""

_board_spec = array_spec.BoundedArraySpec(
    shape=(8, 8), minimum=-6, maximum=6, dtype=np.int32
    )
"""
The spec for the chess board. It has dimensions 8x8 (=64 squares), each of which
contains a piece. The minimum and maximum values of each square equal the board
encryption (-6 to 6).
"""

_color_spec = array_spec.BoundedArraySpec(
    shape=(), minimum=0, maximum=1, dtype=np.int32
    )
"""
The spec for a color.
- 0 = white
- 1 = black
"""

_castling_spec = array_spec.BoundedArraySpec(
    shape=(4,), minimum=0, maximum=1, dtype=np.int32
    )
"""
The spec for castling. Each dimension represents a right for castling
(KQkq), each of which is made up of two values:
- 0 = cannot caslte
- 1 = can castle
"""

_en_passant_spec = array_spec.BoundedArraySpec(
    shape=(2,), minimum=0, maximum=8, dtype=np.int32
    )
"""
The spec for capturing en-passant. This spec has two dimensions:
the rank and the file. Therefore, each dimension has a value between
0 and 8. Values of 0 mean no pawn can be captured en-passant.
"""

_natural_number_spec = array_spec.BoundedArraySpec(
    shape=(), minimum=0, maximum=10**6, dtype=np.int32
    )
"""
The spec for a natural number. Values between 0 and one million are
allowed.
"""

######################################################
# Environment
######################################################

class ChessEnvironment(py_environment.PyEnvironment):
    """
    This class represents a chess environment. One episode represents a whole
    game and the final reward depends on the color of the environment.
    """

    def __init__(self, color: int = 0):
        """
        Initializes the environment. The initial state is the initial position
        of a chessboard.

        :param color:
            The color to play with
        """
        # picks a square, moves to another square and promotes if necessary
        self._action_spec = {
            'from': _square_spec,
            'to': _square_spec,
            'prom': _prom_spec
        }
        # the FEN position of the board
        self._observation_spec = {
            'board': _board_spec,
            'color': _color_spec,
            'castling': _castling_spec,
            'enPassant': _en_passant_spec,
            'halfMoves': _natural_number_spec,
            'fullMoves': _natural_number_spec,
            'selfColor': _color_spec
        }
        self._color = color
        self._state = fen_to_state(chess_utils.STARTING_FEN, self._color)
        self._board = chess.Board(chess_utils.STARTING_FEN)

    def action_spec(self) -> Dict[str, array_spec.BoundedArraySpec]:
        """
        Defines the possible actions for a valid board position.
        Each move is made up of picking one piece, moving the
        piece to another square, and promoting the piece (pawn)
        if necessary. Therefore, an action is made up of a dictionary
        with the following structure:
        - "from":
            The square a piece is located at. It has the shape (1, 1) with values
            between 1 and 8 (file, rank).
        - "to":
            The square to move the selected piece to. It has the shape (1, 1) with values
            between 1 and 8 (file, rank).
        - "prom":
            A promotion if necessary. The piece is a number between -6 and 6, equivalent
            to the board encryption. Not all numbers express a legal promotion. 0 means
            no promotion.
        
        :return:
            The dictionary that defines the above spec
        """
        return self._action_spec

    def observation_spec(self) -> Dict[str, array_spec.BoundedArraySpec]:
        """
        Defines the observed environment. The observation represents the FEN
        position of the board. Therefore, it is made up of a dictionary with the
        following keys:
        - `board`:
            The observerd board. It has the shape (8, 8) with values between -6
            and 6. Each value is equal to a piece of the board encryption.
        - `color`:
            The color to move. There are only two possible values:
            - 0 = white
            - 1 = black
        - `castling`:
            The castling rights. Each dimension represents a right for castling (KQkq),
            each of which is made up of two values:
            - 0 = can not castle
            - 1 = can castle
        - `enPassant`:
            The square behind the pawn that can be captured en-passant. It has the shape (1, 1) with values
            between 0 and 8 (file, rank). Values of 0 mean no pawn can be captured en-passant.
        - `halfMoves`:
            The number of halfmoves since the last capture or pawn advance, used
            for the fifty-move rule. The minimum value is 1.
        - `fullMoves`:
            The number of the full move. It starts at 1, and is incremented after Black's move.
        - `selfColor`:
            The color of this player. This is used to compute the reward.
            - 0 = white
            - 1 = black

        :return:
            The dictionary that defines the above spec
        """
        return self._observation_spec

    def _reset(self):
        """
        Resets the environment.
        """
        self._state = fen_to_state(chess_utils.STARTING_FEN, self._color)
        self._board = chess.Board(chess_utils.STARTING_FEN)
        return ts.restart(self._state)

    def _step(self, action: Dict[str, np.ndarray]):
        # end the episode if the game is over
        if self._board.is_game_over():
            return self.reset()
        # else: play move
        s1 = file_dec[action['from'][0]] + rank_dec[action['from'][0]]
        s2 = file_dec[action['to'][0]] + rank_dec[action['to'][0]]
        prom = '' if (0 in action['prom']) else board_dec[action['prom'][0]]
        uci = s1 + s2 + prom
        # if the move is illegal: lose immediately
        try:
            move = self._board.parse_uci(uci)
            self._board.push(move)
            self._state = fen_to_state(self._board.fen(), self._color)
        except ValueError:
            return ts.termination(self._state, reward=-1)
        # compute reward if game ended after the move
        if self._board.is_game_over():
            result = self._board.result()
            factor = 1 if self._color == 0 else -1  # adds a dependency to the player's color
            reward = factor * (1 if (result == '1-0') else 0 if (result == '0-1') else 0)
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward=0, discount=1)
