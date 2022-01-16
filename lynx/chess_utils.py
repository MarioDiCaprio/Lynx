from typing import Tuple, List
import chess


STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
"""
The FEN position of the initial board.
"""


def fen_to_board(fen: str) -> List[List[str]]:
    """
    Extracts the board representation from the given FEN.
    The board is given as a 8x8 matrix (nested lists) of
    strings. Here, every uppercase letter (PNBRQK) represents
    an equivalent piece that belongs to white and every lowercase
    letter (pnbrqk) belongs to black. Empty spaces are designated
    by a dot (.).

    :param fen:
        The FEN position of the board
    :return:
        The matrix reoresentation of the board position
    """
    result = []
    board = chess.Board(fen).__str__().split('\n')
    for row in board:
        tmp = []
        for x in row:
            if x in 'pnbrqkPNBRQK.':
                tmp.append(x)
        result.append(tmp)
    return result


def fen_to_tuple(fen: str) -> Tuple[List[List[str]], str, List[str], str, int, int]:
    """
    Parses the given FEN board position and returns a tuple that is made of its components.
    The composition is made as follows:
    1. The board as a matrix (see 'fen_to_board')
    2. The color to move (string)
    3. Castling privileges (each character as an element of a list)
    4. En-passant capture (the square as a string)
    5. Half-move clock (int)
    6. Full-move clock (int)

    :param fen:
        The FEN position of the board
    :return:
        The FEN parsed as a tuple
    """
    tmp = fen.split()
    return (
        # board
        fen_to_board(fen),
        # color to move
        tmp[1],
        # castling rights
        [x for x in tmp[2]],
        # en-passant square
        tmp[3],
        # half-move clock
        int(tmp[4]),
        # full-move clock
        int(tmp[5])
    )
