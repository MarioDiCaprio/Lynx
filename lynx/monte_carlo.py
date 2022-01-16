from __future__ import annotations

from typing import List

import chess
import chess_utils


class Node:

    parent: Node
    """ This is the parent node. """
    children: List[Node] = []
    """ These are the child-nodes of this node. """
    position: str
    """ This is the FEN board position of this node. """

    def __init__(self, parent: Node = None, fen: str = chess_utils.STARTING_FEN, make_children: bool = False):
        self.position = fen
        self.parent = parent
        if make_children:
            self.make_children()

    def make_children(self):
        board = chess.Board(self.position)
        for move in board.legal_moves:
            board.push(move)
            self.children.append(Node(board.fen()))
            board.pop()

    def play(self, move_uci: str):
        board = chess.Board(self.position)
        board.push(chess.Move.from_uci(move_uci))
        
