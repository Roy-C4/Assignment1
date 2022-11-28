#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class create_node:
   def __init__(self, score):
        self.left = None
        self.right = None
        self.score = score


def insertion(arr, i ,N):
    root = None
    depth =0
    if i<N:
       root = create_node(arr[i])
        
       root.left = insertion(arr, 2 * i + 1, N)
 
       root.right = insertion(arr, 2 * i + 2, N)
       
       depth += 1 
         
       return root
        
        
 #need to find the scores retriveal function to add it into an array

"""def minimax(game_state, depth):
    if game_state.taboo_moves:
        return -1
    
   # if depth == 0:
   #     return 1 if max_player else -1
    
    if max_player:
        return max(minimax(game_state.initial_board*2, False), 
                   minimax(((position*2)+1), False)
                   
   else:
        return min(minimax(game_state.initial_board*2, False), 
                   minimax(((position*2)+1), False))  """ 

class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()   
        
        
    

    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        N = game_state.board.N
        n = game_state.board.n
        m = game_state.board.m

        def possible(i, j, value):
            return game_state.board.get(i, j) == SudokuBoard.empty \
                   and not TabooMove(i, j, value) in game_state.taboo_moves

        all_moves = [Move(i, j, value) for i in range(N) for j in range(N)
                     for value in range(1, N+1) if possible(i, j, value)]

        for e in all_moves:
            print(e)
        for e in all_moves:
            if e.i != 0 or e.i != N-1: # if not row 0 or row N
                for u in range(0, e.i) : # check cells above
                    for b in range(e.i+1, N): # check cells below
                        if game_state.board.get(u, e.j) == e.value or game_state.board.get(b, e.j) == e.value:
                            all_moves.remove(e)
            if e.i == 0: # if row 0
                for b in range(e.i+1, N): # check cells below
                    if game_state.board.get(e.i, b) == e.value:
                            all_moves.remove(e)
            if e.i == N-1: # if last row
                for u in range(0, e.i): # check cells above
                    if game_state.board.get(u, e.j) == e.value:
                            all_moves.remove(e)
            if e.j != 0 or e.j != N-1: # if not col 0 or col N
                for l in range(0, e.j) : # check cells left
                    for r in range(e.j+1, N): # check cells right
                        if game_state.board.get(e.i, l) == e.value | game_state.board.get(e.i, r) == e.value:
                            all_moves.remove(e)
            if e.j == 0: # if first col
                for r in range(e.j+1, N): # check right
                    if game_state.board.get(e.i, r) == e.value:
                        all_moves.remove(e)
            if e.j == N-1: # if last col
                for l in range(0, e.j): # check left
                    if game_state.board.get(e.i, l) == e.value:
                        all_moves.remove(e)
            
            subgrids = N/m






        for e in all_moves:
            print(e)
        move = random.choice(all_moves)
        self.propose_move(move)
        while True:
            time.sleep(0.2)
            self.propose_move(random.choice(all_moves))

