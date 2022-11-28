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
    
    if i<N:
       root = create_node(arr[i])
        
       root.left = insertion(arr, 2 * i + 1, N)
 
       root.right = insertion(arr, 2 * i + 2, N)
         
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
        
        

        def possible(i, j, value):
            return game_state.board.get(i, j) == SudokuBoard.empty \
                   and not TabooMove(i, j, value) in game_state.taboo_moves

        all_moves = [Move(i, j, value) for i in range(N) for j in range(N)
                     for value in range(1, N+1) if possible(i, j, value)]
        move = random.choice(all_moves)
        self.propose_move(move)
        while True:
            time.sleep(0.2)
            self.propose_move(random.choice(all_moves))

