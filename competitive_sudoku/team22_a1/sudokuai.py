#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
import math

# def eval_function(list_score):
#     diff_score = list_score[0] - list_score[1]
#     return diff_score

# class create_node:
#    def __init__(self, score):
#         self.left = None
#         self.right = None
#         self.score = score


# def insertion(arr, i ,N):
#     root = None
#     depth =0
#     if i<N:
#        root = create_node(arr[i])
        
#        root.left = insertion(arr, 2 * i + 1, N)
 
#        root.right = insertion(arr, 2 * i + 2, N)
       
#        depth += 1 
         
#        return root
        
        
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
        
    def compute_best_move(self, game_state: GameState) -> None:
        N = game_state.board.N
        m = game_state.board.m

        all_moves = []
        rewards = []
        reward = 0
        unique_values_col = []
        unique_values_row = []
        unique_values_square = []
        for i in range(N):
            for j in range(N):
                for v in range(1, N+1):
                    # if cell is empty
                    if (game_state.board.get(i, j) == SudokuBoard.empty):                 
                        # if cell to the left exists check left and is within boundaries
                        if j-1 >= 0:
                            for l in range(0, j):
                                unique_values_row.append(game_state.board.get(i, l))
                        # if cell to the right exists check right and is within boundaries
                        if j+1 <= N-1:
                            for r in range(j+1, N):
                                unique_values_row.append(game_state.board.get(i, r))
                         # if cell above exists check up and is within boundaries
                        if i-1 >= 0:
                            for u in range(0, i):
                                unique_values_col.append(game_state.board.get(u, j))
                        # if cell below exists check down and is within boundaries
                        if i+1 <= N-1:
                            for b in range(i+1, N):
                                unique_values_col.append(game_state.board.get(b, j))
                       
                        # find out in which region, coordinates of region and then check all values in that region
                        # find left corner of "the square" where the current coordinate belongs to
                        square_lc_i = (m * (math.floor( (i/m) )))
                        square_lc_j = (m * (math.floor( (j/m) )))
                        
                        # loop through all values in the square and add it to a list
                        for si in range(square_lc_i, square_lc_i+m):
                            for sj in range(square_lc_j, square_lc_j+m):
                                if (si, sj) != (i,j):
                                    unique_values_square.append(game_state.board.get(si, sj))

                        # create move if value is not seen in the regions and if move is not declared taboo before
                        if (v not in unique_values_col) and (v not in unique_values_row) and (v not in unique_values_square) and (not TabooMove(i, j, v) in game_state.taboo_moves): 
                            all_moves.append(Move(i,j,v))
                            # if move yields completion of ALL regions
                            if (0 not in unique_values_col) and (0 not in unique_values_row) and (0 not in unique_values_square):
                                reward += 7
                            # if move yields completion of TWO regions
                            elif ((0 not in unique_values_col) and (0 not in unique_values_row) and (0 in unique_values_square)) or \
                                ((0 not in unique_values_col) and (0 not in unique_values_square) and (0 in unique_values_row)) or \
                                ((0 not in unique_values_row) and (0 not in unique_values_square) and (0 in unique_values_col)):
                                reward += 2
                            # if move yields completion of ONE region
                            elif (0 not in unique_values_col) or (0 not in unique_values_row) or (0 not in unique_values_square): 
                                reward += 1 # reward of 1
                            else:
                                reward = 0
                            
                            rewards.append(reward)
                           
                        # empty list for next element
                        reward = 0
                        unique_values_row = []
                        unique_values_col = []
                        unique_values_square = []

        # choose random move
        move = random.choice(all_moves)
        self.propose_move(move)   
        while True:
            time.sleep(0.2)
            self.propose_move(random.choice(all_moves))
            