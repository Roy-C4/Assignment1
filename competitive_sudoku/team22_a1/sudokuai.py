#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
import math

#class for creating individual nodes for the game tree
class Node:
       def __init__(self, key,mov):
        self.child= []
        self.key = key
        self.mov = mov
        
def newNode():
    temp = Node(0,Move(0,0,0)) #initialising the node value
    return temp

class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()   
        
    def compute_best_move(self, game_state: GameState) -> None:
            
        def GetLegalMoves(game_state: GameState) -> None:
            N = game_state.board.N
            m = game_state.board.m
            n = game_state.board.n

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
                            square_lc_j = (n * (math.floor( (j/n) )))
                            
                            # loop through all values in the square and add it to a list
                            for si in range(square_lc_i, square_lc_i+m):
                                for sj in range(square_lc_j, square_lc_j+n):
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
                            
                        
            return all_moves,rewards

        #checks if the cell to be filled is empty and it is not a taboo move
        def possible(i, j, value, game_state:GameState):
            return game_state.board.get(i, j) == SudokuBoard.empty \
                    and not TabooMove(i, j, value) in game_state.taboo_moves

        #create the game tree with depth of 2
        def create_tree(game_state: GameState):
            all_moves, score = GetLegalMoves(game_state)

            root = newNode()
            k=0
            for move in all_moves:
                if not possible(move.i, move.j, move.value,game_state):
                    continue
            
                (root.child).append(Node(score[k],move))

                temp_game = game_state
                temp_game.board.put(move.i, move.j, move.value)
                legal_move, rewards = GetLegalMoves(temp_game)
                opp = 0
                #for each depth add the scores for the new possible legal moves to the tree
                for move_opp in legal_move:
                    if not possible(move_opp.i, move_opp.j, move_opp.value,game_state):
                        continue

                    (root.child[k].child).append(Node(rewards[opp],move_opp))
                    opp = opp + 1
                    
                k= k+1

            return root

        #implementation of minimax algorithm with alpha beta pruning
        def minimax(curr_depth, head, max_player, depth, alpha, beta) -> Node:   
            if curr_depth == depth or len(head.child)==0:
                return head
            
            #check if maximising player
            if (max_player):
                nodes = [minimax(curr_depth + 1, c, False, depth, alpha, beta) for c in head.child]
                mx = float('-inf')
                max_node = None #initialising the max_node to none value 
                
                for node in nodes:
                    if node is not None and node.key > mx:   #check if the value in nodes is not null
                        mx = node.key                        #if mx is less than the current node value, then update mx, basically finding the maximum of node.key and mx
                        max_node = node                      #update max_node
                    alpha = max(alpha, max_node.key)
                    if beta <= alpha:
                        break
                return max_node                              #return the max_node for each depth of the game tree
            
            else:
                nodes = [minimax(curr_depth + 1, c,
                            True, depth, alpha, beta)
                        for c in head.child]
                mn = float('inf')
                min_node = None
                for node in nodes:
                    if node.key < mn:                       #if mn is greater than the current node value, then update mn, basically finding the minimum of node.key and mn
                        mn = node.key
                        min_node = node                     #update min_node
                    beta = min(beta, min_node.key)
                    if beta <= alpha:
                        break
                return min_node

        def choose_move(game_state: GameState) -> Move:     #choose the optimal move with the minimax algorithm
            game_tree_head = create_tree(game_state)
            return minimax(0,game_tree_head,True,2, float('-inf'), float('inf')).mov
                          
        move = choose_move(game_state)                      #pick an optimal move 
        self.propose_move(move)   





            
