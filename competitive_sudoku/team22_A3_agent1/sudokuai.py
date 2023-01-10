#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

from itertools import chain
import numpy as np
import math
from math import inf
import random
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class Node():
    """
        Node object that represents a game state

        @param GameState state: the current game state
        @param int depth: current depth of node
        @param int ti: number of wins
        @param int ni: visit count of node
        @param int N: visit count of parent 
        @param Move(i,j,v) mov: move object (e.g. (0, 3) -> 4),
        @param int reward: contains the reward that a move yields
        @param Node parent: the parent node
        @param list score: current score of the game state
        @param list child: contains the children of the node
    """
    def __init__(self, state, depth=0, ti=0, ni=0, N=0, move=False, reward=False, parent=False):
        # current state 
        self.state = state
        # current depth
        self.depth = depth
        # number of wins
        self.ti = ti
        # number of visits
        self.ni = ni
        # number of visits of parent
        self.N = N
        # move 
        self.move = move
        # reward of move
        self.reward = reward
        # parent node
        self.parent = parent
        # score of game state [P1-P2]
        self.score = [0, 0]
        # children
        self.child = []
        
        
class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()   
        
    def compute_best_move(self, game_state: GameState) -> None:
        """
            Computes the best move for the agent for each turn
            Following methods are defined within this method: 
                -- helper functions
                get_legal_moves(game_state): get legal moves given a game state/board,
                evaluate_score(score): calculate difference in score, indicate who has won and return a value
                ro_policy(score): rollout policy
                simulate(a_i,, s_i, reward, parent, depth): put action on board, get new state, and get the attributes: state, reward, score, parent, and depth
                -- monte carlo search tree functions
                select(current_node, C): return/select the child with the highest UCB1 value
                expand(current_node): add children with their respective attributes
                rollout: rollout current node, and get value 
                backpropagate: backprogate the value upwards, and update the attributes (visit count, win count etc.)
   
            @param GameState game_state: state of the game
            @return: None

        """       
        def get_legal_moves(game_state: GameState): 
            """
                Gets the legal moves of a game state

                @param GameState game_state: the current game state
                @return dict all_moves: a dict of legal moves of the current game state split into unsure and sure moves
                @return list unsure_moves: list that stores moves created when for a given pos 
                multiple values can be filled in
                @return list rewards: a list of the respective rewards, so the first legal move 
                in the all_moves list has the reward rewards[0] and so on. 
                @return int empty_cells: number of empty cells, indicates total depth/turns of game
                to be used for tree construction
            """      
            m = game_state.board.region_height()
            n = game_state.board.region_width()
            N = m*n

            def InRow(i, board):
                """
                    Store all values seen in the row

                    @param int i: current i pos
                    @param Board board: current board of the game
                    @return list in_row: list containing the values seen in the row
                """
                in_row = []
                for j in range(N):
                       in_row.append(board.get(i,j))
                return in_row
                    
            def InCol(j, board):
                """
                    Store all values seen in the col

                    @param int j: current j pos
                    @param Board board: current board of the game
                    @return list in_col: list containing the values seen in the col
                """
                in_col = []
                for i in range(N):
                    in_col.append(board.get(i,j))
                return in_col
                
            def InSquare(i, j, board, m, n):
                """
                    Store all values seen in the box/square region of the current pos
        
                    @param int i: the current i index 
                    @param int j: the current j index
                    @param Board board: current board of the game
                    @param int m: region/square height
                    @param int n: region/square width
                    @return list nums: stores the values seen in the square region
                """
                # stores values seen in the square region of the current pos
                nums = []

                # find out in which region, coordinates of region and then check all values in that region
                # find left corner of "the square" where the current coordinate belongs to
                            
                # gives i coordiante of upper left corner of the subgrid that the current pos belongs to
                square_lc_i = (m * (i // m) )
                # gives j coordinate of upper left corner of the subgrid that the current pos belongs to
                square_lc_j = (n * (j // n) )
                            
                # loop through each value in the square
                for si in range(square_lc_i, square_lc_i+m):
                    for sj in range(square_lc_j, square_lc_j+n):
                        nums.append(board.get(si, sj))
                return nums
                
            # set of all values that can be filled in given N
            all_num = set(range(1,N+1))
           
            # list that stores all legal moves and resp. rewards according to C0 split into unsure and sure moves
            all_moves = {"sure-moves": {}, "unsure-moves": {}}
            # keep track of number of empty cells -> indication of total depth
            empty_cells = 0
            # loop through board
            for i in range(N):
                for j in range(N):
                    # if cell is empty
                    if(game_state.board.get(i,j) == SudokuBoard.empty):
                        # increase counter
                        empty_cells = empty_cells+1

                        # values present in the row except for 0
                        present_in_row = set(InRow(i, game_state.board))-set([0])
                        # values present in the col ......
                        present_in_col = set(InCol(j, game_state.board))-set([0])
                        # values present in the square
                        present_in_square = set(InSquare(i,j, game_state.board, m, n)) - set([0])
                        # all possible values that can be filled in the current pos/cell
                        all_k = all_num - present_in_row - present_in_col - present_in_square
                        
                        # count how many cells are empty in the respective region
                        r1 = N - len(present_in_row)
                        r2 = N - len(present_in_col)
                        r3 = N - len(present_in_square)
                        # score that keeps track of how many regions are completed
                        score = 0
                        # if only one cell is empty increase counter
                        if(r1 == 1):
                            score = score+1
                        if(r2 == 1):
                            score = score+1
                        if(r3 == 1):
                            score = score+1

                        # if there are only two cells and we fill one in, that means the opponent can easily score, so we give a penalty later.
                        if (r1 == 2):
                            score = score-1
                        if (r2 == 2):
                            score = score-1
                        if (r3 == 2):
                            score = score-1

                        # for possible value                    
                        for val in all_k:
                            # if it is a taboo then do nothing with it
                            if(TabooMove(i, j, val) in game_state.taboo_moves):
                                continue
                                 
                            # calc rewards
                            # if only ONE region is completed, reward of 1 is given
                            if(score == 1):
                                rewards = 1
                            # if only TWO regions are completed, reward of 3 is given
                            elif(score == 2):
                                rewards = 3
                            # if all THREE regions completed ...
                            elif(score == 3):
                                rewards = 7
                            # NO regions yields a reward of 0
                            elif(score == 0):
                                rewards = 0

                            # if the opponent is able to complete one region after our move, give a penalty 
                            elif(score == -1):
                                rewards = -2 
                            # same for two regions
                            elif(score == -2):
                                rewards = -5
                            # and for three
                            elif(score == -3):
                                rewards = -7

                        # if possible values per pos is at least two
                        if len(all_k) >= 2:
                            # loup through the values
                            for v in all_k:
                                # if not seen yet
                                if (i,j,v) not in all_moves["unsure-moves"]:
                                    # create the move and add it to unsure_move list
                                    all_moves["unsure-moves"].update({(i,j, v): rewards})
                        else:
                            if len(all_k) == 1:
                                all_moves["sure-moves"].update({(i,j, list(all_k)[0]): rewards})
                    
            return all_moves, empty_cells
       
        
        def heuristic(state):
            """
                unsure-sure moves heuristic

                @param GameState state: current state of the game
                @return dict considered_moves: moves that are considered
                @return int empty_cells: nr of empty cells aka depth
            """
            # nr of unsure moves that need to be at least present
            nr = 2
            # nr of unsure moves chosen
            rm = 2

            # get dictionary
            all_moves, empty_cells = get_legal_moves(state)

            unsure_moves = all_moves["unsure-moves"]
            sure_moves = all_moves["sure-moves"]

            # if unsure moves list is not empty and it contains less than x moves given x or unsure moves is empty
            if (unsure_moves != None and len(unsure_moves) < nr) or (unsure_moves == None):
                considered_moves = sure_moves
            # if unsure moves is not empty and it contains more than x moves given x
            elif unsure_moves != None and len(unsure_moves) >= nr:
                chosen_unsure_moves = dict(random.sample(all_moves["unsure-moves"].items(), rm))
                considered_moves = dict(chain(all_moves["sure-moves"].items(), chosen_unsure_moves.items()) )   

            # make one big list for testing purposes
            return considered_moves, empty_cells

        def evaluate_score(score):
            """
                Evaluate the score of the game state and return a value 
            
                @param list score: score of game state
                @return int: 1 if agent has won, 0 if it is a draw, and -1 if agent has lost. 
            
            """
            # if agent starts, subtract first index from second index
            if game_state.current_player() == 1:
                diff = score[0] - score[1]
            # else other way around
            else:
                diff = score[1] - score[0]
            
            # if diff is positive, it is a win
            if diff > 0:
                return 1
            # else a draw
            elif diff == 0:
                return 0
            # -1 if lose
            else:
                return -1
        
        def ro_policy(moves):
            """
                Rollout policy: randomly pick a move
                @param list moves: list with possible moves
                @return Move move: random move
                @return int index: index of random move
            """
            # pick a random move
            move = random.choice(list(moves))
            return move

        def simulate(a_i, s_i, reward, parent, depth):
            """
                During rollout, put the move on the board, get the new state, store all the info in a Node variable

                @param Move a_i: current action to be simulated
                @param GameState s_i: current state
                @param int reward: reward of move
                @param Node parent: parent node
                @param int depth: current depth of node
                @return Node node: node created during rollout/simulation
            """
            # put move on board and return new state
            s_i.board.put(a_i[0], a_i[1], a_i[2])
   
            # save new state object
            self.save(s_i)

            # load state object and store the information in a variable of type Node
            node = Node(self.load())

            # assign depth
            node.depth = depth 

            # assign reward
            node.reward = reward 

            # assign its parent 
            node.parent = parent

            # assign the move made
            node.move = Move(a_i[0], a_i[1], a_i[2])

            if node.depth % 2 == 0:
                # accumulate reward, to keep track of score
                node.score[0] = node.parent.score[0]
                node.score[1] = node.parent.score[1] + reward
            else:
                # accumulate reward, to keep track of score
                node.score[0] = node.parent.score[0] + reward
                # copy parents score
                node.score[1] = node.parent.score[1]

            # undo the move
            s_i.board.put(a_i[0], a_i[1], 0)

            # return the new node
            return node
        
        def select(current_node, C):
            """
                Phase 1 of Monte Carlo Search Tree Alg.: 
                Select child node with highest UCB1 value

                @param Node current_node: current node
                @param int C: constant for fine-tuning / trade-off between exploration and exploitation
                @return Node child: child with highest UCB1 value
            """
            UCB1_values = []
            # iterate over children
            for child in current_node.child:
                ti = child.ti
                ni = child.ni
                N = child.N
                try:
                    # average value: number wins / number visits
                    exploitation_term = ti / ni
                    # exploration term
                    exploration_term = C * math.sqrt(math.log( N ) / ni)
                    # UCB1 formula - Upper Confidence Bounds Monte-Carlo
                    UCB1 = exploitation_term + exploration_term
                # if ni is zero, this yields UCB1 outcome of infinity
                except ZeroDivisionError:
                    UCB1 = inf

                UCB1_values.append(UCB1)

            # return child with highest UCB1 value
            return current_node.child[np.argmax(UCB1_values)]
        
        def expand(current_node):
            """
                Phase 2 of Monte Carlo Search Tree Alg.:
                Expand currentnode/add its children and attributes.
                
                @param Node current_node: current node in tree
                @return: None 
            """

            # get legal moves from current state
            all_moves, empty_cells = heuristic(current_node.state)

            # iterate over possible moves
            for move, reward in all_moves.items():

                # depth of child
                depth = current_node.depth + 1

                # update attributes of child: state, reward, score, parent, and depth
                child = simulate(move, current_node.state, reward, current_node, depth)
      
                # add child
                current_node.child.append(child)

                # once done with this child, undo the move
                current_node.state.board.put(move[0], move[1], 0)

        def rollout(s_i):
            """
                Phase 3 of Monte Carlo Search Tree Alg.:
                Rollout current node and retrieve value

                @param Node s_i: current node 
                @return int: value/outcome of game (win/draw/lose)
                @return Node s_i: leaf node
            """
            # as long as the current state has moves to play
            while True:
                # get legal moves from current state
                all_moves, empty_cells = heuristic(s_i.state)
                # if no more moves left
                if not all_moves:
                    # return value and state
                    return evaluate_score(s_i.score),  s_i
                
                # get random move, and its index
                a_i = ro_policy(all_moves)
                
                # get its respective reward
                reward = all_moves[a_i]

                # depth of child
                depth = s_i.depth + 1

                # play/simulate move until leaf
                s_i = simulate(a_i, s_i.state, reward, s_i, depth)
               
        def backpropagate(node, value):
            """
                Phase 4 of Monte Carlo Search Tree Alg.:
                Backpropagate the value upwards, and update the attrbiutes along the way:
                the visit count of each node, the nr. of wins, and the attribute: parent visit count

                @param Node node: leaf node
                @param int value: outcome of gamestate
                @return: None
            """

            # while parent attribute exists
            while node.parent:
                # increment visit count of node
                node.ni = node.ni + 1
                # increment attribute: parent visit count
                node.N = node.N + 1
                # if outcome is: win
                if value == 1:
                    # increment number of wins
                    node.ti = node.ti + value
                # new node is parent
                node = node.parent

            # also make sure to update root node
            node.ni = node.ni + 1
            if value == 1:
                node.ti = node.ti + value
        
        def mcts(root, nr_iterations, C):
            """
                Monte Carlo Search Tree Alg.:
                Outline, self.propose best move based on highest visit count with every iteration 
                latest best move proposed until time limit is up

                @param Node root: root node of tree/initial state
                @param int nr_iterations: number of iterations of MCTS search
                @param int C: fine tuning parameter

                @return: None
            
            """
            for i in range(nr_iterations):
                # start with initial root node
                node = root

                # select
                while node.child:
                    node = select(node, C)
                
                # print("CHOSEN NODE", node.move, "ITS REWARD", node.reward, "ITS PARENT", node.parent.move, "ITS STATE", node.state, "ITS SCORE", node.score, "ITS DEPTH", node.depth)

                # if node has been visited before
                if node.ni != 0:
                    # expand
                    all_moves, empty_cells = heuristic(node.state)
                    if all_moves:
                        expand(node)
                        # print("EXPANDED NODE", node.move,  "ITS STATE", node.state, "ITS PARENT", node.parent.move, "ITS REWARD", node.reward, "ITS DEPTH", node.depth, "ITS SCORE", node.score, "ITS CHILDREN", node.child)
                        # select and continue with rollout
                        node = select(node, C)
                        # print("CHOSEN NODE AFTER EXPANSION", node.move, "ITS REWARD", node.reward, "ITS PARENT", node.parent.move, "ITS STATE", node.state, "ITS DEPTH", node.depth, "ITS SCORE", node.score)

                # simulate/rollout
                value, leafnode = rollout(node)

                # backpropagate
                backpropagate(leafnode, value)
            
                # after each iteration, choose node with highest visit count - robut child as best move               
                most_visited_index = np.argmax([child.ni for child in root.child])
                best_node = root.child[most_visited_index]
                best_move = best_node.move
                # propose move
                self.propose_move(best_move)
            # print("BEST MOVE", best_move)

        # initialize root node
        root = Node(game_state)

        # expand root node with its children 
        expand(root)

        # start Monte Carlo Search Tree Alg. with nr_iterations:1000, and with C = 2
        mcts(root, 10000, 2)

  
       

    
        
        
