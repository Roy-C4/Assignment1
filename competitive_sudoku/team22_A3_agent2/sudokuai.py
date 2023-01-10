#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
from competitive_sudoku.execute import solve_sudoku
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai

class Node:
    """
        Node object that represents a game state

        @param list child = contains the children of the node
        @param Move(i,j,v) mov: move object (e.g. (0, 3) -> 4),
        @param int reward: contains the reward that a move yields
        @param int diff: stores the difference between the "final scores" of a game state
        @param int value: stores best value seen when minimax is applied
        @param list final_score: stores the "final scores ([5, 2])" of type list
    """
    def __init__(self, mov, reward, diff=False, value=False, final_score=False):
        
        self.child= []
        self.mov = mov
        self.reward = reward
        self.empty_cells = -1
        self.diff = diff
        self.value = value
        self.final_score = []
        
        
    
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
                new_node(): initializes a node, 
                eval_function(node): takes a node and calculates the difference in scores,
                get_legal_moves: get legal moves given a game state/board,
                get_children: get children nodes of current node, 
                minimax: search through game tree and find best move with alpha beta pruning,
                tree: construct game tree, keep track of the final scores, calc difference between
                the scores and assign them to the nodes
            @param GameState game_state: state of the game
            @return: None

        """
        
        def new_node():
            """
                Initiates a node
                @return: a node object
            """
            temp = Node(Move(0,0,0), 0) 
            return temp
        
        def eval_function(node):
            """
                Compute difference in final scores.

                @param Node node: a Node object with its respective attributes
                @return int: the difference of the scores from the first player's perspective
            """
            list_score = node.final_score
            diff_score = list_score[0] - list_score[1]
            return diff_score

        def get_legal_moves(game_state: GameState): 
            """
                Gets the legal moves of a game state

                @param GameState game_state: the current game state
                @return list all_moves: a list of legal moves of the current game state
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
           
            # list that stores all legal moves according to C0
            all_moves = [] 
            # list that stores the respective rewards   
            rewards = []
            # list that stores moves 
            unsure_moves = []
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
                        # penaly to keep track of potential penalty if the opponent can score easy points
                        score = 0
                        penalty = 0

                        # if only one cell is empty increase counter
                        if(r1 == 1):
                            score = score+1
                        if(r2 == 1):
                            score = score+1
                        if(r3 == 1):
                            score = score+1

                        # if there are only two cells and we fill one in, that means the opponent can easily score, so we give a penalty later.
                        if (r1 == 2):
                            penalty = penalty-1
                        if (r2 == 2):
                            penalty = penalty-1
                        if (r3 == 2):
                            penalty = penalty-1

                        # for possible value                    
                        for val in all_k:
                            # if it is a taboo then do nothing with it
                            if(TabooMove(i, j, val) in game_state.taboo_moves):
                                continue

                            # append move to all_moves list
                            all_moves.append(Move(i,j,val))
            
                            # calc rewards
                            # if only ONE region is completed, reward of 1 is given
                            if(score == 1):
                                rewards.append(1)
                            # if only TWO regions are completed, reward of 3 is given
                            elif(score == 2):
                                rewards.append(3)
                            # if all THREE regions completed ...
                            elif(score == 3):
                                rewards.append(7)
                            # NO regions yields a reward of 0
                            elif(score == 0):
                                rewards.append(0)

                            
                            # if the opponent is able to complete one region after our move, give a penalty 
                            if(penalty == -1):
                                rewards[0] -= 2
                            # same for two regions
                            elif(penalty == -2):
                                rewards[0] -= 5
                            # and for three
                            elif(penalty == -3):
                                rewards[0] -= 7
                            

                        # if possible values per pos is at least two
                        if len(all_k) >= 2:
                            # loup through the values
                            for v in all_k:
                                # if not seen yet
                                if Move(i,j,v) not in unsure_moves:
                                    # create the move and add it to unsure_move list
                                    unsure_moves.append(Move(i,j,v))
                    
            return all_moves, unsure_moves, rewards, empty_cells
        
        def get_children(node):
            """
                Get children of the node
                @param Node node: node object
                @return list children: list containing children of the node which are also of type Node
            """
            children = node.child
            return children

        def minimax(node, depth, mdepth, isMaximisingPlayer, alpha, beta) -> Node:   
            """
            Returns the best node found in the tree with alpha beta pruning
            
            @param Node node: a node representing a game state
            @param int depth: keeps track of the depth
            @param int mdepth: max depth to search for
            @param Boolean isMaximisingPlayer: a boolean that is set to True if it is the 
            maximising player and to false if it is the minimizing player
            @param float alpha: holds best alternative found for the maximising player
            @param float beta: holds best alternative found for the minimizing player  
            @return Node best_node: return the node that contains the best move
            """
            # if depth equal mdepth, copy diff value to the attribute value and return the node
            # or if node has no children
            if depth == mdepth and not node.child:
                node.value = node.diff
                return node

            # check for taboo branches
            elif depth != mdepth and len(node.child) == 0:

                # if we are player 1 and nr of empty cells is even (meaning we won't get the last turn),
                # give a taboo move as best move so that we skip our turn
                
                if game_state.current_player() == 1 and node.empty_cells % 2 == 0:
                    if(isMaximisingPlayer):
                        node.value = 10000.0
                    else:
                        node.value = float('-inf')
                    
                    return node
                
                elif game_state.current_player() == 2 and node.empty_cells % 2 == 0:
                    if(isMaximisingPlayer):
                        node.value = 10000.0
                    else:
                        node.value = float('-inf')
                    
                    return node
                
                elif(isMaximisingPlayer):
                    node.value = float('-inf')
                else:
                    node.value = float('inf')

                return node
            
            
            # get children of node
            children = get_children(node)

            # maximising player
            if (isMaximisingPlayer):
                # initialize best_value and best_node
                best_value = float('-inf')
                best_node = new_node()
                # for each child run minimax again with an increased depth and set the boolean to False
                for child in children:
                    # returns best found node
                    value = minimax(child, depth+1, mdepth, False, alpha, beta)
                    # get diff in scores which is stored in the value attribute of the node
                    # if this diff is greater than best_value then we update best_value and best_node
                    diff = value.value
                    if diff > best_value:
                        # update best_value
                        best_value = diff
                        # update parent's value attribute
                        child.value = best_value
                        # update best_node to parent's node
                        best_node = child
                    # pruning
                    if best_value > alpha:
                        alpha = best_value
                    if beta <= alpha:
                        break
                return best_node
            else:
                # minimizing player
                best_value = float('inf')
                best_node = new_node()
                for child in children:
                    value = minimax(child, depth+1, mdepth, True, alpha, beta)
                    diff = value.value
                    # if diff is less than best_value update
                    if diff < best_value:
                        best_value = diff
                        child.value = best_value
                        best_node = child
                    if best_value < beta:
                        beta = best_value
                    # pruning
                    if beta <= alpha:
                        break
                return best_node

        def tree(state, currentnode, cdepth, tdepth, nr=False, rm=False):
            """
                Constructs the game tree, incl. keeping track of the final scores, calculating the difference and
                assigning them to the respective node

                @param GameState state: state of the game
                @param Node currentnode: node object
                @param int cdepth: current depth of the node
                @param int tdepth: total depth of tree aka number of empty cells in the board
                @param int nr = number of unsure_moves that at least need to be present in the unsure_moves list
                to consider unsure moves for exploration
                @param int rm = amount of randomly picked unsure moves
                @return: does not return anything but it creates the tree
            """
            # get legal moves/children, unsure moves and list of rewards of game state
            children, unsure_moves, score, mdepth = get_legal_moves(state)
            currentnode.empty_cells = mdepth
            # if current depth is greater or equal to 0 (boundary check)
            # and current depth is not equal to total depth
            if cdepth != tdepth and cdepth >=0:
                # if there are children
                if children != None:
                    k=0
                    for child in children:
                        # if unsure moves list is not empty and it contains less than x moves given x
                        if unsure_moves != None and len(unsure_moves) < nr:
                            # then only consider moves that are not "unsure"
                            if child not in unsure_moves:
                                # create child node with its respective score and
                                # diff is initialized to 0
                                (currentnode.child).append(Node(child, score[k]))
                        # if unsure moves is not empty and it contains more than x moves given x
                        elif unsure_moves != None and len(unsure_moves) >= nr:
                            # create sure moves list
                            sure_moves = list(set([(c.i, c.j, c.value) for c in children]).difference(set([(c.i, c.j, c.value) for c in unsure_moves])))
                            # choose 5 random unsure moves, and join it with the "sure_moves" list
                            # to allow for some exploration in this area
                            combined = list(set(random.choices([(c.i, c.j, c.value) for c in unsure_moves], k=rm)).union(set(sure_moves)))
                            # if child is in this list
                            if (child.i, child.j, child.value) in combined:
                                (currentnode.child).append(Node(child, score[k]))    
                        # else if there are no unsure moves, proceed normally
                        else:
                            (currentnode.child).append(Node(child, score[k]))
                    
                        # to get next reward
                        k=k+1
                    # increase current depth once children are created
                    cdepth = cdepth + 1
                    for child in currentnode.child:
                        # keep track of final scores per state by retrieving/accumulating the rewards
                        # of the next depth (these are the rewards that you can get when
                        # making a choice and so they are everytime at an odd depth) It
                        # does not matter if we are player 1 or 2, we always start with depth = 0
                        
                        # if cdepth is even
                        if (cdepth % 2 == 0):
                            # then accumulate the reward to the second index 
                            # for the other player
                            # use state.scores list to keep track of the scores
                            state.scores[1] = state.scores[1] + child.reward
                        # if cdepth is odd
                        elif (cdepth % 2 != 0):
                            # then accumulate the reward to the first index
                            # for the other player
                            state.scores[0] = state.scores[0] + child.reward
                        
                        # put the child on the board
                        state.board.put(child.mov.i, child.mov.j, child.mov.value)
                        
                        # recurse, do the same process until no more children are left
                        tree(state, child, cdepth, tdepth)
                        
                        # once done with the recursion/reached a leaf node or 
                        # cdepth = tdepth
                        # undo the effects of accumulating the rewards
                        if (cdepth % 2 == 0):
                            state.scores[1] = state.scores[1] - child.reward
                        elif (cdepth % 2 != 0):
                            state.scores[0] = state.scores[0] - child.reward
                        # undo the move
                        state.board.put(child.mov.i, child.mov.j, 0)
                    # decrease the depth
                    cdepth = cdepth - 1
            # if current depth is total depth or if there are no children
            if cdepth == tdepth or not children:
                # ONLY STORE FINAL SCORE AND DIFF IN THE LEAF NODE
                # append the final scores list with the accumulated scores for player
                # 1 and 2 at index 0 and 1 respectively.

                # if we are player one, store the accumulated scores in this order
                # because first index is for the first player
                # state.scores[0] keeps track of score P1 and we are P1
                # state.scores[1] keeps track of score P2, and so
                # final score will be: [P1, P2] 
                if state.current_player() == 1:
                    currentnode.final_score.append(state.scores[0])
                    currentnode.final_score.append(state.scores[1])
                else: 
                # if we are second, then store it the other way around
                # as the index for player 2 is at index 2
                # e.g. state.scores[1] keeps track of score for P1
                # and state.scores[0] keeps track of score for P2 (us)
                # so final score will now look like: [P1, P2] e.g. [8-10]
                    currentnode.final_score.append(state.scores[1])
                    currentnode.final_score.append(state.scores[0])
                # calc diff and put it in the diff attribute
                currentnode.diff = eval_function(currentnode)

            return mdepth

        ###################################### VARIABLES #######
        # get total/max depth from current game state
        moves, unsure_moves, rewards, mdepth = get_legal_moves(game_state)
        # root representing initial state
        root = new_node()
        # current depth initialization to be used when constructing the tree but also for tree search
        cdepth = 0
        # number of unsure moves that at least need to be present in the unsure moves list
        nr = 2
        # number of elements to randomly pick from the unsure_moves list for exploration
        rm = 2
        ##############################################################

        # construct tree & apply minimax and propose best move depth wise
        # starting from depth 0 uptil total depth - iterative deepening
        for i in range(mdepth+1):
            # create tree up till depth i
            empty_cells = tree(game_state, root, cdepth, i, nr, rm)
            
            # once level/depth is completed apply minimax on it and already propose a move
            # if we start the game, then we are the maximising player
            if game_state.current_player() == 1:
                # search tree from the root to depth i, depthwise search -> iterative deepening
                move = minimax(root, cdepth, i, True, float('-inf'), float('inf')).mov
            # we are the minimizing player aka second player
            else:
                move = minimax(root, cdepth, i, False, float('-inf'), float('inf')).mov
             
            # make move object first
            move_obj = Move(move.i, move.j, move.value)
            # propose the move
            self.propose_move(move_obj)

