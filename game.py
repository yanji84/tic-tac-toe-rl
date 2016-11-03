import random
import sys
import copy

class Game:
    def __init__(self):
        self.board = [-1.0] * 9
        self.winning_combos = (
        [6, 7, 8], [3, 4, 5], [0, 1, 2], [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6],)
        self.corners = [0,2,6,8]
        self.sides = [1,3,5,7]
        self.middle = 4
        self.player_marker, self.bot_marker = self.get_marker()

    def get_marker(self):
        return (1.0,0.0)

    def reset(self):
        self.board = [-1.0] * 9
        return self.board

    def step(self, action):
      over = False
      reward = 0

      # penalty for cheating
      if not self.is_space_free(self.board, action):
        return None, -10, True

      self.make_move(self.board, action, self.player_marker)
      # winning
      if(self.is_winner(self.board, self.player_marker)):
        reward = 100
        over = True
      # drawing
      elif self.is_board_full():
        reward = 10
        over = True
      else:
        # opponent makes move
        bot_move = self.get_bot_move()
        self.make_move(self.board, bot_move, self.bot_marker)
        if (self.is_winner(self.board, self.bot_marker)):
          # losing
          reward = -1
          over = True
        elif self.is_board_full():
          # drawing again
          reward = 10
          over = True

      return self.board, reward, over

    def is_winner(self, board, marker):
        for combo in self.winning_combos:
            if (board[combo[0]] == board[combo[1]] == board[combo[2]] == marker):
                return True
        return False

    def get_bot_move(self):
        # check if bot can win in the next move
        for i in range(0,len(self.board)):
            board_copy = copy.deepcopy(self.board)
            if self.is_space_free(board_copy, i):
                self.make_move(board_copy,i,self.bot_marker)
                if self.is_winner(board_copy, self.bot_marker):
                    return i

        for i in range(0,len(self.board)):
            board_copy = copy.deepcopy(self.board)
            if self.is_space_free(board_copy, i):
                self.make_move(board_copy,i,self.player_marker)
                if self.is_winner(board_copy, self.player_marker):
                    return i

        # check for space in the corners, and take it
        move = self.choose_random_move(self.corners)
        if move != None:
            return move

        # If the middle is free, take it
        if self.is_space_free(self.board,self.middle):
            return self.middle

        # else, take one free space on the sides
        return self.choose_random_move(range(9))

    def is_space_free(self, board, index):
        "checks for free space of the board"
        return board[index] == -1.0

    def is_board_full(self):
        "checks if the board is full"
        for i in range(1,9):
            if self.is_space_free(self.board, i):
                return False
        return True

    def make_move(self,board,index,move):
        board[index] =  move

    def choose_random_move(self, move_list):
        possible_winning_moves = []
        for index in move_list:
            if self.is_space_free(self.board, index):
                possible_winning_moves.append(index)
        if len(possible_winning_moves) != 0:
            return random.choice(possible_winning_moves)
        else:
            return None