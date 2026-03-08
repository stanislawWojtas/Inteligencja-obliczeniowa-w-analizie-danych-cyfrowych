from easyAI import TwoPlayerGame, AI_Player, Negamax, Human_Player
import random

class TicTacDoh(TwoPlayerGame):
	"""Klasyczna implementacja gry kółko krzyżyk dla silnika easyAI. Logika losowości jest przeniesiona
	do pętli gry. Dzięki temu deterministyczny silnik easyAI radzi sobię z grą."""
	
	def __init__(self, players, first_player=1):
		self.players = players
		self.board = [0 for i in range(9)] #plansza będzie reprezentowana jako 0 - puste, 1 - pierwszy gracz, 2 - drugi gracz
		self.current_player = first_player
	
	def possible_moves(self):
		return [str(idx + 1) for idx, n in enumerate(self.board) if n == 0]
	
	def make_move(self, move):
		self.board[int(move) - 1] = self.current_player

	def unmake_move(self, move):
		self.board[int(move) - 1] = 0

	def is_over(self):
		return self.lose() or self.possible_moves() == []

	def lose(self):
		win_positions = [[1,2,3], [4,5,6], [7,8,9], [1,4,7], [2,5,8], [3,6,9], [1,5,9], [3,5,7]]
		for line in win_positions:
			if all(self.board[c - 1] == self.opponent_index for c in line):
				return True
		return False
	
	def scoring(self):
		if self.lose():
			return -100
		else:
			return 0
		
	def show(self):
		symbols = {0 : '.', 1 : 'O', 2 : 'X'}
		print("\n" + "\n".join(
            [" ".join([symbols[self.board[3 * j + i]] for i in range(3)])
             for j in range(3)]
        ))