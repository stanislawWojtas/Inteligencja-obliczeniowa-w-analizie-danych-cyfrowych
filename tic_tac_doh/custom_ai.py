
from game import TicTacDoh


class NegamaxWithoutAB:
	"""Własna implementacja algorytmu Negamax bez odcięcia alfa-beta, dla porównania z implementacją z easyAI."""

	def __init__(self, depth):
		self.depth = depth
	
	def __call__(self, game: TicTacDoh):
		self.negamax_no_ab(game, self.depth, self.depth)
		return game.ai_move

	def negamax_no_ab(self, game: TicTacDoh, depth: int, original_depth: int):
		if depth == 0 or game.is_over():
			return game.scoring() * (1 + 0.001 * depth) # Dodawany jest mały bonus za szybsze zwycięstwo
		possible_moves = game.possible_moves()
		state = game
		unmake_move = hasattr(state, "unmake_move")  # Sprawdzenie, czy gra obsługuje cofanie ruchów

		best_score = -float('inf')
		best_move = possible_moves[0]

		for move in possible_moves:
			if not unmake_move:
				game = state.copy()
			
			game.make_move(move)
			game.switch_player()

			move_value = -self.negamax_no_ab(game, depth - 1, original_depth)

			# cofnięcie ruchu
			if unmake_move:
				game.switch_player()
				game.unmake_move(move)

			if move_value > best_score:
				best_score = move_value
				best_move = move
		
		if depth == original_depth:
			state.ai_move = best_move
		return best_score
	
class ExpectiMinimax:
	def __init__(self, depth, fail_probability=0.2):
		self.depth = depth
		self.fail_probability = fail_probability
		self.success_probability = 1.0 - fail_probability
	
	def __call__(self, game: TicTacDoh):
		self.expecti_negamax(game, self.depth, -float('inf'), float('inf'), self.depth)
		return game.ai_move
	
	def expecti_negamax(self, game:TicTacDoh, depth, alpha, beta, origDepth):
		if depth == 0 or game.is_over():
			return game.scoring() * (1 + 0.001 * depth)
		
		possible_moves = game.possible_moves()
		stake = game
		unmake_move = hasattr(stake, "unmake_move")

		game.switch_player()
		v_fail = -self.expecti_negamax(game, depth - 1, -float('inf'), float('inf'), origDepth)
		game.switch_player()  # przywrócenie tury aktualnego gracza przed pętlą ruchów

		best_value = -float('inf')
		best_move =	 possible_moves[0]

		for move in possible_moves:
			if not unmake_move:
				game = stake.copy()
			
			game.make_move(move)
			game.switch_player()

			alpha_succ = (alpha - self.fail_probability * v_fail) / self.success_probability
			beta_succ = (beta - self.fail_probability * v_fail) / self.success_probability

			val_opp = self.expecti_negamax(game, depth - 1, -beta_succ, -alpha_succ, origDepth)
			v_succ = -val_opp

			if unmake_move:
				game.switch_player()
				game.unmake_move(move)
			
			#Wartość oczekiwana
			ev = self.success_probability * v_succ + self.fail_probability * v_fail

			if ev > best_value:
				best_value = ev
				best_move = move
			
			# Klasyczne odcięcie alfa-beta
			alpha = max(alpha, ev)
			if alpha >= beta:
				break
		
		if depth == origDepth:
			stake.ai_move = best_move
		
		return best_value