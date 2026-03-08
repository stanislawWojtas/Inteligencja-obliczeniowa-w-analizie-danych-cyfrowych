import random
from game import TicTacDoh
from easyAI import AI_Player, Negamax


def simulate_game(game: TicTacDoh, is_probabilistic=True, fail_probability=0.2):
	"""Funckja przeprowadzająca pojedynczą grę między dwoma AI.
	  Nie wyświetla aktualnego stanu planszy a jedynie zwraca zwycięzce.
	  Może działać w trybie probabilistycznym, gdzie istnieje szansa że ruch się nie powiedzie.
	  Zwraca 0 jeżeli remis, 1 jeżeli wygrał pierwszy gracz, 2 jeżeli wygrał drugi gracz."""
	
	while not game.is_over():
		move = game.players[game.current_player - 1].ask_move(game)
		chance = random.random()
		if not is_probabilistic or chance >= fail_probability:
			game.make_move(move)
		game.switch_player()
	if game.lose():
		return game.opponent_index
	else:
		return 0

def run_tournament(n=10, max_depth=5, is_probabilistic=True):
	print("=== Rozpoczęcie turnieju AI vs AI ===" \
	f"\nLiczba gier: {n}" \
	f"\nMax głębokość Negamax AI: {max_depth}")
	if is_probabilistic:
		print("WERSJA PROBABILISTYCZNA")
	else:
		print("WERSJA DETERMINISTYCZNA")

	counter = {0: 0, 1: 0, 2: 0} # licznik wyników: remis, zwycięstwo gracza 1, zwycięstwo gracza 2}

	ai1 = Negamax(max_depth)
	ai2 = Negamax(max_depth)


	ai_player_O = AI_Player(ai1)
	ai_player_X = AI_Player(ai2)

	players = [ai_player_O, ai_player_X]

	for i in range(n):
		if i%2 != 0:
			game = TicTacDoh(players=players, first_player=2)
		else:
			game = TicTacDoh(players=players, first_player=1)
		winner = simulate_game(game, is_probabilistic=is_probabilistic)
		counter[winner] += 1

	print("\n=== Wyniki turnieju ===")
	print(f"Remisy: {counter[0]} ({counter[0]/n*100:.2f}%)")
	print(f"Zwycięstwa gracza O (AI1): {counter[1]} ({counter[1]/n*100:.2f}%)")
	print(f"Zwycięstwa gracza X (AI2): {counter[2]} ({counter[2]/n*100:.2f}%)")


if __name__ == "__main__":
	run_tournament(n=10, max_depth=6, is_probabilistic=True)