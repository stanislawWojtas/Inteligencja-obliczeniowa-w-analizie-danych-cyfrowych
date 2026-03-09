import random
import time
from game import TicTacDoh
from easyAI import AI_Player, Negamax
from custom_ai import ExpectiMinimax
from time_decorator import calc_function_time

FAIL_PROBABILITY = 0.2

def simulate_game(game: TicTacDoh, timing_stats: dict, is_probabilistic=True, fail_probability=FAIL_PROBABILITY):
	"""Przeprowadza pojedynczą grę między dwoma AI.
	Zwraca 0 jeżeli remis, 1 jeżeli wygrał pierwszy gracz, 2 jeżeli wygrał drugi gracz."""

	while not game.is_over():
		player_idx = game.current_player
		t_start = time.perf_counter()
		move = game.players[player_idx - 1].ask_move(game)
		t_elapsed = time.perf_counter() - t_start
		timing_stats[player_idx]['total'] += t_elapsed
		timing_stats[player_idx]['count'] += 1

		chance = random.random()
		if not is_probabilistic or chance >= fail_probability:
			game.make_move(move)
		game.switch_player()

	if game.lose():
		return game.opponent_index
	else:
		return 0

@calc_function_time
def run_expecti_tournament(n=10, max_depth=5, is_probabilistic=True):
	print("=== Turniej: ExpectiMinimax vs Negamax (easyAI) ===" \
	f"\nLiczba gier: {n}" \
	f"\nMax głębokość: {max_depth}")
	if is_probabilistic:
		print(f"WERSJA PROBABILISTYCZNA (p_fail={FAIL_PROBABILITY})")
	else:
		print("WERSJA DETERMINISTYCZNA")

	counter = {0: 0, 1: 0, 2: 0}

	ai1 = ExpectiMinimax(max_depth, fail_probability=FAIL_PROBABILITY)
	ai2 = Negamax(max_depth)

	ai_player_O = AI_Player(ai1)
	ai_player_X = AI_Player(ai2)

	players = [ai_player_O, ai_player_X]

	ai_names = {
		1: f"ExpectiMinimax(depth={max_depth}, p_fail={FAIL_PROBABILITY})",
		2: f"Negamax easyAI(depth={max_depth})",
	}
	timing_stats = {
		1: {'total': 0.0, 'count': 0},
		2: {'total': 0.0, 'count': 0},
	}

	for i in range(n):
		if i % 2 != 0:
			game = TicTacDoh(players=players, first_player=2)
		else:
			game = TicTacDoh(players=players, first_player=1)
		winner = simulate_game(game, timing_stats, is_probabilistic=is_probabilistic)
		counter[winner] += 1

	print("\n=== Wyniki turnieju ===")
	print(f"Remisy: {counter[0]} ({counter[0]/n*100:.2f}%)")
	print(f"Zwycięstwa gracza O (AI1): {counter[1]} ({counter[1]/n*100:.2f}%)")
	print(f"Zwycięstwa gracza X (AI2): {counter[2]} ({counter[2]/n*100:.2f}%)")

	print("\n=== Średni czas wyboru ruchu ===")
	for player_idx, stats in timing_stats.items():
		if stats['count'] > 0:
			avg_ms = stats['total'] / stats['count'] * 1000
			print(f"Gracz {player_idx} ({ai_names[player_idx]}): {avg_ms:.3f} ms/ruch  (łącznie {stats['count']} ruchów)")


if __name__ == "__main__":
	run_expecti_tournament(n=100, max_depth=7, is_probabilistic=True)
