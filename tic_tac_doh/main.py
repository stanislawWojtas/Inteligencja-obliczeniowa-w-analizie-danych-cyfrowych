from game import TicTacDoh
from easyAI import Negamax, AI_Player, Human_Player
import random

def play_game(game: TicTacDoh, fail_probability=0.2):
	print("\nNumeracja pól:" \
	"\n1 2 3" \
	"\n4 5 6" \
	"\n7 8 9\n")

	game.show()
	while not game.is_over():
		current_symbol = 'O' if game.current_player == 1 else 'X'
		# pobieramy obiekt gracza żeby sprawdzić czy to AI czy gracz normalny
		current_player_obj = game.players[game.current_player - 1]
		if(isinstance(current_player_obj, AI_Player)):
			move = current_player_obj.ask_move(game)
			print(f"\nAI ({current_symbol}) wybiera pole: {move}")
		else:
			while True:
				move = input(f"\nGracz {current_symbol}, wybierz pole (1-9): ").strip()
				if move in game.possible_moves():
					break
				print(f"Nieprawidłowy ruch! Dostępne pola to: {', '.join(game.possible_moves())}")
		
		# Logika losowości
		chance = random.random()
		if chance < fail_probability:
			print(f">>> PECH! Ruch gracza {current_symbol} na pole {move} się nie udał. Strata kolejki <<<")
		else:
			game.make_move(move)
			game.show()
		game.switch_player()
	if game.lose():
		winner_symbol = 'X' if game.current_player == 1 else 'O'
		print(f"\nWygrał gracz {winner_symbol}!!!")
	else:
		print("\nRemis!")


if __name__ == "__main__":
	print('=== Gra w Kółko i Krzyżyk ===')
	print("1. Gracz vs Gracz")
	print("2. Gracz vs AI")
	print("3. AI vs AI")
	
	while True:
		choice = input("Wybierz tryb (1-3): ").strip()

		ai = Negamax(9)
		if choice == "1":
			players = [Human_Player(), Human_Player()]
		elif choice == "3":
			players = [AI_Player(ai), AI_Player(ai)]
		else:
			players = [Human_Player(), AI_Player(ai)]
		game = TicTacDoh(players)
		play_game(game)