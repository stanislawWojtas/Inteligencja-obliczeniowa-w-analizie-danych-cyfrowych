import time
from functools import wraps
def calc_function_time(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		start_time = time.time()
		result = func(*args, **kwargs)
		end_time = time.time()

		print("Czas wykonania funkcji {}: {:.4f} sekund".format(func.__name__, end_time - start_time))
		if 'n' in kwargs:
			print("Średni czas na grę wyniósł {:.4f} sekund".format((end_time - start_time) / kwargs['n']))
		return result
	return wrapper