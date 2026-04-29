from functools import reduce
factorial = lambda n: 1 if n == 0 else reduce(lambda a, b: a * b, [i for i in range(1, n+1)], 1)
