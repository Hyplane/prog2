
from time import perf_counter as pc

from MA3 import sphere_volume_parallel

n = 1000000
d = 11

if __name__ == '__main__':
    start = pc()

    sphere_volume_parallel(n, d)

    stop = pc()

    print(f'The parallel time is: {stop - start}')


