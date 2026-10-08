""" MA3.py

Student:
Mail:
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
import numpy as np
from numba import njit

# Exc1
def approximate_pi(n):
    n_c = 0
    inside_x = []
    inside_y = []
    outside_x = []
    outside_y = []
    for x in range(n):
        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        if x**2+y**2 < 1:
            n_c+=1
            inside_x.append(x)
            inside_y.append(y)
        else:
            outside_x.append(x)
            outside_y.append(y)
        
    pi = 4*n_c/n
    print(f'Antal punkter: {n}')
    print(f'Approximation av pi: {pi}')    
    plt.figure(figsize=(6,6))
    plt.scatter(inside_x,inside_y,color='red')
    plt.scatter(outside_x,outside_y,color='blue')
    plt.axis("equal")
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    if n == 1000:
        plt.savefig("pi_1000.png")
    elif n == 10000:
        plt.savefig("pi_10000.png")
    elif n == 100000:
        plt.savefig("pi_100000.png")
    plt.show()
    return pi

# Exc2, approximation
def sphere_volume(n, d): 
    
    x_i = np.array([np.array([random.uniform(-1,1) for nums in range(d)]) for nums in range(n)])
    points_in_sphere = list(filter(lambda coord: np.sum(coord**2)<1,x_i))
    hypercube_vol = 2**d
    ratio = len(points_in_sphere)/n
    volume = ratio*hypercube_vol

    return volume

#Exc2, real value
def hypersphere_exact(n,d):
    vol = (m.pi**(d/2))/m.gamma(d/2+1)
    return vol

#Exc3: numba version
@njit
def sphere_volume_numba(n:int, d:int)->float:
    x_i = np.array([np.array([random.uniform(-1,1) for nums in range(d)]) for nums in range(n)])
    points_in_sphere = list(filter(lambda coord: np.sum(coord**2)<1,x_i))
    hypercube_vol = 2**d
    ratio = len(points_in_sphere)/n
    volume = ratio*hypercube_vol
    
    return volume

#Exc4: parallel code - parallelize actual computations by splitting data
def count_points(args):
    n, d = args
    points_inside_sphere = 0

    for _ in range(n):
        point = [random.uniform(-1, 1) for _ in range(d)]

        if sum(x**2 for x in point) < 1:
            points_inside_sphere += 1

    return points_inside_sphere


def sphere_volume_parallel(n, d, np=10):
    points_per_process = n // np
    remaining_points = n % np

    work_items = []

    for process_id in range(np):
        points_in_process = points_per_process

        if process_id < remaining_points:
            points_in_process += 1

        work_items.append((points_in_process, d))

    with future.ProcessPoolExecutor(max_workers=np) as ex:
        points_per_process = ex.map(count_points, work_items)

    points_inside_sphere = sum(points_per_process)

    hypercube_volume = 2**d
    ratio = points_inside_sphere / n
    volume = ratio * hypercube_volume

    return volume
    
def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    n = 100000
    d = 11
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    # Exc3
    n = 1000000
    d = 11
    for i in range(3):
        start = pc()
        sphere_volume(n, d)
        stop = pc()
        print(f"Exc3: Run {i}, Sequential time of {d} and {n}: {stop-start}")
        start = pc()
        sphere_volume_numba(n,d)
        stop = pc()
        print(f"Exc3:Run {i}, Numba time of {d} and {n}: {stop-start}")

    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")
    start = pc()
    sphere_volume_parallel(n,d)
    stop = pc()
    print(f"Exc4: Parallel time of {d} and {n}: {stop-start}")

    
    

if __name__ == '__main__':
	main()

