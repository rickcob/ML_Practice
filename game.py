# -*- coding: utf-8 -*-
"""
By IBRL
"""
import multiprocessing as mp
import numpy as np
import random
import pandas as pd
import scipy
# import sklearn
import math
import matplotlib.pyplot as plt




N = 200
num_centroids = 300
num_turns = 1000
num_reps = 1000

alpha = 1
beta = 1

# Lambdas
lambdas = {}
lambdas[1] = int(0.009*N)
lambdas[2] = int(0.025*N)
lambdas[3] = int(0.04*N)

money = 1e9

# gains
G = {}
G[1] = (30, 50)
G[2] = (10, 33)
G[3] = (0,12)



p=[0.5, 0.2, 0.3]

cent = np.random.poisson(num_centroids)


# def AsteroidGenerator():
#     X = np.zeros((N, N))
#     for i in range(cent):
#         z = np.random.choice([1, 2, 3], p=p)
#         x, y = np.random.randint(0,N), np.random.randint(0,N)
#         radius = np.random.poisson(lambdas[z])
#         temp = X[x-radius:x+radius, y-radius:y+radius] 
#         gains = np.random.beta(alpha, beta, size = temp.shape)
#         gains = gains*(G[z][1]-G[z][0])+G[z][0]
#         mask = np.random.choice([True, False], size=temp.shape, p=[0.5, 0.5])
#         mask[radius//2:int(3*radius//2), radius//2:int(3*radius//2)] = True
#         X[x-radius:x+radius, y-radius:y+radius] += gains*mask
#     X = money*((X)/np.sum(X))
#     return X







file_path = './strategy.py'

f = open(file_path, 'r').read()
eval(compile(f, file_path, "exec"))

num_cores = mp.cpu_count()
# num_cores = 1


def run_game(j):
    np.random.seed()
    X = np.zeros((N, N))
    for i in range(cent):
        z = np.random.choice([1, 2, 3], p=p)
        x, y = np.random.randint(0,N), np.random.randint(0,N)
        radius = np.random.poisson(lambdas[z])
        temp = X[x-radius:x+radius, y-radius:y+radius] 
        gains = np.random.beta(alpha, beta, size = temp.shape)
        gains = gains*(G[z][1]-G[z][0])+G[z][0]
        mask = np.random.choice([True, False], size=temp.shape, p=[0.5, 0.5])
        mask[radius//2:int(3*radius//2), radius//2:int(3*radius//2)] = True
        X[x-radius:x+radius, y-radius:y+radius] += gains*mask
    X = money*((X)/np.sum(X))
    history = []
    r = 0
    for i in range(num_turns):
        pos =  strategy(history)
        history.append([pos, X[pos[0], pos[1]]])
        r += history[-1][1]
        X[pos[0], pos[1]] = 0
    if j % 100 == 0:
        print('Game {}, Reward {:.3f}'.format(j,r))
    # write csv file
    return r



# R = np.zeros(num_reps)
# for j in range(num_reps):
#     X = AsteroidGenerator()
#     history = []
#     for i in range(num_turns):
#         pos =  strategy(history)
#         history.append([pos, X[pos[0], pos[1]]])
#         R[j] += history[-1][-1]
#         X[pos[0], pos[1]] = 0
#     if j % 100 == 0:
#         print('Game {}, Reward {:.3f}'.format(j,R[j]))


if __name__ == '__main__':
    pool = mp.Pool(processes=num_cores)
    R = []
    pool.map_async(run_game, range(num_reps), callback=R.append)
    pool.close()
    pool.join()
    R = np.asarray(R).flatten()
    print('Mean Reward: ', round(np.mean(R), 3), 'Std: ', round(np.std(R), 3))
    plt.hist(R, bins=10)
    plt.show()

# fig = plt.figure(figsize=(100,100))
# ax = fig.add_subplot(1, 1, 1)

# # Major ticks every 20, minor ticks every 5
# # major_ticks = np.arange(0, n+1, 100)
# # minor_ticks = np.arange(0, n+1, 1)

# # ax.set_xticks(major_ticks)
# # ax.set_xticks(minor_ticks, minor=True)
# # ax.set_yticks(major_ticks)
# # ax.set_yticks(minor_ticks, minor=True)
# ax.imshow(game.X, cmap='plasma')
# # # And a corresponding grid
# # ax.grid(which='both')
# ax.axis('off')
# fig.savefig('asteroid.pdf', bbox_inches='tight')
# plt.show()    