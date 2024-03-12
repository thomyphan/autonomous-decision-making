import random
import numpy as np
import math

def random_bandit(Q_values, action_counts):
    return random.choice(range(len(Q_values)))
    
def epsilon_greedy(Q_values, action_counts, epsilon=0.1):
    if np.random.rand() <= epsilon:
        return random.choice(range(len(Q_values)))
    else:
        return np.argmax(Q_values)
        
def boltzmann(Q_values, action_counts, temperature=1.0):
    E = np.exp(Q_values/temperature)
    return np.random.choice(range(len(Q_values)), p=E/sum(E))
        
def UCB1(Q_values, action_counts, exploration_constant=1):
    UCB1_values = []
    N_total = sum(action_counts)
    for Q, N in zip(Q_values, action_counts):
        if N == 0:
            UCB1_values.append(math.inf)
        else:
            exploration_term = exploration_constant
            exploration_term *= np.sqrt(np.log(N_total)/N)
            UCB1_values.append(Q + exploration_term)
    return np.argmax(UCB1_values)

