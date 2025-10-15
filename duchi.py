import numpy as np
import random
import itertools
import math
from math import comb

class Duchi():
    def __init__(self, d, eps, rng, range=1.0, k=1):
        self.d = d
        self.eps = eps
        self.rng = rng
        self.range = range
        self.k = k

        self.eps_k = self.eps / self.k
        
        # precomputed
        self.ee = np.exp(self.eps_k)
        self.p = (self.ee-1) / (2*self.ee + 2)
        self.A = (self.ee + 1) / (self.ee - 1)
    
    def Duchi_batch(self, data):
        data = data.reshape(-1) / self.range
        noisy_output = np.zeros_like(data)

        P1 = data * self.p + 0.5

        u = self.rng.random(len(data))
        minus_idx = np.argwhere(u >= P1).reshape(-1)
        plus_idx = np.argwhere(u < P1).reshape(-1)

        noisy_output[minus_idx] = -self.A
        noisy_output[plus_idx] = self.A

        return noisy_output  * self.range