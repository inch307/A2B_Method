import numpy as np
import math
import duchi

class PM():
    def __init__(self, d, eps, rng, range=1.0, k=1):
        self.d = d
        self.eps = eps
        self.k = k
        self.eps_k = self.eps / self.k

        self.A = (np.exp(eps/2/self.k) + 1) / (np.exp(eps/2/self.k) - 1)
        self.rng = rng
        self.range = range
    
    def PM_batch(self, data):
        data = data.reshape(-1) / self.range
        noisy_output = np.zeros_like(data)

        u = self.rng.random(len(noisy_output))

        l = data * (self.A + 1) / 2 - (self.A - 1) / 2
        r = l + self.A - 1

        inner_idx = np.argwhere(u < np.exp(self.eps_k/2) / (np.exp(self.eps_k/2)+1)).reshape(-1)
        outer_idx = np.argwhere(u >= np.exp(self.eps_k/2) / (np.exp(self.eps_k/2)+1)).reshape(-1)

        inner_y = self.rng.random(len(inner_idx))
        # print(inner_y.shape)
        # print((r[inner_idx]-l[inner_idx]).shape)
        inner_y = (r[inner_idx]-l[inner_idx])*inner_y + l[inner_idx]
        # print(inner_y.shape)
        noisy_output[inner_idx] = inner_y

        length_l = np.abs(l[outer_idx] + self.A)
        legnth_r = np.abs(self.A - r[outer_idx])
        interval_l = length_l / (length_l + legnth_r)
        interval_random = self.rng.random(len(outer_idx))
        left_idx = outer_idx[interval_random < interval_l]
        right_idx = outer_idx[interval_random >= interval_l]
        
        left_y = self.rng.random(len(left_idx))
        left_y = (l[left_idx] + self.A) * left_y - self.A
        noisy_output[left_idx] = left_y

        right_y = self.rng.random(len(right_idx))
        right_y = (self.A - r[right_idx]) * right_y + r[right_idx]
        noisy_output[right_idx] = right_y

        return noisy_output * self.range