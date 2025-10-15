import numpy as np
import time

class DE():
    def __init__(self, d, eps, bins, rng):
        self.rng = rng
        self.d = d
        self.bins = bins
        self.eps = eps

        # precomputed
        self.ee = np.exp(self.eps)
        self.p = self.ee / (self.ee + self.bins -1)
        self.q = 1 / (self.ee + self.bins - 1)

    def batch(self, data):
        # data is indices of histogram
        data = data.reshape(-1)
        noisy_output = np.zeros_like(data)

        p_random = self.rng.random(len(data))
        p_inds = np.argwhere(p_random < (self.p - self.q)).reshape(-1)
        noisy_output[p_inds] = data[p_inds]
        q_inds = np.argwhere(p_random >= (self.p - self.q)).reshape(-1)
        uniform_random = self.rng.choice([i for i in range(self.bins)], len(q_inds))
        noisy_output[q_inds] = uniform_random

        # noisy output is histogram
        # hist = np.unique(noisy_output, return_counts=True)[1]
        hist = np.bincount(noisy_output, minlength=self.bins)
        est_hist = (hist / len(data) -  self.q) / (self.p - self.q)

        return est_hist