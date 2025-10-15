import numpy as np
import os
import pickle

class Noutput:
    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        a_values_full = data['a_values']
        p_matrix_half = data['p_matrix']
        x_values_half = data['a_values'] @ data['p_matrix']
        print('x_values half')
        print(x_values_half)
        
        x_unique = np.unique(x_values_half)
        x_pos = x_unique[x_unique > 0]
        self.x_values = np.concatenate((-x_pos[::-1], [0], x_pos))

        p_zero_col = p_matrix_half[:, np.isclose(x_values_half, 0)].mean(axis=1, keepdims=True)
        p_pos_cols = p_matrix_half[:, np.isin(x_values_half, x_pos)]
        
        p_neg_cols = np.flip(p_pos_cols, axis=0)
        p_neg_cols = np.fliplr(p_neg_cols)
        
        self.p_matrix = np.hstack((p_neg_cols, p_zero_col, p_pos_cols))
        
        self.a_values = a_values_full
        self.metadata = data['metadata']

    def get_probability_vector(self, x_scaled):
        x_clipped = np.clip(x_scaled, self.x_values[0], self.x_values[-1])
        j = np.searchsorted(self.x_values, x_clipped, side='left')
        j = np.clip(j, 1, len(self.x_values) - 1)

        x_j, x_jm1 = self.x_values[j], self.x_values[j-1]
        p_j, p_jm1 = self.p_matrix[:, j], self.p_matrix[:, j-1]
        
        dx = x_j - x_jm1
        if abs(dx) < 1e-9: return p_j.copy()

        slope = (p_j - p_jm1) / dx
        prob_vector = slope * (x_clipped - x_j) + p_j
        
        prob_vector[prob_vector < 0] = 0
        prob_sum = np.sum(prob_vector)
        if prob_sum > 1e-9:
            prob_vector /= prob_sum
            
        return prob_vector

    def get_probability_vectors_vectorized(self, x_scaled_vector):
        x_scaled_vector = np.atleast_1d(x_scaled_vector)
        x_clipped = np.clip(x_scaled_vector, self.x_values[0], self.x_values[-1])
        j_indices = np.searchsorted(self.x_values, x_clipped, side='left')
        j_indices = np.clip(j_indices, 1, len(self.x_values) - 1)

        x_j = self.x_values[j_indices]
        x_jm1 = self.x_values[j_indices - 1]
        p_j = self.p_matrix[:, j_indices]
        p_jm1 = self.p_matrix[:, j_indices - 1]
        
        dx = x_j - x_jm1

        prob_vectors = np.divide(p_j - p_jm1, dx, out=np.zeros_like(p_j), where=abs(dx) > 1e-9)
        prob_vectors = prob_vectors * (x_clipped - x_j) + p_j

        prob_vectors = prob_vectors.T

        prob_vectors[prob_vectors < 0] = 0
        prob_sums = np.sum(prob_vectors, axis=1, keepdims=True)
        np.divide(prob_vectors, prob_sums, out=prob_vectors, where=prob_sums > 1e-9)
        
        return prob_vectors

    def perturb_batch(self, x_scaled_batch):
        x_scaled_batch = np.asarray(x_scaled_batch)
        batch_size = len(x_scaled_batch)

        prob_matrix = self.get_probability_vectors_vectorized(x_scaled_batch)

        cumulative_probs = np.cumsum(prob_matrix, axis=1)

        random_values = np.random.rand(batch_size, 1)

        perturbed_indices = (random_values < cumulative_probs).argmax(axis=1)

        return self.a_values[perturbed_indices]
