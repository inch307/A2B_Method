import numpy as np
from scipy.stats import norm

INCREASE = 1
DECREASE = -1

class BoundTuning:
    def __init__(self, eps, l=-1., r=1., alpha=0.01, lr=0.5, zeta=0.3, tau=2):
        self.eps = eps
        self.lr = lr

        self.l = l
        self.r = r

        self.alpha = alpha
        self.zeta = zeta
        self.tau = tau

        self.r_direction = None
        self.l_direction = None

    def fit(self, theta_hat_minus, theta_hat_plus, theta_hat_0):
        theta_hat_0 = max(theta_hat_0, self.zeta) 
        
        error_l_linear = theta_hat_minus - self.alpha
        error_l = np.sign(error_l_linear) * np.power(np.abs(error_l_linear), 1/self.tau)
        
        delta_l = -self.lr * error_l * (self.r - self.l) / theta_hat_0
        
        error_r_linear = theta_hat_plus - self.alpha
        error_r = np.sign(error_r_linear) * np.power(np.abs(error_r_linear), 1/self.tau)
        
        delta_r = self.lr * error_r * (self.r - self.l) / theta_hat_0
        
        self.l = self.l + delta_l
        self.r = self.r + delta_r

        return
    
    def transform(self, x):        
        x_clipped = np.clip(x, self.l, self.r)

        clipped_lower = x < self.l
        clipped_upper = x > self.r
        no_clip = (~clipped_lower) & (~clipped_upper)

        indices_clipped_lower = np.where(clipped_lower)[0]
        indices_clipped_upper = np.where(clipped_upper)[0]
        indices_no_clip = np.where(no_clip)[0]

        indices_no_clip = np.where(no_clip)[0]

        y = (2 * x_clipped - (self.l + self.r)) / (self.r - self.l)
        return y, indices_clipped_lower, indices_clipped_upper, indices_no_clip
    
    def inverse_transform(self, y):
        return (y * (self.r - self.l) + (self.l + self.r)) / 2