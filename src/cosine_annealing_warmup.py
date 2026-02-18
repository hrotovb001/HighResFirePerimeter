import keras
from keras.callbacks import Callback
import numpy as np

class CosineAnnealingWarmRestarts(Callback):
    def __init__(self, T_0, T_mult=1, eta_max=0.01, eta_min=0, verbose=0):
        super().__init__()
        self.T_0 = T_0  # Number of epochs for first restart period
        self.T_mult = T_mult  # Multiplicative factor for increasing T_i
        self.eta_max = eta_max  # Maximum learning rate
        self.eta_min = eta_min  # Minimum learning rate
        self.T_i = T_0  # Current restart period length
        self.last_restart = 0
        self.verbose = verbose
        
    def on_epoch_begin(self, epoch, logs=None):
        # Calculate learning rate using cosine annealing with warm restarts
        if self.T_mult == 1:
            # Simple cyclic decay (no increase in cycle length)
            T_cur = epoch % self.T_0
            T_i = self.T_0
        else:
            # With increasing cycle length (SGDR)
            if epoch >= self.last_restart + self.T_i:
                self.last_restart = epoch
                self.T_i = int(self.T_i * self.T_mult)
            T_cur = epoch - self.last_restart
            T_i = self.T_i
        
        # Cosine annealing formula
        lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (
            1 + np.cos(np.pi * T_cur / T_i)
        )
        
        self.model.optimizer.learning_rate.assign(lr)
        
        if self.verbose:
            print(f'Epoch {epoch}: setting learning rate to {lr:.6f}')

