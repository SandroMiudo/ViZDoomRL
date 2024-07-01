import tensorflow
import numpy

""""
This Network uses a single Q-Network, to approach the Q-Function
"""

import keras 
from keras import layers
from keras import ops
import tensorflow as tf
from ...env import RL_env


class RLSingleQNetwork(keras.Model): 

    def __init__(self, rl_environment):
        self.rl_enviroment = rl_environment
        x = layers.Input(shape=self.rl_enviroment.get_env_shape())
        pass

    def train_step(self, data):
        return super().train_step(data)
    
    def compile(self, optimizer="rmsprop", loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, jit_compile=None, **kwargs):
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, jit_compile, **kwargs)

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        return super().compute_loss(x, y, y_pred, sample_weight)

    def compute_metrics(self, x, y, y_pred, sample_weight):
        return super().compute_metrics(x, y, y_pred, sample_weight)

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose="auto", callbacks=None, validation_split=0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False):
        return super().fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)

    def evaluate(self, x=None, y=None, batch_size=None, verbose="auto", sample_weight=None, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, return_dict=False, **kwargs):
        return super().evaluate(x, y, batch_size, verbose, sample_weight, steps, callbacks, max_queue_size, workers, use_multiprocessing, return_dict, **kwargs)

    def forward():
        pass

    def back_prop():
        pass

m : keras.Model = RLSingleQNetwork(0, 0)
m.fit()