""""
This Network uses a single Q-Network, to approach the Q-Function
"""

import tensorflow as tf
import keras
from keras import layers
from keras import metrics
from keras import optimizers
import numpy as np
from ..env import RL_Env
import matplotlib.pyplot as plt
from scipy.stats import uniform
import time

class RLSingleQNetwork(keras.Model): 

    def __init__(self, rl_environment, learning_rate):
        super().__init__()
        self.rl_environment = rl_environment
        self.learning_rate = learning_rate
        print("environment shape : ", self.rl_environment.get_env_shape())
        print("buttons supported : ", self.rl_environment.count_buttons_supported())
        print("buttons defined   : ", self.rl_environment.def_buttons())
        # (240, 320, 3)
        self.conv_layer1   = layers.Conv2D(filters=8, kernel_size=(3,3), activation="relu", \
            input_shape=self.rl_environment.get_env_shape())
        self.pool_layer1   = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))
        self.conv_layer2   = layers.Conv2D(16, (3,3), activation="relu")
        self.pool_layer2   = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))
        self.conv_layer3   = layers.Conv2D(32, (3,3), activation="relu")
        self.pool_layer3   = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))
        self.conv_layer4   = layers.Conv2D(64, (3,3), activation="relu")
        self.pool_layer4   = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))
        self.conv_layer5   = layers.Conv2D(128, (3,3), activation="relu")
        self.pool_layer5   = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))
        self.reshape_layer = layers.Reshape((70, 128))
        self.flatten_layer = layers.Flatten()
        self.dense_layer   = layers.Dense(self.rl_environment.count_buttons_supported())

    def call(self, game_state):
        x = self.conv_layer1(game_state)
        x = self.pool_layer1(x)
        x = self.conv_layer2(x)
        x = self.pool_layer2(x)
        x = self.conv_layer3(x)
        x = self.pool_layer3(x)
        x = self.conv_layer4(x)
        x = self.pool_layer4(x)
        x = self.conv_layer5(x)
        x = self.pool_layer5(x)
        x = self.flatten_layer(x)
        x = self.dense_layer(x)
        return x

    def train_step(self, game_state, seed, *args, delta=1, epsilon=0.005):
        q_values = self(game_state)
        q_value_index = tf.argmax(q_values[0])
        rnd_uniform = uniform.rvs()
        if(abs(rnd_uniform - seed) <= epsilon):
            if(len(args) > 0 and args[0] == 'debug'):
                print("epsilon area entered ...")
            tmp = q_value_index
            q_value_index = tf.convert_to_tensor(time.time_ns() % self.rl_environment.count_buttons_supported())
            if(len(args) > 0 and args[0] == "debug"):
                print(f"Actual index : {tmp} -- Exploration index : {q_value_index}")
        actions = tf.one_hot(q_value_index, depth=self.rl_environment.count_buttons_supported(), dtype=tf.int32)
        reward = self.rl_environment.perform_action(actions.numpy())
        updated_game_state = self.rl_environment.get_game_state()
        reshaped_screen_buffer = tf.convert_to_tensor(np.asarray(updated_game_state.screen_buffer) \
            .astype(np.float32).reshape(240, 320, 3))
        reshaped_screen_buffer = tf.expand_dims(reshaped_screen_buffer, axis=0)
        q_star_values = self(reshaped_screen_buffer)
        q_star_value = tf.reduce_max(q_star_values[0])

        with tf.GradientTape() as gradient_tape:
            q_values = self(game_state)  # Forward pass again to get the gradients
            q_value = tf.reduce_max(q_values[0])
            loss = self.compute_loss(q_value, q_star_value, reward, delta)
        
        gradients = gradient_tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return reward, q_value, q_star_value, loss, q_value_index

    def compute_loss(self, q_value, q_star_value, reward, delta):
        # delta ==> 0 = importance converges towards current reward
        # delta ==> 1 = more far-sighted, since next actions are considered important as well
        # Q(s,a) = r + delta * max Q(s*, a*) => Greedy policy
        # Q(s,a) = r + delta *  { if eps : rand Q(s*, a*) if ! eps : max Q(s*, a*)}
        
        # L(theta) = E(s,a,r,s*)~D^A[((r + delta * max Q(s*, a*; theta)) - Q(s, a, theta))^2 ] : batch training
        # L(theta) = ((r + delta * max Q(s*, a*; theta)) - Q(s, a; theta)^2 : single training

        return ((reward + delta * q_star_value) - q_value) ** 2

    def fit(self, *args, epochs=500000):
        l          = []
        q_val      = []
        q_star_val = []
        e     = []
        names = [b.name for b in self.rl_environment.buttons_supported]
        count = np.zeros_like(names, dtype=np.int32)
        rnd_seed = uniform.rvs()
        for epoch in range(epochs):
            game_state = self.rl_environment.get_game_state()
            reshaped_screen_buffer = np.asarray(game_state.screen_buffer).astype(np.float32).reshape(240, 320, 3)
            reshaped_screen_buffer = tf.expand_dims(reshaped_screen_buffer, axis=0)
            reward, q_value, q_star_value, loss, action = self.train_step(
                reshaped_screen_buffer, rnd_seed, *args)
            self.metrics[0].update_state(reward)
            if (args[0] == 'debug'):
                count[action] += 1
                q_val.append(q_value)
                q_star_val.append(q_star_value)
                l.append(loss)
                e.append((epoch+1))
                if (epoch % 10000) == 0:
                    fig, ax = plt.subplots(2, 1)
                    ax[0].set_xlabel("Q-values")
                    ax[1].set_xlabel("Loss")
                    ax[0].scatter(e, q_val)
                    ax[0].scatter(e, q_star_val)
                    ax[1].plot(e, l)
                    plt.show()
                elif (epoch % 1000) == 0:
                    plt.title(f"Actions performed in epoch : {epoch}")       
                    plt.bar(names, count)
                    plt.show()
                
            if (epoch % 100) == 0:
                    print(f"Epoch:{epoch} -- Reward:{self.metrics[0].result()} -- Loss:{loss} "
                        f"-- Q-Val:{q_value} -- Q*-Val:{q_star_value}")
            
        self.rl_environment.close_env()
        
    def evaluate(self, x=None, y=None, batch_size=None, verbose="auto", sample_weight=None, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, return_dict=False, **kwargs):
        return super().evaluate(x, y, batch_size, verbose, sample_weight, steps, callbacks, max_queue_size, workers, use_multiprocessing, return_dict, **kwargs)

rl_env = RL_Env.RL_Env("deadly_corridor.cfg")    
model = RLSingleQNetwork(rl_env, 0.05)
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), 
              metrics=[metrics.Mean()])
model.fit("debug")

model.save("../model.keras")