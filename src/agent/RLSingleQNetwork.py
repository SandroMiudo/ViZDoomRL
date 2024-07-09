""""
This Network uses a single Q-Network, to approach the Q-Function
"""

import tensorflow as tf
import keras
from keras import layers
from keras import metrics
from keras import optimizers
from keras import losses
from keras import callbacks
import numpy as np
from ..env import RL_Env
import matplotlib.pyplot as plt
from scipy.stats import uniform
import time
import argparse
import os

class DebugCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        hist_ :DebugHistory = logs["hist"]
        debug_history = hist_.history_of(0, epoch, 'all')

        if (epoch % 1000) == 0:
            e = np.arange(1, epoch+1)
            _, ax = plt.subplots(2, 1)
            ax[0].set_xlabel("Q-values")
            ax[1].set_xlabel("Loss")
            ax[0].scatter(e, debug_history['q_value'])
            ax[0].scatter(e, debug_history['q*_value'])
            ax[1].plot(e, debug_history['loss'])
            plt.show()
    
    def on_train_begin(self, logs=None):
        tmp           = logs["a_ind"]
        q_value_index = logs["exp_ind"]
        print("Epsilon area entered ...")
        print(f"Actual index : {tmp} -- Exploration index : {q_value_index}")

class TrainCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        reward_metric = logs["metric"]
        q_star_value  = logs["q*_value"]
        q_value       = logs["q_value"]
        loss          = logs["loss"]
        if (epoch % 100) == 0:
            print(f"Epoch:{epoch} -- Reward:{reward_metric.result()} -- Loss:{loss} "
                f"-- Q-Val:{q_value} -- Q*-Val:{q_star_value}")

class DebugHistory():
    
    def __init__(self, ) -> None:
        self.global_loss           = []
        self.global_q_values       = []
        self.global_q_star_values  = []

    def update_history(self, entry : tuple[int, int, int, int]):
        self.global_q_values.append(entry[0])
        self.global_q_star_values.append(entry[1])
        self.global_loss.append(entry[2])

    """
    args : 'all', 'loss', 'q_value', 'q*_value'
    """
    def history_of(self, epoch_from, epoch_to, *args):
        hist_ = 0
        if ('all' and 'loss' and 'q_value' and 'q*_value') not in args:
            return None
        
        if('all' in args):
            hist_ = {'loss' : self.global_loss[epoch_from:epoch_to],
                     'q_value' : self.global_q_values[epoch_from:epoch_to],
                     'q*_value' : self.global_q_star_values[epoch_from:epoch_to]}
        elif('loss' in args):
            hist_ = self.global_loss[epoch_from:epoch_to]
        elif('q_value' in args):
            hist_ = self.global_q_values[epoch_from:epoch_to]
        elif('q*_value' in args):       
            hist_ = self.global_q_star_values[epoch_from:epoch_to]

        return hist_

class RLSingleQNetwork(keras.Model): 

    def __init__(self, rl_environment, fixed_reduce_value=100):
        super().__init__()
        self.rl_environment = rl_environment
        self.fixed_reduce_value = fixed_reduce_value
        self.handling_callbacks = callbacks.CallbackList()
        
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
        self.conv_layer6   = layers.Conv2D(256, (3,3), activation="relu")
        self.pool_layer6   = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))
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
        x = self.conv_layer6(x)
        x = self.pool_layer6(x)
        x = self.flatten_layer(x)
        x = self.dense_layer(x)
        return x

    def reduce_over_time(self, epoch, epsilon, epsilon_update, epsilon_l_b, reduce_update):
        limit_reached = False
        if((epoch % reduce_update == 0) and (epsilon > epsilon_l_b)):
            if(epsilon - epsilon_update < epsilon_l_b):
                epsilon = epsilon_l_b
                limit_reached = True
            else: 
                epsilon -= epsilon_update

        return epsilon, limit_reached

    def reduce_fixed(self, epoch, epsilon, epsilon_update, epsilon_l_b, reduce_update):
        limit_reached = False
        if((epoch < reduce_update and (epsilon > epsilon_l_b))):
            epsilon, limit_reached = self.reduce_over_time(epoch,  epsilon, epsilon_update, epsilon_l_b, self.fixed_reduce_value)
        else:
            epsilon = epsilon_l_b
            limit_reached = True
        
        return epsilon, limit_reached

    def train_step(self, game_state, seed, delta, epsilon, *args):
        q_values = self(game_state)
        q_value_index = tf.argmax(q_values[0])
        rnd_uniform = uniform.rvs()
        if(abs(rnd_uniform - seed) <= epsilon):    
            tmp = q_value_index
            q_value_index = tf.convert_to_tensor(time.time_ns() % self.rl_environment.count_buttons_supported())
            self.handling_callbacks.on_train_begin({"a_ind":tmp, "exp_ind":q_value_index})
        actions = tf.one_hot(q_value_index, depth=self.rl_environment.count_buttons_supported(), dtype=tf.int32)
        reward = self.rl_environment.perform_action(actions.numpy())
        updated_game_state = self.rl_environment.get_game_state()
        reshaped_screen_buffer = tf.convert_to_tensor(np.asarray(updated_game_state.screen_buffer) \
            .astype(np.float32).reshape(self.rl_environment.get_env_shape()))
        reshaped_screen_buffer = tf.expand_dims(reshaped_screen_buffer, axis=0)
        q_star_values = self(reshaped_screen_buffer)
        q_star_value = tf.reduce_max(q_star_values[0])

        with tf.GradientTape() as gradient_tape:
            q_values = self(game_state)
            q_value = tf.constant(0.0)
            if(abs(rnd_uniform - seed) <= epsilon):
                q_value = q_values[0, (time.time_ns() % self.rl_environment.count_buttons_supported())]
            else: 
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

        return self.loss(tf.convert_to_tensor([q_value]), tf.convert_to_tensor([reward + delta * q_star_value]))

    def to_data_buffer(self, game_state):
        a = np.asarray(game_state.screen_buffer) \
            .astype(np.float32).reshape(self.rl_environment.get_env_shape())
        return tf.expand_dims(a, axis=0)

    def register_callback(self, callback: callbacks.Callback):
        self.handling_callbacks.append(callback)

    """
    reduce_fctn : over_time or fixed
    reduce_update : over_time -> update value , fixed -> fixed value to stop
    """
    def fit(self, *args, epochs=100000, delta=0.7, epsilon=0.1, epsilon_update=0.001, 
            epsilon_l_b=0.0001, reduce_update=100, reduce_fctn="over_time"):
        rnd_seed = uniform.rvs()
        reduce_function = None
        limit_reached = False
        history = None
        if reduce_fctn == "over_time":
            reduce_function = self.reduce_over_time
        elif reduce_fctn == "fixed":
            reduce_function = self.reduce_fixed

        if "debug" in args:
            self.register_callback(DebugCallback())
            history = DebugHistory()

        self.register_callback(TrainCallback())

        for epoch in range(epochs):
            game_state = self.rl_environment.get_game_state()
            reshaped_screen_buffer = self.to_data_buffer(game_state)
            reward, q_value, q_star_value, loss, action = self.train_step(
                reshaped_screen_buffer, rnd_seed, delta, epsilon, *args)
            self.metrics[0].update_state(reward)
            if(not limit_reached):
                epsilon, limit_reached = reduce_function(epoch, epsilon, epsilon_update, 
                    epsilon_l_b, reduce_update)
            if(history != None):
                history.update_history((q_value, q_star_value, loss))
                
            self.handling_callbacks.on_epoch_end(epoch+1, {"hist":history, "metric":self.metrics[0],
                "loss":loss, "q*_value":q_star_value, "q_value":q_value})
        self.rl_environment.close_env()
        
    def evaluate(self, x=None, y=None, batch_size=None, verbose="auto", sample_weight=None, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, return_dict=False, **kwargs):
        return super().evaluate(x, y, batch_size, verbose, sample_weight, steps, callbacks, max_queue_size, workers, use_multiprocessing, return_dict, **kwargs)




def train_model_routine(debug_act: bool, config : str, res : tuple[int, int],
    dir : str, file: str):
    rl_env = RL_Env.RL_Env(config, res)    
    model = RLSingleQNetwork(rl_env)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
              loss=losses.MeanSquaredError(),
              metrics=[metrics.Mean()])
    # fixed update over time
    debug_str = ""
    if(debug_act):
        debug_str = "debug"
    model.fit(debug_str, epsilon=0.08, reduce_update=1000, epsilon_update=0.0)
    model_loc = os.path.join(dir, file)
    model.save(model_loc)

def load_model_routine():
    pass

def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-l", "--load-weights", help='load trained weights', 
        dest="load_weights", action='store_true')
    arg_parser.add_argument("--version", action='version', version='1.0')
    arg_parser.add_argument("-d", "--debug", help='prints debug messages while training',
        dest="debug", action='store_true')
    arg_parser.add_argument("configuration", help='name of configuration',
        default="deadly_corridor.cfg", metavar='CNF', dest='config')
    arg_parser.add_argument("-t", "--train", help='specifies if train should occur',
        action='store_true', dest='trainable')
    arg_parser.add_argument("-x", "--resX", default=320, help='resolution x (rows)',
        dest="res", action='append', type=int)
    arg_parser.add_argument("-y", "--resY", default=240, help='resolution y (cols)',
        dest="res", action='append', type=int)
    arg_parser.add_argument("--out-dir", default="..", help='specifiy output directory',
        dest="dir")
    arg_parser.add_argument("--out-file", default="model.keras", help='specifiy file name',
        dest="file")
    """arg_parser.add_argument("--epsilon", help="exploration radius : x -> 0 -- less exploration"
                            "x -> 1 -- more exploration")
    arg_parser.add_argument("--") """
    name_space_obj = arg_parser.parse_args()
    
    var_dict = vars(name_space_obj)
    if (var_dict["load_weigths"]):
        load_model_routine()
    elif(var_dict["trainable"]):    
        train_model_routine(var_dict["debug"], var_dict["config"], tuple(var_dict["res"]),
            var_dict["dir"], var_dict["file"])

if __name__ == "__main__":
   parse_arguments()