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

    def __init__(self, debug_in_epochs=1000):
        super().__init__()
        self.debug_in_epochs = debug_in_epochs

    def on_epoch_end(self, epoch, logs):
        hist_ :History = logs["hist"]
        debug_history = hist_.history_of(0, epoch, 'all')

        if (epoch % self.debug_in_epochs) == 0:
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

class CheckpointCallback(callbacks.Callback):

    def __init__(self, checkpoint_when=100000):
        super().__init__()
        self.checkpoint_on_epoch = checkpoint_when

    def on_epoch_end(self, epoch, logs=None):
        model = logs["rl_model"]
        if(epoch % self.checkpoint_on_epoch == 0):
            print("Checkpoint reached -- storing weights ...")
            model.model_checkpointing()

class TrainCallback(callbacks.Callback):

    def __init__(self, log_training_in_epochs=50, log_debug_training_in_epochs=100):
        super().__init__()
        self.log_training_in_epochs = log_training_in_epochs
        self.log_debug_training_in_epochs = log_debug_training_in_epochs

    def on_epoch_end(self, epoch, logs=None):
        reward_metric = logs["metric"]
        q_star_value  = logs["q*_value"]
        q_value       = logs["q_value"]
        loss          = logs["loss"]
        hist_         = logs["hist"]
        if (epoch % self.log_debug_training_in_epochs) == 0:
            print(f"Sample --> Epoch:{epoch} -- Loss:{loss} "
                f"-- Q-Val:{q_value} -- Q*-Val:{q_star_value}")
            self.log_dbg_training(hist_, epoch, self.log_debug_training_in_epochs, 
                reward_metric)

        if (epoch % self.log_training_in_epochs) == 0:
            self.log_training(epoch, q_star_value, q_value, loss)
            
    def log_training(self, epoch, q_star_value, q_value, loss):   
        with open(os.path.join(os.getcwd(), "logs", "run.log"), 'a') as file:
            file.write(f"Logging for epoch {epoch} ...\n")
            file.write(f"Q-Value = {q_value} , Target Q-Value = {q_star_value} , Loss = {loss}\n")

    def log_dbg_training(self, _hist, epoch, minus_epoch, reward_metric):
        history = _hist.history_of(epoch-minus_epoch, epoch, 'all')
        mean_loss = np.mean(history["loss"])
        mean_q_value = np.mean(history["q_value"])
        mean_q_target_value = np.mean(history["q*_value"])
        with open(os.path.join(os.getcwd(), "logs", "dbg.log"), 'a') as file:
            file.write(f"Logging mean values from epoch {epoch-minus_epoch} to epoch {epoch}\n")
            file.write(f"Mean Loss     = {mean_loss}\n")
            file.write(f"Mean Q-Value  = {mean_q_value}\n")
            file.write(f"Mean Q-Target = {mean_q_target_value}\n")
            file.write(f"Mean Reward   = {reward_metric.result()}\n")
        
    def on_train_end(self, logs=None):
        with open("tmp/model_parameters.info", 'w') as file:
            pass

class History():
    
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
        if ('all' not in args) and ('loss' not in args) and ('q_value' not in args) and \
            ('q*_value' not in args):
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

@keras.saving.register_keras_serializable(name="RL-Q-Network")
class RLSingleQNetwork(keras.Model): 

    def __init__(self, rl_environment, fixed_reduce_value=100):
        super().__init__()
        self.rl_environment = rl_environment
        self.fixed_reduce_value = fixed_reduce_value
        self.ckpt_reached       = 0
        self.handling_callbacks = callbacks.CallbackList()
        
        print("environment shape : ", self.rl_environment.get_env_shape())
        print("buttons supported : ", self.rl_environment.count_buttons_supported())
        print("buttons defined   : ", self.rl_environment.def_buttons())
        self.conv_layer1   = layers.Conv2D(16, (6,6), activation="relu",
            input_shape=self.rl_environment.get_env_shape(), use_bias=False)
        self.pool_layer1   = layers.MaxPooling2D(pool_size=(3,3), padding='same')
        self.conv_layer2   = layers.Conv2D(32, (5,5), activation="relu", use_bias=False)
        self.pool_layer2   = layers.MaxPooling2D(pool_size=(3,3), padding='same')
        self.conv_layer3   = layers.Conv2D(64, (4,4), activation="relu", use_bias=False)
        self.pool_layer3   = layers.MaxPooling2D(pool_size=(2,2), padding='same')
        self.conv_layer4   = layers.Conv2D(128, (3,3), activation="relu", use_bias=False)
        self.pool_layer4   = layers.MaxPooling2D(pool_size=(2,2), padding='same')
        self.conv_layer5   = layers.Conv2D(256, (2,2), activation="relu", use_bias=False) 
        self.pool_layer5   = layers.MaxPooling2D(pool_size=(2,2), padding='same')
        self.flatten_layer = layers.Flatten()
        self.dense_layer1  = layers.Dense(256)
        self.leaky_relu1   = layers.LeakyReLU()
        self.dense_layer2  = layers.Dense(128)
        self.leaky_relu2   = layers.LeakyReLU()
        self.dense_layer3  = layers.Dense(self.rl_environment.count_buttons_supported())

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
        x = self.dense_layer1(x)
        x = self.leaky_relu1(x)
        x = self.dense_layer2(x)
        x = self.leaky_relu2(x)
        x = self.dense_layer3(x)
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
        reshaped_screen_buffer = self.to_data_buffer(updated_game_state)
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

        return self.loss(tf.convert_to_tensor([reward + delta * q_star_value]), tf.convert_to_tensor([q_value]))

    def to_data_buffer(self, game_state):
        a = np.asarray(game_state.screen_buffer) \
            .astype(np.float32).reshape(self.rl_environment.get_env_shape())
        a = self.normalize_image(a)
        return tf.expand_dims(a, axis=0)

    def normalize_image(self, reshaped_screen_buffer): # [-1, 1]
        return (reshaped_screen_buffer - 127.5) / 127.5

    def register_callback(self, callback: callbacks.Callback):
        self.handling_callbacks.append(callback)

    def model_checkpointing(self):
        print(f"Writing weights to ckpt_{self.ckpt_reached+1}.weights.h5")
        path_str = os.path.join("checkpoints", f"ckpt_{self.ckpt_reached+1}.weights.h5")
        self.save_weights(path_str, overwrite=True)
        self.ckpt_reached += 1

    """
    reduce_fctn : over_time or fixed
    reduce_update : over_time -> update value , fixed -> fixed value to stop
    """
    def fit(self, *args, epochs=1, delta=0.95, epsilon=0.1, epsilon_update=0.001, 
            epsilon_l_b=0.0001, reduce_update=1000, reduce_fctn="over_time"):
        rnd_seed = uniform.rvs()
        reduce_function = None
        limit_reached = False
        if reduce_fctn == "over_time":
            reduce_function = self.reduce_over_time
        elif reduce_fctn == "fixed":
            reduce_function = self.reduce_fixed

        if "debug" in args:
            self.register_callback(DebugCallback())
            
        history = History()
        self.register_callback(TrainCallback())
        self.register_callback(CheckpointCallback())
        self.log_parameters_used(epochs, delta, epsilon, epsilon_update,
            epsilon_l_b, reduce_update, reduce_function, 0)

        self.handling_callbacks.on_train_begin()

        for epoch in range(epochs):
            game_state = self.rl_environment.get_game_state()
            reshaped_screen_buffer = self.to_data_buffer(game_state)
            reward, q_value, q_star_value, loss, _ = self.train_step(
                reshaped_screen_buffer, rnd_seed, delta, epsilon, *args)
            self.metrics[0].update_state(reward)
            if(not limit_reached):
                epsilon, limit_reached = reduce_function(epoch, epsilon, epsilon_update, 
                    epsilon_l_b, reduce_update)
            history.update_history((q_value, q_star_value, loss))
                
            self.handling_callbacks.on_epoch_end(epoch+1, {"hist":history, "metric":self.metrics[0],
                "loss":loss, "q*_value":q_star_value, "q_value":q_value, "rl_model":self})
        #self.handling_callbacks.on_train_end()
        self.rl_environment.close_env()

    def log_parameters_used(self, epochs, delta, epsilon, epsilon_update,
        epsilon_lower_bound, reduce_update, reduce_function, learning_rate):
        _optimizer = str(self.optimizer)
        _metric    = str(self.metrics[0])
        _loss      = str(self.loss)

        with open(os.path.join(os.getcwd(), "logs", "run.log"), 'a') as file:
            file.write("Model components :\n")
            file.write(f"Optimzer = {_optimizer}\n")
            file.write(f"Metric   = {_metric}\n")
            file.write(f"Loss     = {_loss}\n\n")
            file.write("Model parameters : \n")
            file.write(f"Epochs          = {epochs}\n")
            file.write(f"Delta           = {delta}\n")
            file.write(f"Epsilon         = {epsilon}\n")
            file.write(f"Epsilon Update  = {epsilon_update}\n")
            file.write(f"Epsilon Lower Bound = {epsilon_lower_bound}\n")
            file.write(f"Reduce Update   = {reduce_update}\n")
            file.write(f"Reduce Function = {str(reduce_function)}\n")
            file.write(f"Learning Rate   = {learning_rate}\n\n")

    def evaluate(self, x=None, y=None, batch_size=None, verbose="auto", sample_weight=None, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, return_dict=False, **kwargs):
        return super().evaluate(x, y, batch_size, verbose, sample_weight, steps, callbacks, max_queue_size, workers, use_multiprocessing, return_dict, **kwargs)

    def predict(self, game_state):
        data   = self.to_data_buffer(game_state)
        return self(data)
    
    def get_model_layers(self):
        return {
            "conv_layer_1" : self.conv_layer1,
            "conv_layer_2" : self.conv_layer2,
            "conv_layer_3" : self.conv_layer3,
            "conv_layer_4" : self.conv_layer4,
            "conv_layer_5" : self.conv_layer5,
            "pool_layer_1" : self.pool_layer1,
            "pool_layer_2" : self.pool_layer2,
            "pool_layer_3" : self.pool_layer3,
            "pool_layer_4" : self.pool_layer4,
            "pool_layer_5" : self.pool_layer5,
            "flattern_flayer" : self.flatten_layer,
            "dense_layer_1" : self.dense_layer1,
            "dense_layer_2" : self.dense_layer2,
            "dense_layer_3" : self.dense_layer3,
            "leaky_relu_1" : self.leaky_relu1,
            "leaky_relu_2" : self.leaky_relu2 
        }

    def get_config(self):
        base_configuration   = super().get_config()
        rl_env_serialization = self.rl_environment.get_config()
        architecture_configuration = {
            "rl_environment" : rl_env_serialization,
            "fixed_reduce_value" : self.fixed_reduce_value
        } 
        return {**base_configuration, **architecture_configuration}

    @classmethod
    def from_config(cls, config):
        rl_environment = RL_Env.RL_Env.from_config(config.pop("rl_environment"))
        fixed_value   = config.pop("fixed_reduce_value")
        return cls(rl_environment, fixed_value)

def create_log():
    if not os.path.exists(os.path.join(os.getcwd(), "logs", "run.log")):
        with open(os.path.join(os.getcwd(), "logs", "run.log"), 'x') as _:
            pass
        with open(os.path.join(os.getcwd(), "logs", "dbg.log"), 'x') as _:
            pass
    else:
        with open(os.path.join(os.getcwd(), "logs", "run.log"), 'w') as _:
            pass
        with open(os.path.join(os.getcwd(), "logs" , "dbg.log"), 'w') as _:
            pass

def train_model_routine(debug_act: bool, config : str, res : tuple[int, int],
    dir : str, file: str, epochs, delta, eps, eps_update, eps_lb, learning):
    rl_env = RL_Env.RL_Env(config, res)    
    model = RLSingleQNetwork(rl_env)
    model.compile(optimizer=optimizers.Adam(learning_rate=learning),
              loss=losses.MeanSquaredError(),
              metrics=[metrics.Mean()])
    # fixed update over time
    debug_str = ""
    if(debug_act):
        debug_str = "debug"
    create_log()
    model.fit(debug_str, epochs=epochs, delta=delta, epsilon=eps, epsilon_update=eps_update,
        epsilon_l_b=eps_lb)
    model_loc = os.path.join(dir, file)
    print("model weigths location : ", model_loc)
    model.save(model_loc)

def load_model_routine(dir : str, file : str,  inference_c : int,
        epochs, delta, eps, eps_update, eps_lb, inference_include_restart : bool, 
        inference : bool, checkpoint : int):
    
    if(inference):
        files_in_dir = os.listdir("checkpoints")
        if(checkpoint < len(files_in_dir)):
            raise Exception(f"Checkpoint is not in valid range - Expected range : [1,{files_in_dir}]"
                            f"But received {checkpoint}\n")
        model = keras.models.load_weights(os.path.join("checkpoint", f"ckpt_{checkpoint}.weights.h5"))
        for inf_i in range(inference_c):
            if(not inference_include_restart and model.rl_environment.is_episode_finished()):
                break
            game_state = model.rl_environment.get_game_state()
            action = model.predict(game_state)
            reward = model.rl_environment.perform_action(action)
            print(f"Inference's {inf_i} reward : {reward}")
        return

    model_loc = os.path.join(dir, file)
    model = keras.models.load_model(model_loc)

    model.fit(epochs=epochs, delta=delta, epsilon=eps, epsilon_update=eps_update,
                epsilon_l_b=eps_lb)
    
def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-l", "--load-model", help='load model (arch, weights, state)', 
        dest="load_model", action='store_true')
    arg_parser.add_argument("--version", action='version', version='1.0')
    arg_parser.add_argument("-d", "--debug", help='prints debug messages while training',
        dest="debug", action='store_true')
    arg_parser.add_argument("--configuration", help='name of configuration',
        default="deadly_corridor.cfg", metavar='CNF')
    arg_parser.add_argument("-t", "--train", help='specifies if train should occur',
        action='store_true', dest='trainable')
    arg_parser.add_argument("-y", "--resY", help='resolution y (cols)',
        dest="res", action='append', type=int)
    arg_parser.add_argument("-x", "--resX", help='resolution x (rows)',
        dest="res", action='append', type=int)
    arg_parser.add_argument("--out-dir", default=".", help='specifiy output directory',
        dest="dir")
    arg_parser.add_argument("--out-file", default="model.keras", help='specifiy file name',
        dest="file")
    arg_parser.add_argument("--epsilon", help="exploration radius : x -> 0 -- less exploration"
                            "x -> 1 -- more exploration", dest="eps", type=float, default=0.08)
    arg_parser.add_argument("--epsilon-update", dest="eps_update", type=float, default=0.001)
    arg_parser.add_argument("--epsilon-lb", dest="eps_lb", type=float, default=0.0001)
    arg_parser.add_argument("--delta", dest="delta", type=float, default=0.95)
    arg_parser.add_argument("--epochs", dest="epochs", type=int, default=500000)
    arg_parser.add_argument("--learning", dest="learning", type=float, default=1e-5)
    arg_parser.add_argument("--inf", dest="inference", action="store_true")
    arg_parser.add_argument("--inf-count", dest="inference_count", type=int, default=1000)
    arg_parser.add_argument("--inf-include-restart", dest="inference_restart_ok", default=True)
    arg_parser.add_argument("--ckpt", dest="checkpoint", type=int, default=1)
    name_space_obj = arg_parser.parse_args()
    
    var_dict = vars(name_space_obj)
    
    if (var_dict["load_model"]):
        load_model_routine(var_dict["dir"], var_dict["file"], var_dict["inference_count"],
            var_dict["epochs"], var_dict["delta"], var_dict["eps"], var_dict["eps_update"],
            var_dict["eps_lb"], var_dict["inference_restart_ok"], var_dict["inference"],
            var_dict["checkpoint"])
    elif(var_dict["trainable"]):    
        train_model_routine(var_dict["debug"], var_dict["configuration"], tuple(var_dict["res"]),
            var_dict["dir"], var_dict["file"], var_dict["epochs"],
            var_dict["delta"], var_dict["eps"], var_dict["eps_update"],
            var_dict["eps_lb"], var_dict["learning"])

if __name__ == "__main__":
   parse_arguments()