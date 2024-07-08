import os
import vizdoom as vzd
from random import choice
from time import sleep

class RL_Env:
   
   def __init__(self, cfg : str, res : tuple[int, int]=(320, 240)):
    self.game = game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, cfg)) # or any other scenario file
    if(res[0] == 320 and res[1] == 256):
      game.set_screen_resolution(vzd.ScreenResolution.RES_320X256)
    elif(res[0] == 400 and res[1] == 300):
      game.set_screen_resolution(vzd.ScreenResolution.RES_400X300)
    elif(res[0] != 320 or res[1] != 240):
      raise Exception("resolution not supported ...")

    self.buttons_supported = game.get_available_buttons()
    game.init()

   def count_buttons_supported(self):
    return len(self.buttons_supported)
   
   def def_buttons(self):
    return [(e.name, e.value) for e in self.buttons_supported]

   def get_game_state(self):
    if self.game.is_episode_finished():
        self.game.new_episode()
    return self.game.get_state()
   
   def get_episode_time(self):
    return self.game.get_episode_time()
   
   def perform_action(self, action):
    if self.game.is_episode_finished():
        self.game.new_episode()
    return self.game.make_action(action) # return reward
   
   def get_env_shape(self):
    return (self.game.get_screen_height(), self.game.get_screen_width(), self.game.get_screen_channels())

   def get_total_reward(self):
    return self.game.get_total_reward()

   def get_last_reward(self):
    return self.game.get_last_reward()

   def close_env(self):
    self.game.close()

"""actions = [[True, True, True, True, True, True, True], [True, True, True, True, True, True, False], [True, True, True, False, True, True, True],
           [False, True, False, True, True, False, False], [True, False, True, False, True, True, False], [False, False, True, True, True, True, True], 
           [True, True, True, True, True, True, False]]

import gymnasium as gym
from vizdoom import gymnasium_wrapper # This import will register all the environments

env = gym.make("VizdoomCorridor-v0", render_mode="human") # or any other environment id
observation, infos = env.reset()

print("", env.action_space, env.observation_space)

for __ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    print("action : ", action)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close() """
