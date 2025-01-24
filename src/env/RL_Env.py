import os
import vizdoom as vzd
from random import choice
from time import sleep

class RL_Env:
   
  def __init__(self, cfg : str, res : tuple[int, int]=(240, 320), 
               start_game=True):
    self.game = game = vzd.DoomGame()
    no_errors_while_configuring = game.load_config(os.path.join(
      vzd.scenarios_path, cfg)) # or any other scenario file
    if(not no_errors_while_configuring):
      raise Exception("error encountered while configuring game")
    self.configuration = cfg
    if(res[0] == 320 and res[1] == 256):
      game.set_screen_resolution(vzd.ScreenResolution.RES_320X256)
    elif(res[0] == 400 and res[1] == 300):
      game.set_screen_resolution(vzd.ScreenResolution.RES_400X300)
    elif(res[0] != 240 and res[1] != 320):
      raise Exception("resolution not supported ...")

    self.buttons_supported = game.get_available_buttons()
    if start_game:
      self.init_state = True
      game.init()

  def count_buttons_supported(self):
    return len(self.buttons_supported)
   
  def def_buttons(self):
    return [(e.name, e.value) for e in self.buttons_supported]

  def get_game_state(self):
    return self.game.get_state()

  def start_new_episode(self):
    self.game.new_episode()

  def get_episode_time(self):
    return self.game.get_episode_time()
   
  def is_episode_finished(self):
    return self.game.is_episode_finished()

  def perform_action(self, action):
    return self.game.make_action(action) # return reward
   
  def get_env_shape(self):
    return (self.game.get_screen_height(), self.game.get_screen_width(), self.game.get_screen_channels())

  def get_total_reward(self):
    return self.game.get_total_reward()

  def get_last_reward(self):
    return self.game.get_last_reward()

  def close_env(self):
    self.game.close()

  def get_config(self):
    x, y, *_ = self.get_env_shape()
    return {
      "cfg" : self.configuration,
      "init_state" : False if not hasattr(self, 'init_state') else True,
      "res" : (x, y)
  }

  @classmethod
  def from_config(cls, config):
    config_str = config.pop("cfg")
    resolution = config.pop("res")
    start_game = config.pop("init_state")
    return cls(config_str, resolution, start_game)