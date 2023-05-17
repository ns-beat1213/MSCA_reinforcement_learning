import os
import sys

import numpy as np
import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from tqdm.auto import tqdm

from sumo_rl import SumoEnvironment

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci

# class for saving model
class SaveModelCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, verbose=1):
        super(SaveModelCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.num_steps = 0

    def _on_step(self) -> bool:
        self.num_steps += 1
        if self.num_steps % self.check_freq == 0:
            self.model.save(os.path.join(self.save_path, f'model_{self.num_steps}'))
            if self.verbose > 0:
                print(f"Saving model checkpoint at step {self.num_steps}")
        return True

# class for progress bar
class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=1):
        super(ProgressBarCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        if self.verbose > 0:
            self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self) -> bool:
        if self.pbar is not None:
            self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()

# episode
n_steps = 100000

# run the model
if __name__ == "__main__":
    env = SumoEnvironment(
        net_file="nets/2way-single-intersection/single-intersection.net.xml",
        route_file="nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        out_csv_name="outputs/2way-single-intersection/dqn",
        single_agent=True,
        use_gui=True,
        num_seconds=n_steps  # /1000,
    )

    save_model_callback = SaveModelCallback(check_freq=100, save_path="saved_models")
    progress_bar_callback = ProgressBarCallback(total_timesteps=n_steps)
    
    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.001,
        learning_starts=0,
        train_freq=1,
        target_update_interval=500,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.01,
        verbose=1,
    )

    # learn
    model.learn(total_timesteps=n_steps,
                callback=CallbackList([save_model_callback, 
                                       progress_bar_callback])
               )  
    
