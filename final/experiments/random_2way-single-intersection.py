import os
import sys

import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

from sumo_rl import SumoEnvironment

env = SumoEnvironment(
    net_file="nets/2way-single-intersection/single-intersection.net.xml",
    route_file="nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
    out_csv_name="outputs/2way-single-intersection/random",
    single_agent=True,
    use_gui=True,
    num_seconds=500,
)

# run simualtion with random actions
obs, info = env.reset()
done = False
while not done:
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated


env.save_csv('outputs/2way-single-intersection/random/random', 1)
env.close()