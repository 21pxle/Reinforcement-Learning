import gymnasium as gym
import numpy as np

from simulation_policy import SimulateUntilTerminated
from simulator import Simulation


def mountain_reward_fn(action: int, observation: np.ndarray):
    return int((observation[1] > 0 and action == 2) or (observation[1] < 0 and action == 0))


if __name__ == '__main__':
    env = gym.make("MountainCar-v0", render_mode="human", max_episode_steps=500)
    simulation = Simulation(env, 3, [2])
    simulation.simulate(reward_function=mountain_reward_fn, plot_file="mountain-car.png",
                        simulation_policy=SimulateUntilTerminated(), verbose=True)
    simulation.close()
