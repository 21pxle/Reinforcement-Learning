from simulation_policy import SimulateUntilTruncated
from simulator import Simulation
import gymnasium as gym

if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode="human")
    simulation = Simulation(env, 2, [4])
    simulation.simulate(plot_file="cart-pole.png", verbose=True,
                        simulation_policy=SimulateUntilTruncated())
    simulation.close()
