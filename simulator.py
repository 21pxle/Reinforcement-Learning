import operator
from typing import Callable

import numpy as np
import functools

from gymnasium import Env

from dqn import Agent
from utils import plot_learning_curve

from simulation_policy import SimulationPolicy, State


class Simulation:
    def __init__(self, agent_env: Env, n_actions: int, input_dims: list[int],
                 eps_function: Callable[[float], float] | None = None, **agent_args):
        """
            Initializes a simulation of the environment.
            :param env: The environment
            :param eps_function: The function
            :param agent_args: The arguments for the agent. Valid args are `gamma`, \
            `epsilon`, `batch_size`, `eps_min`, and `lr` (learning rate).
            """
        valid_args = ['gamma', 'epsilon', 'eps_min', 'batch_size', 'lr']

        for key in agent_args:
            if key not in valid_args:
                raise ValueError(f"{key} is not a valid agent argument. Valid arguments are "
                                 f"'gamma', 'epsilon', 'batch_size', 'eps_min', or 'lr'.")

        default_eps_function = functools.partial(operator.mul, 1-5e-5)
        self.agent = Agent(eps_fn=eps_function or default_eps_function,
                           input_dims=input_dims, n_actions=n_actions, **agent_args)
        self.agent_env = agent_env

    def simulate(self, plot_file: str, verbose: bool, simulation_policy: SimulationPolicy, reward_function: Callable[[np.ndarray | float, np.ndarray], float] | None = None):
        scores, eps_history = [], []
        state: State = State(0, 0, 0)

        while simulation_policy.can_continue(state):
            score = 0
            done = False
            observation, _ = self.agent_env.reset()
            self.agent_env.render()
            while not done:
                action = self.agent.choose_action(observation)
                new_observation, reward, terminated, truncated, info = self.agent_env.step(action)

                if reward_function is not None:
                    reward = reward_function(action, observation)

                score += reward
                done = terminated or truncated
                state.terminated_times += int(terminated)
                state.truncated_times += int(truncated)
                self.agent.store_transition(observation, action, reward, new_observation, done)
                self.agent.learn()
                observation = new_observation
            scores.append(score)
            eps_history.append(self.agent.epsilon)
            avg_score = np.mean(scores[-20:])
            state.episode_num += 1

            if verbose:
                print(f'Episode {state.episode_num + 1} Results')
                print(f'Score: {score:.2f}')
                print(f'Average Score: {avg_score:.2f}')
                print(f'Epsilon: {self.agent.epsilon:.2f}')

            del observation
        x = list(range(1, state.episode_num + 1))
        plot_learning_curve(x, scores, eps_history, plot_file)

    def close(self):
        self.agent_env.close()
