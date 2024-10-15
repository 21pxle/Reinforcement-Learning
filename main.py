import random

import gymnasium as gym
import numpy as np

from dqn import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make("CartPole-v1", render_mode="human")
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=2,
                  eps_min=0.01, input_dims=[4], lr=0.001)
    scores, eps_history = [], []
    n_games = 0
    has_truncated = False

    while not has_truncated:
        score = 0
        done = False
        observation, _ = env.reset()
        env.render()
        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            done = terminated or truncated
            has_truncated = truncated
            agent.store_transition(observation, action, reward, new_observation, done)
            agent.learn()
            observation = new_observation
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-20:])
        n_games += 1

        print(f'Episode {n_games + 1} Results')
        print(f'Score: {score:.2f}')
        print(f'Average Score: {avg_score:.2f}')
        print(f'Epsilon: {agent.epsilon:.2f}')
    x = list(range(1, n_games + 1))
    filename = 'cart_pole.png'
    plot_learning_curve(x, scores, eps_history, filename)
