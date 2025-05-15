import pickle
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt

DECAY_RATE = 0.001

def run(episodes=1000, render=False):
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode='human' if render else None)

    # Q-Table initialization
    q = np.zeros((env.observation_space.n, env.action_space.n))

    learning_rate_a = 0.9  # alpha
    discount_factor_g = 0.9  # gamma

    epsilon = 1  # 100 % random actions
    epsilon_decay_rate = DECAY_RATE  # decay rate for epsilon

    rng = np.random.default_rng()  # random number generator

    rewards_per_episode = np.zeros(episodes)

    for episode in tqdm(range(episodes), desc="Training episodes"):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])
            new_state, reward, terminated, truncated, _ = env.step(action)
            q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
            )
            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if epsilon == 0:
            learning_rate_a = DECAY_RATE

        if reward == 1:
            rewards_per_episode[episode] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])
        plt.plot(sum_rewards)
        plt.savefig('frozen_lake_q_8x8.png')

    with open("../frozen_lake8x8_rewards.pkl", "wb") as f:
        pickle.dump(sum_rewards, f)


if __name__ == '__main__':
    run(1000, render=True)
