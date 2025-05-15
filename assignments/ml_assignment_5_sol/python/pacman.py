import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt

# Automatically choose best available device: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

# Q-Network
class DQN(nn.Module):
    def __init__(self, action_space):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

    def forward(self, x):
        return self.net(x)

# Preprocess image: grayscale + resize
transform = T.Compose([
    T.ToPILImage(),
    T.Grayscale(),
    T.Resize((84, 84)),
    T.ToTensor()
])

def preprocess(obs):
    obs = transform(obs).to(device)
    return obs.unsqueeze(0)  # add batch dimension

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 100000
TARGET_UPDATE = 1000
MEMORY_SIZE = 100000
LR = 1e-4
EPISODES = 500

def run():
    env = gym.make("ALE/Centipede-v5", render_mode="human")
    n_actions = env.action_space.n

    policy_net = DQN(n_actions).to(device)
    target_net = DQN(n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = deque(maxlen=MEMORY_SIZE)

    epsilon = EPSILON_START
    total_steps = 0
    rewards = []

    for episode in tqdm(range(EPISODES), desc="Training"):
        obs, _ = env.reset()
        state = preprocess(obs)
        episode_reward = 0
        done = False

        while not done:
            total_steps += 1
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = q_values.argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocess(next_obs)

            memory.append((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward

            if len(memory) >= BATCH_SIZE:
                # Sample batch
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards_batch, next_states, dones = zip(*batch)

                states = torch.cat(states).to(device)
                next_states = torch.cat(next_states).to(device)
                actions = torch.tensor(actions, device=device).unsqueeze(1)
                rewards_batch = torch.tensor(rewards_batch, device=device, dtype=torch.float32)
                dones = torch.tensor(dones, device=device, dtype=torch.bool)

                q_values = policy_net(states).gather(1, actions).squeeze()
                with torch.no_grad():
                    max_next_q = target_net(next_states).max(1)[0]
                    target_q = rewards_batch + GAMMA * max_next_q * (~dones)

                loss = nn.MSELoss()(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target net
            if total_steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        rewards.append(episode_reward)
        epsilon = max(EPSILON_END, EPSILON_START - total_steps / EPSILON_DECAY)

    env.close()
    plt.plot(np.convolve(rewards, np.ones(10)/10, mode='valid'))
    plt.title("Episode Reward (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pacman_dqn_rewards.png")
    plt.show()

if __name__ == '__main__':
    run()
