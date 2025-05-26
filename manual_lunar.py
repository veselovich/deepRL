import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from src.utils import get_device
from src.model import save_model


lr = 1e-3
episodes = 500
rollout_len = 150
n_envs = 64

device = get_device()
print("Using device:", device)

# Create vectorized environment
env = make_vec_env('LunarLander-v3', n_envs=n_envs)

# Policy network
class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=1)

policy = Policy(env.observation_space.shape[0], env.action_space.n).to(device)
optimizer = optim.Adam(policy.parameters(), lr=lr)

# REINFORCE training loop with vectorized env
def train(env, policy, episodes=episodes, gamma=0.99, rollout_len=rollout_len, verbose=False):
    for episode in range(episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)

        log_probs = []
        rewards = []

        for _ in range(rollout_len):
            probs = policy(state)  # shape: [n_envs, action_dim]
            dist = Categorical(probs)
            actions = dist.sample()  # shape: [n_envs]

            log_prob = dist.log_prob(actions)  # [n_envs]

            actions_np = actions.cpu().numpy()
            actions_list = [int(a) for a in actions_np]
            next_state, reward, done, info = env.step(actions_list)

            log_probs.append(log_prob)
            rewards.append(torch.tensor(reward, dtype=torch.float32).to(device))

            state = torch.tensor(next_state, dtype=torch.float32).to(device)

        # Stack and compute returns
        log_probs = torch.stack(log_probs, dim=1)   # shape: [n_envs, rollout_len]
        rewards = torch.stack(rewards, dim=1)       # shape: [n_envs, rollout_len]

        returns = torch.zeros_like(rewards)
        for t in reversed(range(rollout_len)):
            if t == rollout_len - 1:
                returns[:, t] = rewards[:, t]
            else:
                returns[:, t] = rewards[:, t] + gamma * returns[:, t + 1]

        # Normalize
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        loss = -torch.sum(log_probs * returns) / n_envs

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_total_reward = rewards.sum(1).mean().item()
        if verbose:
            print(f"Episode {episode}, avg_total_reward={avg_total_reward:.2f}")

        if avg_total_reward > 250.0:
            break

train(env, policy, verbose=True)


# Manual evaluation
def evaluate(policy, env_name='LunarLander-v3', episodes=10, verbose=False):
    env = gym.make(env_name, render_mode="human")
    total_rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            probs = policy(state_tensor)
            action = torch.argmax(probs, dim=1).item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        total_rewards.append(ep_reward)
        if verbose:
            print(f"Eval Episode {ep}: Reward = {ep_reward:.2f}")

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nMean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward

mean_reward, std_reward = evaluate(policy, verbose=True)

if (mean_reward - std_reward) >= 200:
    save_model(model=policy, target_dir="model", model_name="LaunarLander3.pth")