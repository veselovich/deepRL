from collections import deque
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
episodes = 512
rollout_len = 128
n_envs = 4
hidden_units = 64

device = get_device()
print("Using device:", device)

# Create vectorized environment
env = make_vec_env('LunarLander-v3', n_envs=n_envs)

# Policy network
class Policy(nn.Module):
    def __init__(self, state_size, hidden_units, action_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units*2)
        self.fc3 = nn.Linear(hidden_units*2, action_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=1)
    
    def act(self, state):
        probs = self.forward(state)  # shape: [n_envs, action_dim]
        dist = Categorical(probs)
        actions = dist.sample()  # shape: [n_envs]
        log_prob = dist.log_prob(actions)
        return actions, log_prob

policy = Policy(env.observation_space.shape[0], hidden_units, env.action_space.n).to(device)
optimizer = optim.Adam(policy.parameters(), lr=lr)

def train(env, policy, episodes=episodes, gamma=0.99, rollout_len=rollout_len, verbose=False, print_every=100, device="cpu"):
    
    policy.to(device)
    scores = []

    for episode in range(1, episodes+1):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)

        log_probs = []
        rewards = []
        masks = []

        for _ in range(rollout_len):
            actions, log_prob = policy.act(state)
            actions_np = actions.cpu().numpy()

            next_state, reward, done, info = env.step(actions_np)

            # Convert to tensors
            reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device)
            done_tensor = torch.tensor(done, dtype=torch.float32).to(device)
            mask = 1.0 - done_tensor  # 0 where done, 1 otherwise

            log_probs.append(log_prob)
            rewards.append(reward_tensor)
            masks.append(mask)

            # Manually reset only those environments where done=True
            for i, d in enumerate(done):
                if d:
                    obs, _ = env.envs[i].reset()
                    obs = np.asarray(obs, dtype=next_state.dtype)
                    next_state[i] = obs

            state = torch.tensor(next_state, dtype=torch.float32).to(device)

        # Stack
        log_probs = torch.stack(log_probs, dim=1)   # [n_envs, rollout_len]
        rewards = torch.stack(rewards, dim=1)       # [n_envs, rollout_len]
        masks = torch.stack(masks, dim=1)           # [n_envs, rollout_len]

        # Compute discounted returns with masking
        returns = torch.zeros_like(rewards)
        for t in reversed(range(rollout_len)):
            if t == rollout_len - 1:
                returns[:, t] = rewards[:, t]
            else:
                returns[:, t] = rewards[:, t] + gamma * returns[:, t + 1] * masks[:, t]

        # Normalize
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        loss = -torch.sum(log_probs * returns) / n_envs

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_returns = rewards.sum(1)
        avg_total_reward = episode_returns.mean().item()
        scores.append(episode_returns)
        if verbose and episode%print_every == 0:
            print(f"Episode {episode}, avg_total_reward={avg_total_reward:.2f}")

    scores = torch.cat(scores)
    return scores

train(env, policy, verbose=True, print_every=10)


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