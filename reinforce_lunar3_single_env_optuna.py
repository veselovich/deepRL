import numpy as np

from collections import deque

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Gym
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
RecordVideo.__del__ = lambda self: None

import optuna

device = torch.device(
    "mps"
    if torch.backends.mps.is_built() and torch.backends.mps.is_available()
    else "cpu"
)

print(device)


env_id = "LunarLander-v3"
# Create the env
env = gym.make(env_id)

# Get the state space and action space
s_size = env.observation_space.shape[0]
a_size = env.action_space.n


class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size * 2)
        self.fc3 = nn.Linear(h_size * 2, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every=100):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Line 3 of pseudocode
    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []
        state, info = env.reset()
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state

            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Line 6 of pseudocode: calculate the return
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)

        for t in range(n_steps)[::-1]:
            disc_return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(gamma * disc_return_t + rewards[t])

        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()
        ## eps is the smallest representable float, which is
        # added to the standard deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Line 7:
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        # Line 8: PyTorch prefers gradient descent
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print(
                "Episode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_deque)
                )
            )

    return np.mean(scores_deque)


lunarlander_hyperparameters = {
    "h_size": 64,
    "n_training_episodes": 2000,
    "n_evaluation_episodes": 3,
    "max_t": 128,
    "gamma": 0.99,
    "lr": 1e-3,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
}


# Create policy and place it to the device
lunarlander_policy = Policy(
    lunarlander_hyperparameters["state_space"],
    lunarlander_hyperparameters["action_space"],
    lunarlander_hyperparameters["h_size"],
).to(device)
lunarlander_optimizer = optim.Adam(
    lunarlander_policy.parameters(), lr=lunarlander_hyperparameters["lr"]
)


def objective(trial):
    env = gym.make(env_id)
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    # Suggest hyperparameters
    h_size = trial.suggest_int("h_size", 32, 256)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    max_t = trial.suggest_int("max_t", 200, 500)
    n_training_episodes = 300  # keep small for fast trials

    # Create policy and optimizer
    policy = Policy(s_size, a_size, h_size).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Run training and get performance
    avg_score = reinforce(policy, optimizer, n_training_episodes, max_t, gamma)
    return avg_score


# # 🔍 Run Optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best arams:", study.best_trial.params)

lunarlander_hyperparameters.update(study.best_trial.params)


scores = reinforce(
    lunarlander_policy,
    lunarlander_optimizer,
    lunarlander_hyperparameters["n_training_episodes"],
    lunarlander_hyperparameters["max_t"],
    lunarlander_hyperparameters["gamma"],
)


def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The Reinforce agent
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state, info = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_rewards_ep += reward

            if done:
                break
            state = next_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


eval_env = gym.make("LunarLander-v3", render_mode="rgb_array")
eval_env = RecordVideo(
    eval_env,
    video_folder="./videos",
    episode_trigger=lambda episode_id: True,  # record every episode
    name_prefix="reinforce_eval",
)

print(
    evaluate_agent(
        eval_env,
        lunarlander_hyperparameters["max_t"],
        lunarlander_hyperparameters["n_evaluation_episodes"],
        lunarlander_policy,
    )
)