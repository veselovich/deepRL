import torch
from torch import nn
import gymnasium as gym
from tqdm import trange

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class PPO(nn.Module):
    def __init__(self, state_dim, num_units, num_layers, action_dim):
        super().__init__()
        layers = [nn.Linear(state_dim, num_units), nn.ReLU()]
        for _ in range(num_layers):
            layers += [nn.Linear(num_units, num_units), nn.GELU()]
        self.layers = nn.Sequential(*layers)
        self.actor = nn.Linear(num_units, action_dim)
        self.critic = nn.Linear(num_units, 1)

    def forward(self, x):
        x = self.layers(x)
        return self.actor(x), self.critic(x)

    def act(self, state):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze().item()

    def eval_actions(self, state, actions):
        logits, val = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, val.squeeze(-1), entropy


env = gym.make("LunarLander-v3", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy = PPO(state_dim=state_dim, num_units=128, num_layers=2, action_dim=action_dim).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)

# PPO specific parameters
clip_epsilon = 0.2
ppo_epochs = 4
entropy_coef = 0.1
critic_coef = 1.0
gamma = 0.99
num_steps = 512
num_episodes = 1000

def collect_rollout(env, model, n_steps, device="cpu"):
    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
    state, _ = env.reset()
    model.to(device)

    for _ in range(n_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob, value = model.act(state_tensor)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)
        values.append(value)

        if done:
            state, _ = env.reset()
        else:
            state = next_state

    return states, actions, rewards, dones, log_probs, values, next_state, done

def compute_gae(rewards, values, dones, next_value, gamma, lam=0.95):
    advantages, returns = [], []
    gae = 0
    values = values + [next_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
    return returns, advantages

for episode in trange(num_episodes):
    policy.train()

    states, actions, rewards, dones, log_probs, values, next_state, next_done = collect_rollout(
        env, policy, num_steps, device
    )

    if next_done:
        next_value = 0.0
    else:
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _, next_value = policy(next_state_tensor)
        next_value = next_value.item()

    returns, advantages = compute_gae(rewards, values, dones, next_value, gamma)
    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Convert everything to tensors
    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.int64).to(device)
    old_log_probs_tensor = torch.stack(log_probs).detach().to(device)
    returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)

    for _ in range(ppo_epochs):
        new_log_probs, value_preds, entropy = policy.eval_actions(states_tensor, actions_tensor)
        ratio = (new_log_probs - old_log_probs_tensor).exp()
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        actor_loss = -torch.min(surrogate1, surrogate2).mean()
        critic_loss = nn.functional.mse_loss(value_preds, returns_tensor)
        entropy_bonus = entropy.mean()

        loss = actor_loss + critic_coef * critic_loss - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if episode % 10 == 0:
        print(f"\nEpisode {episode}, Return: {sum(rewards):.2f}, Loss: {loss.item():.2f}")

# Evaluation remains unchanged
def evaluate_policy(model, episodes=5, device="cpu"):
    model.eval()
    total_reward = 0
    for _ in range(episodes):
        done = False
        state, _ = env.reset()
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(state_tensor)
                action = torch.argmax(logits, dim=1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
    avg = total_reward / episodes
    print(f"Average reward: {avg}")
    if avg >= 100.:
        torch.save(policy.state_dict(), "models/lunar_lander_ppo_policy.pth")

evaluate_policy(policy)
