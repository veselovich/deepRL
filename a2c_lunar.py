import torch
from torch import nn

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"


class A2C(nn.Module):
    def __init__(self, state_dim, num_units, num_layers, action_dim):
        super().__init__()

        layers = []
        layers.append(nn.Linear(in_features=state_dim, out_features=num_units))
        layers.append(nn.ReLU())
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features=num_units, out_features=num_units))
            layers.append(nn.GELU())
        
        self.layers = nn.Sequential(*layers)

        self.actor = (nn.Linear(in_features=num_units, out_features=action_dim))
        self.critic = (nn.Linear(in_features=num_units, out_features=1))

    def forward(self, x):
        x = self.layers(x)
        return self.actor(x), self.critic(x)

    def act(self, state):
        logits, value = self.forward(state)
        probs = torch.distributions.Categorical(logits=logits)
        action = probs.sample()
        return action.item(), probs.log_prob(action), value.squeeze().item()

    def eval_actions(self, state, actions):
        logits, val = self.forward(state)
        probs = torch.distributions.Categorical(logits=logits)
        log_probs = probs.log_prob(actions)
        entropy = probs.entropy()
        return log_probs, val.squeeze(-1), entropy
    

import gymnasium as gym

env_id = "LunarLander-v3"
env = gym.make(env_id, render_mode="human")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
num_units = 128
num_layers = 2


policy = A2C(state_dim=state_dim, num_units=num_units, num_layers=num_layers, action_dim=action_dim)
optimizer = torch.optim.Adam(params=policy.parameters(), lr=5e-4)

gamma = 0.99
num_steps = 512
num_episodes = 1000
entropy_coef = 0.1
critic_coef = 1.0


def collect_rollout(env, model, n_steps, device="cpu"):
    states = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []

    state, info = env.reset()
    model.to(device)

    for _ in range(n_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.inference_mode():
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


def compute_rewards(rewards, values, dones, next_value, gamma):
    returns = []
    advs = []
    R = next_value
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R * (1 - dones[t])
        adv = R - values[t]
        returns.append(R)
        advs.append(adv)
    
    return returns[::-1], advs[::-1]


from tqdm import trange

for episode in trange(num_episodes):
    policy.train()

    states, actions, rewards, dones, log_probs, values, next_state, next_done = collect_rollout(
        env=env,
        model=policy,
        n_steps=num_steps,
        device=device
        )
    
    if next_done:
        next_value = 0.0
    else:
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _, next_value = policy(next_state_tensor)
        next_value = next_value.item()

    returns, advantages = compute_rewards(
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=next_value,
        gamma=gamma
        )
    

    
    states_tensor = torch.tensor(states, dtype=torch.float32, requires_grad=False).to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.int64, requires_grad=False).to(device)
    returns_tensor = torch.tensor(returns, dtype=torch.float32, requires_grad=False).to(device)
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32, requires_grad=False).to(device)

    new_log_probs, value_preds, entropy = policy.eval_actions(
        state=states_tensor,
        actions=actions_tensor
        )
    
    advantages_tensor = advantages_tensor / (advantages_tensor.std() + 1e-8)

    actor_loss = -(advantages_tensor * new_log_probs).mean()
    critic_loss = torch.nn.functional.mse_loss(value_preds, returns_tensor)
    entropy_bonus = entropy.mean()

    loss = actor_loss + critic_coef * critic_loss - entropy_coef * entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 10 == 0:
        print()
        print(f"Episode {episode}, Return: {sum(rewards):.2f}, Loss: {loss.item():.2f}")


def evaluate_policy(model, episodes=5, device="cpu"):
    model.to(device)
    model.eval()
    total_reward = 0
    for _ in range(episodes):
        done = False
        state, info = env.reset()
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(state_tensor)
                action = torch.argmax(logits, dim=1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
    print(f"Average reward: {(avg := total_reward / episodes)}")

    if avg >= 100.:
        torch.save(obj=policy.state_dict(), f="models/lunar_lander_a2c_policy.pth")

evaluate_policy(policy)