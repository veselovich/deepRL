import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

env = gym.make('LunarLander-v3')

model = PPO(policy = 'MlpPolicy',
    env = env,
    n_steps = 1024,
    batch_size = 64,
    n_epochs = 4,
    gamma = 0.999,
    gae_lambda = 0.98,
    ent_coef = 0.01,
    verbose=1)
model.learn(1500000)

def evaluate2(model, env_name='LunarLander-v3', episodes=10, verbose=False):

    env = gym.make(env_name, render_mode="human")
    total_rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            # Используем model.predict(), а не напрямую policy(state)
            action, _ = model.predict(state, deterministic=True)

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


evaluate2(model, verbose=True)

model_name = "ppo-LunarLander-v3"
model.save(model_name)