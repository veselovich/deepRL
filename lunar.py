import gymnasium as gym

from huggingface_sb3 import load_from_hub

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


# Create the environment
env = make_vec_env('LunarLander-v3', n_envs=16)

# We added some parameters to accelerate the training
model = PPO(
    policy = 'MlpPolicy',
    env = env,
    n_steps = 1024,
    batch_size = 64,
    n_epochs = 4,
    gamma = 0.999,
    gae_lambda = 0.98,
    ent_coef = 0.01,
    verbose=1)

timesteps = 1000000

model.learn(total_timesteps=timesteps)
# Save the model
model_name = "ppo-LunarLander-v3"
model.save(model_name)


eval_env = Monitor(gym.make("LunarLander-v3", render_mode='rgb_array'))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

if mean_reward - std_reward > 200:

    env_id = "LunarLander-v3"
    model_architecture = "PPO"
    repo_id = "veselovich/ppo-lunarlander-v3-roman" # Change with your repo id, you can't push with mine 😄
    commit_message = "Upload PPO LunarLander-v3 trained agent"
    filename = "ppo-LunarLander-v3.zip" # The model filename.zip