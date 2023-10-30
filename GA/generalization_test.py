
import torch

from envs.adversarial import AdversarialEnv9x9

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


env = AdversarialEnv9x9()
model = PPO.load("./logs/generalization_ga", device=DEVICE)

# evaluate generalization
mean_rew, std_rew = evaluate_policy(model.policy, Monitor(env),
                                    n_eval_episodes=100,
                                    render=False,
                                    deterministic=False)
print(f"Mean reward: {mean_rew:.2f} +/- {std_rew:.2f}")

