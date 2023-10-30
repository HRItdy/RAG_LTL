
import gym
import torch as th

from custom_policy import CustomActorCriticPolicy
from envs.adversarial import AdversarialEnv9x9
from envs.myopic import AdversarialMyopicEnv9x9

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from customcallback import CustomCallback


MYOPIC = False
DEVICE = "cuda" if th.cuda.is_available() else "cpu"


env = AdversarialEnv9x9() if not MYOPIC else AdversarialMyopicEnv9x9()
model = PPO(CustomActorCriticPolicy, env, verbose=1, tensorboard_log="./tensorboard", device=DEVICE)

# Load pre-trained weights for the LTL module
model.policy.mlp_extractor.ltl_embedder.load_state_dict(th.load("./pre_logs/weights_ltl.pt"))
model.policy.mlp_extractor.ga.load_state_dict(th.load("./pre_logs/weights_ga.pt"))

# Callback function to save the best model
eval_callback = CustomCallback(Monitor(env), min_ep_rew_mean=.9, n_eval_episodes=20,
                               best_model_save_path='./logs/', log_path='./logs/',
                               eval_freq=int(1e4), verbose=1, render=False)

# Training
model.learn(int(4e6), callback=eval_callback)

# Evaluation
mean_rew, std_rew = evaluate_policy(model.policy, Monitor(env),
                                    n_eval_episodes=25,
                                    render=False,
                                    deterministic=False)
print(f"Mean reward: {mean_rew:.2f} +/- {std_rew:.2f}")

