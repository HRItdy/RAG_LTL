
import os
import time
import torch

from envs.adversarial import AdversarialEnv9x9
from envs.myopic import AdversarialMyopicEnv9x9

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


MANUAL_CONTROL = False
MYOPIC = False
MODEL = "ga"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def control(letter):
    ''' Helper function to manually control the agent. '''
    if letter == 'a':   return 0 # sx
    elif letter == 'd': return 1 # dx
    elif letter == 'w': return 2 # forward
    else:               return 3 # no move


env = AdversarialEnv9x9() if not MYOPIC else AdversarialMyopicEnv9x9()
model = PPO.load(os.path.join("logs", MODEL), device=DEVICE) if not MANUAL_CONTROL else None


# Evaluation #
mean_rew, std_rew = evaluate_policy(model.policy, Monitor(env),
                                    n_eval_episodes=100,
                                    render=False,
                                    deterministic=False)
print(f"Mean reward: {mean_rew:.2f} +/- {std_rew:.2f}")
exit()


obs = env.reset()
for i in range(10000):
    action = control(input()) if MANUAL_CONTROL else model.predict(obs)[0]
    obs, rew, done, _ = env.step(action)
    env.render()
    time.sleep(.25)
    if done:
        env.reset()

