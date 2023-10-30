import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import gym
import numpy as np

from stable_baselines3.common import base_class, logger
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.utils import safe_mean


class CustomCallback(EvalCallback):

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        min_ep_rew_mean: float, # new param
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str = None,
        best_model_save_path: str = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 0,
        warn: bool = True,
    ):
        super().__init__(eval_env, callback_on_new_best, n_eval_episodes,
                         eval_freq, log_path, best_model_save_path,
                         deterministic, render, verbose, warn)

        # minimum mean_reward to trigger evaluation callback
        self.min_ep_rew_mean = min_ep_rew_mean


    def _on_step(self) -> bool:

        if(
            self.eval_freq > 0 and
            self.n_calls % self.eval_freq == 0 and
            safe_mean([ep_info['r'] for ep_info in self.model.ep_info_buffer]) >= self.min_ep_rew_mean
        ):

            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # # Save success log if present
                # if len(self._is_success_buffer) > 0:
                #     self.evaluations_successes.append(self._is_success_buffer)
                #     kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            # success_rate = len(self._is_success_buffer) / float(self.n_eval_episodes)
            # if self.verbose > 0:
            #     print(f"Success rate: {100 * success_rate:.2f}%")
            # self.logger.record("eval/success_rate", success_rate)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True

