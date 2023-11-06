
from math import e
from os import fpathconf
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from envs.ltl_bootcamp import LTLBootcamp

from customcallback import CustomCallback
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from GA import GA

DEVICE = "cuda" if th.cuda.is_available() else "cpu"



class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # LTL module
        #self.ltl_embedder = nn.Embedding(11, 16, padding_idx=0)
        self.ltl_embedder = nn.Embedding(14, 16, padding_idx=0)
        self.ga = GA(seq_len=10, num_rel=8, num_heads=32, num_layers=3, device=DEVICE)

        # Policy network
        self.policy_net = nn.Linear(160, last_layer_dim_pi)
        # Value network
        self.value_net = nn.Linear(160, last_layer_dim_vf)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        batch_size = features.shape[0]
        formula = features[:, :10].to(th.long)
        rels = features[:, 10:].reshape(batch_size, 10, 10).to(th.long)
        
        # LTL module
        embedded_formula = self.ltl_embedder(formula)
        embedded_formula = self.ga(embedded_formula, rels) # [B,10,16]
        embedded_formula = embedded_formula.reshape(batch_size, -1) # [B,160]
        self.ef = embedded_formula

        # RL module
        return self.policy_net(embedded_formula), self.value_net(embedded_formula)
    
    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(self.ef)



class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)




env = LTLBootcamp(samplenum=600)
model = PPO(CustomActorCriticPolicy, env, verbose=1, device=DEVICE)

# Training
model.learn(int(1e7))

# Evaluation
mean_rew, std_rew = evaluate_policy(model.policy, Monitor(env),
                                    n_eval_episodes=25,
                                    render=False,
                                    deterministic=False)
print(f"Mean reward: {mean_rew:.2f} +/- {std_rew:.2f}")

# Save LTL module pre-trained weights
th.save(model.policy.mlp_extractor.ltl_embedder.state_dict(), "./pre_logs/weights_ltl.pt")
th.save(model.policy.mlp_extractor.ga.state_dict(), "./pre_logs/weights_ga.pt")

