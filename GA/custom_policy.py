from math import e
from os import fpathconf
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy

from GA import GA


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

        # Store the embedding of LTL task
        self.ltl_embedding = None

        # Env module
        self.image_embedder = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2),
            nn.ReLU(),
            nn.Flatten()
        )

        # LTL module
        # TODO: parameterize
        # self.ltl_embedder = nn.Embedding(11, 8, padding_idx=0)
        self.ltl_embedder = nn.Embedding(14, 16, padding_idx=0)
        self.ga = GA(seq_len=10, num_rel=8, num_heads=32, num_layers=3, device="cuda")


        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(288, 64), # 208
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, last_layer_dim_pi),
            # nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(288, 64), # 208
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, last_layer_dim_vf),
            # nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        batch_size = features.shape[0]

        # TODO: parameterize
        img = features[:, :147].reshape(batch_size,7,7,3).permute(0, 3, 1, 2)
        direction = features[:, 147]
        formula = features[:, 148:158].to(th.long)
        rels = features[:, 158:]
        rels = rels.reshape(batch_size, 10, 10).to(th.long)

        # Env module
        embedded_image = self.image_embedder(img) #128
        
        # LTL module
        embedded_formula = self.ltl_embedder(formula)
        embedded_formula = self.ga(embedded_formula, rels) # [B,10,8], [B,10,16]
        embedded_formula = embedded_formula.reshape(batch_size, -1) # [B,80], [B,160]

        # RL module
        composed_state = th.cat([embedded_image, embedded_formula], dim=1) # [B,208], [B,288]
        return self.policy_net(composed_state), self.value_net(composed_state)


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

