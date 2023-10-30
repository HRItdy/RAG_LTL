from math import e
from os import fpathconf
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy



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

        # # direction embedding
        # self.dir_embedding = nn.Embedding(4, 16)

        # LTL module
        self.ltl_embedder = nn.Embedding(13, 8, padding_idx=0)
        self.rnn = nn.GRU(8, 32, num_layers=2, bidirectional=True, batch_first=True)


        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, last_layer_dim_pi),
            # nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(192, 64),
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

        img = features[:, :147].reshape(batch_size,7,7,3).permute(0, 3, 1, 2)
        direction = features[:, 147]
        formula = features[:, 148:]

        # Env module
        embedded_image = self.image_embedder(img) #128
        # embedded_dir = self.dir_embedding(direction.to(th.long)) # [B,16]
        
        # LTL module
        embedded_formula = self.ltl_embedder(formula.to(th.long))
        _, h = self.rnn(embedded_formula)
        embedded_formula = h[:-2,:,:].transpose(0,1).reshape(batch_size, -1) # [B,64]

        # RL module
        composed_state = th.cat([embedded_image, embedded_formula], dim=1) # [B,192]
        # composed_state = th.cat([embedded_image, embedded_dir, embedded_formula], dim=1) # [B,192]
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

