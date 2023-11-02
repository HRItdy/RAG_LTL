# RAG_LTL
This project is used to embed the ltl tasks. This repo is built using [bebbo-LTL2Action](https://github.com/bebbo203/LTL2Action) and [LTL2Action](https://github.com/LTL2Action/LTL2Action)

The repository implements training the embedding of LTL tasks through reinforcement learning executed on the LTLbootcamp environment.
More details can be found in the **[Report](https://github.com/bebbo203/LTL2Action/blob/main/report.pdf)**. 

## Installation

### Requirements
* [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/)
* [gym-minigrid](https://github.com/maximecb/gym-minigrid)

or, for better compatibility, the following command can be used:

### Set up the conda environment
```
conda env create -f environment.yml
```

## How to Use

### Pretrain the embedding 

```
python pretrain.py
```

## References
- Vaezipoor, Pashootan, Li, Andrew, Icarte, Rodrigo Toro, and McIlraith, Sheila (2021). “LTL2Action:Generalizing LTL Instructions for Multi-Task RL”. In:International Conference on MachineLearning (ICML)
- Icarte, Rodrigo Toro, Klassen, Toryn Q., Valenzano, Richard Anthony, and McIlraith, Sheila A.(2018). “Teaching Multiple Tasks to an RL Agent using LTL.” In:International Conferenceon Autonomous Agents and Multiagent. Systems (AAMAS), (pp. 452–461)

## TODO
- Write code to evaluate the similarity among ltl tasks and evaluate the learnt embedding model.
- Construct the LLM model using langchain.
- - 1. Construct the model to generate similar ltl tasks and corresponding behaviors, and save them into possbible errors-revised task pairs.
- - 2. Construct the embedding and use the retrieved content to construct the prompt.
- - 3. Construct groundtruth dataset.
- - 4. How to build up robotic experiments to evaluate?
