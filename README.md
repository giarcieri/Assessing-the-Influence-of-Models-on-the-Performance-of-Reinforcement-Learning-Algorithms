# Assessing the Influence of Models on the Performance of Reinforcement Learning Algorithms applied on Continuous Control Tasks


## Requirements

This project is tested on Python 3.6.

First, you can perform a minimal installation of OpenAI Gym with

```bash
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
```

Then, you can install Pybullet-Gym with 

```bash
git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip install -e .
```

*Important*: Do not use ```python setup.py install``` or other Pybullet-Gym installation methods.

Finally, you can install all the dependencies with 

```bash
pip install -r requirements.txt
```

## Project Description

### Models

The models are defined in the folder ```models```:

- ```deterministicNN.py```: it includes  the deterministic NN (```NN```) and the deterministic ensemble (```ens_NNs```).

- ```PNN.py```: here the Anchored Ensembling is defined following original author's [implementation](https://github.com/TeaPearce/Bayesian_NN_Ensembles). ```PNN``` defines one NN of the Anchored Ensembling. This is needed to define ```ens_PNNs``` which is the Anchored Ensembling.

- ```ConcreteDropout.py```: it defines the Concrete Dropout NN, mainly based on Yarin Gal's [implementation](https://github.com/yaringal/ConcreteDropout/blob/master/concrete-dropout-keras.ipynb). First, the ConcreteDropout Layer is defined, then the Concrete Dropout NN (```BNN```).

- ```GP.py```: it defines the Gaussian Process model based on [GPflow library](https://github.com/GPflow/GPflow). Two different versions are implemented: the ```GPR``` and the ```SVGP``` (choose by setting the parameter ```gp_model```). Only the ```GPR``` model is used in the paper.

### RL algorithm

The model performance is evaluated in the following files:

1) ```main.py```: it contains the function ```main``` which takes all the params that are passed to ```MB_trainer```. Five ```MB_trainer``` are initialized, each with a different seed, which are run in parallel. It is also possible to run two models in parallel by setting the param ```model2``` as well. 

2) ```MB_trainer.py```: it includes the initialization of the env and the model as well as the RL training loop. The function ```play_one_step``` computes one step of the loop. The model is trained with the function ```training_step```. At the end of the loop, a pickle file is saved, wich includes all the rewards achieved by the model in all the episodes of the env.

3) ```play_one_step.py```: it includes all the functions to compute one step (i.e. to choose one action): the epsilon greedy policy for the exploration, the Information Gain exploration, and the exploitation of the model with MPC+RS (function ```get_action```). The rewards as well as the RS trajectories are computed with the cost functions in ```cost_functions.py```.

4) ```training_step.py```: first the relevant information is prepared by the function ```data_training```, then the model is trained with the function ```training_step```.

5) ```cost_functions.py```: it includes all the cost functions of the envs. 

