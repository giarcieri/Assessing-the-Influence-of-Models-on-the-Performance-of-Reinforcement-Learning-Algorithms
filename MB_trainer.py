import os
import time
import gym
import pybulletgym
import numpy as np
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
mpl.rc('animation', html='jshtml')

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

from collections import deque

from training_step import training_step, data_training
from play_one_step import play_one_step, epsilon_greedy_policy, get_action, trajectory_cost_fn, IG_exploration

from cost_functions import pendulum_cost_fn, half_cheetah_cost_fn, inverted_pendulum_cost_fn, reacher_cost_fn, double_inverted_pendulum_cost_fn, hopper_cost_fn, walker2D_cost_fn, ant_cost_fn


class MB_Trainer:
    
    def __init__(self, params):
        """
        MB_trainer is the class where all the params are initialized and the RL alrgorithm is executed.
        
        Here the environment is created using self.env_name, then for each environment the episode length and the cost function are set.
        
        Arguments:
        ----------
        params['env_name']: str
            name of the environment to be created, it is in list ['Pendulum-v0', ..., 'AntMuJoCoEnv-v0']
        params['seed']: int
            seed number for the models
        params['model']: str 
            model to run, it is in list ['deterministicNN', 'deterministic_ensemble', 'probabilisticNN', 'probabilistic_ensemble', 'dropoutNN', 'dropout_ensemble', 'gaussian_process']
        params['n_layers']: int
            number of hidden layers of the model
        params['n_hidden']: int
            number of units/neurons of each hidden layer
        params['activation_in']: str
            activation function of hidden layers
        params['epochs']: int
            epochs of training
        params['learning_rate']: float
            learning rate of the NN
        params['ensemble_size']: int
            number of NNs in the ensemble (usually 5)
        params['replay_buffer']: int
            max lenght of the dataset (replay_memory), usually 2000
        Raises: 
        -------
        RuntimeError:
            If params['env_name'] is not in the expected list
        
        """
        self.params = params
        self.env_name = params['env_name']
        #self.seed = params['seed']                 # it is set for each process in main.py
        self.nb_layers = params['n_layers']
        self.n_hidden = params['n_hidden']
        self.activation_in = params['activation_in']
        self.epochs = params['epochs']
        self.lr = params['learning_rate']
        self.n_ensemble = params['ensemble_size']
        self.replay_buffer = params['replay_buffer']

        if self.env_name == 'Pendulum-v0':
            params['ep_len'] = 200
            cost_fn = pendulum_cost_fn
        elif self.env_name == 'HalfCheetahMuJoCoEnv-v0': 
            params['ep_len'] = 500
            cost_fn = half_cheetah_cost_fn
        elif self.env_name == 'InvertedPendulumMuJoCoEnv-v0':
            params['ep_len'] = 100
            cost_fn = inverted_pendulum_cost_fn
        elif self.env_name == 'ReacherPyBulletEnv-v0':
            params['ep_len'] = 50
            cost_fn = reacher_cost_fn
        elif self.env_name == 'InvertedDoublePendulumMuJoCoEnv-v0':
            params['ep_len'] = 100
            cost_fn = double_inverted_pendulum_cost_fn
        elif self.env_name == 'HopperMuJoCoEnv-v0':
            params['ep_len'] = 500
            cost_fn = hopper_cost_fn
        elif self.env_name == 'Walker2DMuJoCoEnv-v0':
            params['ep_len'] = 500
            cost_fn = walker2D_cost_fn
        elif self.env_name == 'AntMuJoCoEnv-v0':
            params['ep_len'] = 500
            cost_fn = ant_cost_fn
        else:
            raise RuntimeError
    
        env = gym.make(self.env_name)
        self.env = env
        self.ep_len = params['ep_len']
        self.cost_fn = cost_fn
        D = self.env.observation_space.shape[0] #output shape model
        A = self.env.action_space.shape[0]
        Q = D + A # input shape model
        self.D = D
        self.A = A
        self.Q = Q
    
    def run_training_loop(self, seed, model_name):
        
        """
        RL algorithm, the agent interacts with the environment, learns the dynamics function and collects rewards
        
        The function begins with importing all the keras/tensorflow related functions and initializing the model through self.model_name and the chosen hyperparameters. 
        Then two booleans (epistemic_uncertainty and ensemble) are defined to control the exploration and the compatibility of the RL algorithm for each different model. 
        This is made because this function will be run in parallel for different seeds, and because of a conflict between keras and multiprocessing, everything related to keras/tensorflow is to be defined within each process.
        Afterwards, the RL algorithm starts. Each seed iteration presents many episode iterations (between 50 and 500) and the length of each episode depends on the enviroment. 
        Core of this function is play_one_step that computes one step and it collects new information wich is appended to the dataset. 
        At each step, the agent chooses to expore with probability epsilon or to exploit the model (1 - epsilon). 
        Epsilon starts at 1 in the first episode (only exploration), then it remains constant at 0.01. 
        At the end of each episode, the model is trained on the whole dataset, which keeps the last 2000 (can be set with params['replay_buffer']) data only.
        
        Arguments:
        ----------
        self.env: 
            environment
        model_name: str 
            model to run, it is in list ['deterministicNN', 'deterministic_ensemble', 'probabilisticNN', 'probabilistic_ensemble', 'dropoutNN', 'dropout_ensemble', 'gaussian_process']
        self.nb_layers: int
            number of hidden layers of the model
        self.n_hidden: int
            number of units/neurons of each hidden layer
        self.activation_in: str
            activation function of hidden layers
        self.epochs: int
            epochs of training
        self.lr: float
            learning rate of the NN
        self.n_ensemble: int
            number of NNs in the ensemble (usually 5)
        self.replay_buffer: int
            max lenght of the dataset (replay_memory), usually 2000
        seed: int
            random seed for each parallel process
            
        Raises: 
        -------
        RuntimeError:
            If model_name is not in the expected list
        
        """
        
        ##########################
        ### Model Initialization
        ##########################
        
        import tensorflow as tf
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
          try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
          except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
        
        #from tensorflow import keras # it is much slower than import keras alone
        import keras
        import gpflow
        from gpflow.ci_utils import ci_niter
        
        #import tensorflow.keras.backend as K
        #from tensorflow.keras import initializers
        #from tensorflow.keras.models import Model, Sequential
        #from tensorflow.keras.layers import Input, Dense, Lambda, Wrapper, concatenate, InputSpec
        
        from models.deterministicNN import NN, ens_NNs
        from models.PNN import PNN, ens_PNNs
        from models.ConcreteDropout import ConcreteDropout, BNN, ens_BNN
        from models.GP import GP

        keras.backend.clear_session()
        np.random.seed(seed)
        tf.random.set_seed(seed)

        if model_name == 'deterministicNN':
            model = NN(env = self.env, nb_layers = self.nb_layers, n_hidden = self.n_hidden, activation_in = self.activation_in, 
                 kernel_initializer="he_normal", epochs = self.epochs, l_rate = self.lr, env_name = self.env_name)
            epistemic_uncertainty = False
            ensemble = False
        elif model_name == 'deterministic_ensemble':
            model = ens_NNs(env = self.env, nb_layers  = self.nb_layers, n_hidden = self.n_hidden, activation_in = self.activation_in,
                            kernel_initializer="he_normal", epochs = self.epochs, l_rate = self.lr, n_ensemble = self.n_ensemble,
                            env_name = self.env_name)
            epistemic_uncertainty = True
            ensemble = True
        elif model_name == 'probabilisticNN':
            model = PNN(env = self.env, reg = 'anc', n_hidden = self.n_hidden, activation_in = 'relu', data_noise = 0.001, 
                        n_data = self.replay_buffer, 
                        epochs = self.epochs, l_rate = self.lr, env_name = self.env_name)
            epistemic_uncertainty = False
            ensemble = False
        elif model_name == 'probabilistic_ensemble':
            model = ens_PNNs(env = self.env, reg = 'anc', n_hidden = self.n_hidden, activation_in = 'relu', data_noise = 0.001, 
                             n_data = self.replay_buffer, 
                             epochs = self.epochs, l_rate = self.lr, n_ensemble = self.n_ensemble, env_name = self.env_name)
            epistemic_uncertainty = True
            ensemble = True
        elif model_name == 'dropoutNN':
            model = BNN(env = self.env, nb_units = self.n_hidden, nb_layers = self.nb_layers, activation = 'relu', 
                        n_data = self.replay_buffer,
                        dropout = 0.1, T = 20, tau = 1.0, lengthscale = 1e-4, train_flag = True, env_name = self.env_name)
            epistemic_uncertainty = True
            ensemble = False
        elif model_name == 'dropout_ensemble':
            model = ens_BNN(env = self.env, nb_units = self.n_hidden, nb_layers = self.nb_layers, activation = 'relu', 
                            n_data = self.replay_buffer, 
                            dropout = 0.1, T = 20, tau = 1.0, lengthscale = 1e-4, n_ensemble = self.n_ensemble, train_flag = True, 
                            env_name = self.env_name)
            epistemic_uncertainty = True
            ensemble = True
        elif model_name == 'gaussian_process':
            gpflow.config.set_default_float(np.float64)
            model = GP(env = self.env, gp_model='GPR')
            epistemic_uncertainty = True
            ensemble = False
        else:
            raise RuntimeError

        self.model = model
        self.epistemic_uncertainty = epistemic_uncertainty
        self.ensemble = ensemble
        
        #####################
        ### RL TRAINING LOOP
        #####################
        
        replay_memory = deque(maxlen=self.params['replay_buffer'])
        self.env.seed(seed)
        seed_rewards = [] # total rewards for each different seed
        print('Starting training loop for seed', seed)
        for episode in range(self.params['n_iter']):
            episode_rewards = 0
            state = self.env.reset().reshape(self.D,)
            for step in range(self.ep_len): 
                epsilon = max(1 - episode / 1, 0.01)
                replay_memory, state, reward, done, info = play_one_step(replay_memory, state, epsilon, episode,
                                                                         horizon = self.params['mpc_horizon'], 
                                                                         num_paths = self.params['mpc_num_action_sequences'],
                                                                         obs_space = self.D,
                                                                         action_space = self.A, 
                                                                         model = self.model,
                                                                         epistemic_uncertainty = self.epistemic_uncertainty,
                                                                         ensemble = self.ensemble,
                                                                         env = self.env, 
                                                                         cost_fn = self.cost_fn) 
                state = state.reshape(self.D,)
                episode_rewards += reward
                #if done:
                #    break
            seed_rewards.append(episode_rewards)
            print("\rEpisode: {}, Steps: {}, rewards: {}, model: {}, seed: {}, eps: {:.3f}".format(episode, step + 1, episode_rewards, 
                                                                                           model_name, seed, epsilon), end="")
            self.model = training_step(replay_memory, model = self.model, 
                                       ensemble = self.ensemble, obs_space = self.D, action_space = self.A) 
            
        file = 'rewards_' + model_name + '_' + self.env_name + '_' +'seed'+str(seed)+'_' + time.strftime("%d-%m-%Y")+ '.pickle'
        with open(file, "wb") as fp:
            pickle.dump(seed_rewards, fp)
        self.env.close()

                    
        
        
        
        
        
        
        
        
        
        
        
        
        
