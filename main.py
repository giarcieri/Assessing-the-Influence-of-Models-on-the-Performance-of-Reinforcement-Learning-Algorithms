import os
import time
import argparse
import cProfile

from multiprocessing import Process

from MB_trainer import MB_Trainer

def main():
    """
        Function to set the parameters for the RL algorithm. 
        
        The function takes all the arguments and converts them to a dictionary params, which is passed to the class MB_trainer that initialize all the parameters, create the environment and the model to run, and starts the RL algorithm through run_training_loop().
        
        Arguments:
        ----------
        env_name: str
            Name of the environment to create
        ep_len: int
            lenght of each episode (steps for episode). No need to be specified since it is initialized depending on the env_name
        n_iter: int
            number of episode to run for each seed
        model: str
            name of the model to run, it is in ['deterministicNN', 'deterministic_ensemble', 'probabilisticNN', 'probabilistic_ensemble', 'dropoutNN', 'dropout_ensemble', 'gaussian_process']
        ensemble_size: int
            number of NNs in the ensemble
        mpc_horizon: int
            planning horizon of each trajectory in the random shooting algorithm 
        mpc_num_action_sequences: int
            number of different trajectories in the random shooting algorithm
        learning_rate: float
            learning rate of the NN
        n_layers: int
            number of hidden layers of the model
        n_hidden: int
            number of units/neurons of each hidden layer
        activation_in: str
            activation function of hidden layers: for 'deterministicNN' and 'deterministic_ensemble' choose 'relu' or 'elu', 
                                                  for 'probabilisticNN' and 'probabilistic_ensemble' choose 'relu', 'tanh' or 'sigmoid',
                                                  for 'dropoutNN' and 'dropout_ensemble' choose 'relu'
        epochs: int
            epochs of training
        replay_buffer: int
            max lenght of the dataset (replay_memory) the model is trained on
        seed: int
            number of the seed to start
        nb_seeds: int
             number of different seeds to run
        """

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str) 
    parser.add_argument('--ep_len', type=int, default=200) 
    parser.add_argument('--n_iter', '-n', type=int, default=20)
    
    parser.add_argument('--model1', type=str)
    parser.add_argument('--model2', type=str)
    parser.add_argument('--ensemble_size', '-e', type=int, default=5)
    parser.add_argument('--mpc_horizon', type=int, default=20)
    parser.add_argument('--mpc_num_action_sequences', type=int, default=500)

    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--n_hidden', '-s', type=int, default=40) 
    parser.add_argument('--activation_in', type=str) 
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--replay_buffer', type=int, default=2000)

    parser.add_argument('--seed', type=int, default=1)
    #parser.add_argument('--nb_seeds', type=int, default=1) 

    args = parser.parse_args() 

    # convert to dictionary
    params = vars(args)


    ###################
    ### RUN TRAINING
    ###################
    
    trainer1 = MB_Trainer(params)
    trainer2 = MB_Trainer(params)
    trainer3 = MB_Trainer(params)
    trainer4 = MB_Trainer(params)
    trainer5 = MB_Trainer(params)
    
    t1 = Process(target=trainer1.run_training_loop, args=(42, params['model1']))
    t2 = Process(target=trainer2.run_training_loop, args=(43, params['model1']))
    t3 = Process(target=trainer3.run_training_loop, args=(44, params['model1']))
    t4 = Process(target=trainer4.run_training_loop, args=(45, params['model1']))
    t5 = Process(target=trainer5.run_training_loop, args=(46, params['model1']))
    
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    
    if params['model2'] is not None:
        trainer6 = MB_Trainer(params)
        trainer7 = MB_Trainer(params)
        trainer8 = MB_Trainer(params)
        trainer9 = MB_Trainer(params)
        trainer10 = MB_Trainer(params)
        
        t6 = Process(target=trainer6.run_training_loop, args=(42, params['model2']))
        t7 = Process(target=trainer7.run_training_loop, args=(43, params['model2']))
        t8 = Process(target=trainer8.run_training_loop, args=(44, params['model2']))
        t9 = Process(target=trainer9.run_training_loop, args=(45, params['model2']))
        t10 = Process(target=trainer10.run_training_loop, args=(46, params['model2']))
        
        t6.start()
        t7.start()
        t8.start()
        t9.start()
        t10.start()
    
    #trainer = MB_Trainer(params)
    #trainer.run_training_loop()


if __name__ == "__main__":
    #pr = cProfile.Profile()
    #pr.enable()
    main()
    #pr.disable()
    #pr.print_stats()