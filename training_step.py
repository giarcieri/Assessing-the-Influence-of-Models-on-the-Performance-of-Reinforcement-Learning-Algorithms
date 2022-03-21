import numpy as np

def data_training(replay_memory):
    """
    Function to extract and prepare the information from the replay buffer for the model training
    
    Arguments:
    ----------
    replay_memory: 
        Dataset with finite buffer 

    Returns:
    --------
    states: array
        array containing all the states visited in the last steps, according to the buffer lenght.
    actions: array
        array containing the action taken at each state
    rewards: array
        reward received for each tuple state-action
    next_states: array
        next states resulted from each tuple state-action
    dones: bool
        whether or not the episode is over (it is a superfluous information since the lenght is set in main.py).
    """
    states, actions, rewards, next_states, dones = np.array(replay_memory)[:,0], np.array(replay_memory)[:,1], np.array(replay_memory)[:,2], np.array(replay_memory)[:,3], np.array(replay_memory)[:,4]
    return states, actions, rewards, next_states, dones

def training_step(replay_memory, model, ensemble, obs_space, action_space):
    """
    Function to train the model.

    The relevant information are extracted from the replay buffer, states and actions are the inputs to predict the next states. 
    The model is not trained to predict the next states directly, but rather the difference from the next state and the current state: in this way the performance has been proved to be more stable (Deisenroth, 2017).

    Arguments:
    ----------
    replay_memory: 
        dataset with finite buffer, it is passed to data_training
    model:
        model to be trained
    ensemble: bool
        boolean for code compatibility
    action_space:
        observation space of the environment to reshape the array
    obs_space:
        action space of the environment to reshape the array

    Returns:
    --------
    model: 
        model after training
    """
    states, actions, rewards, next_states, dones = data_training(replay_memory)
    batch_size = states.shape[0]
    states = np.concatenate(states)
    states = states.reshape(batch_size, obs_space)
    actions = np.concatenate(actions).reshape(batch_size, action_space)
    data = np.concatenate((states, actions), axis=1)
    next_states = np.concatenate(next_states)
    next_states = next_states.reshape(batch_size, obs_space)
    delta_states = next_states - states
    if ensemble:
        model.train(data, delta_states)
    else:
        model.fit(data, delta_states)
    return model 
    