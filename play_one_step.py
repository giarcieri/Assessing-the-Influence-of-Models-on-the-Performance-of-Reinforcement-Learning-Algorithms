import numpy as np

def play_one_step(replay_memory, state, epsilon, episode, horizon, num_paths, obs_space, action_space, model, epistemic_uncertainty, ensemble, env, cost_fn):
    """
    Function to compute one step of the loop.

    The action of the current step is chosen by using the epsilon_greedy_policy function. The agent takes the action in the environment and observes the next state, the reward, if done and further info. The reward is computed by using our own cost functions. Then all the information of the current step is collected to the dataset.

    Arguments:
    ----------
    replay_memory: list
        Dataset of finite buffer, used to append all the information
    state:
        current state, it is passed to epsilon_greedy_policy
    epsilon: float
        current epsilon, it is passed to epsilon_greedy_policy to decide whether to explore or exploit
    episode: int
        current episode, it is passed to epsilon_greedy_policy
    horizon: int
        horizon of random shooting, it is passed to epsilon_greedy_policy
    num_paths: int
        number of trajectories of random shooting, it is passed to epsilon_greedy_policy
    obs_space: int
        observation space to reshape the observation array
    action_space: int
        action space to reshape the action array
    model:
        model to be exploited in the optimization algorithm (random shooting + MPC), it is passed to epsilon_greedy_policy
    epistemic_uncertainty: bool
        boolean to decide how to explore the environment (random exploration or Information Gain exploration), it is passed to epsilon_greedy_policy
    ensemble: bool
        boolean for compatibility of different functions, it is passed to epsilon_greedy_policy
    env:
        environment the agent interacts with, it is passed to epsilon_greedy_policy
    cost_fn:
        cost function of the environment, it is passed to epsilon_greedy_policy
    Returns:
    --------
    replay_memory: list
        The updated dataset
    next_state:
        the new state after the agent takes the action
    reward:
        the reward the agent receives for taking that action
    done: bool
        whether the episode is over or not
    info:
        further information
    """
    action = epsilon_greedy_policy(state, epsilon, episode, horizon, num_paths, model, epistemic_uncertainty, ensemble, env, cost_fn)
    action = action.reshape(action_space, 1) 
    next_state, _, done, info = env.step(action)
    next_state = next_state.reshape(obs_space,)
    reward = cost_fn(state.reshape(1, obs_space), action.reshape(1, action_space), env)
    replay_memory.append((state, action, reward, next_state, done))
    return replay_memory, next_state, reward, done, info

def epsilon_greedy_policy(state, epsilon, episode, horizon, num_paths, model, epistemic_uncertainty, ensemble, env, cost_fn):
    """
    Policy of the agent to select the action, deciding between exploration and exploitation.

    A random float in [0,1] is generated, if the number is less than epsilon (i.e. with probability epsilon) the agent explores, otherwise exploits the model through MPC and random shooting (get_action). If the model explores, then if the model can quantify its uncertainty and it is trained -> explores maximizing the Information Gain, otherwise explores randomly.

    Arguments:
    ----------
    state:
        current state, it is used both for exploration and exploitation
    epsilon: float
        current epsilon, probability of exploration
    episode: int
        current episode, to know whether the model is trained
    horizon: int
        horizon of random shooting, it is passed to get_action
    num_paths: int
        number of trajectories of random shooting, it is passed to get_action
    model:
        model to be used both for exploration and exploitation
    epistemic_uncertainty: bool
        boolean to decide how to explore the environment, then it is passed to get_action for code compatibility of the different models
    ensemble: bool
        boolean for compatibility of different functions, it is passed to get_action and IG_exploration
    env:
        environment the agent interacts with
    cost_fn:
        cost function of the environment, it is passed to get_action

    Returns:
    --------
    IG_exploration(): 
        action from IG exploration
    env.action_space.sample()[:, np.newaxis]:
        action from random exploration
    get_action():
        action from MPC + RS
    """
    if np.random.rand() < epsilon:
        if epistemic_uncertainty and episode > 0:
            return IG_exploration(state, model, ensemble, env) # IG exploration
        else: 
            return env.action_space.sample()[:, np.newaxis] # epsilon greedy exploration
    else:
        return get_action(state, horizon, num_paths, model, epistemic_uncertainty, ensemble, env, cost_fn) # MPC + random shooting
    
def get_action(state, horizon, num_paths, model, epistemic_uncertainty, ensemble, env, cost_fn):
    """
    Optimization algorithms to exploit the model, it combines Model Predictive Control (MPC) and random shooting.

    Several trajectories, with finite horizon, are computed in parallel. For each trajectory, a random action is sampled and the model predicts the next state, given this action and the current state, until the horizon lenght. All the trajectories are evaluated with the trajectory_cost_fn function, then the first action [0] of the best trajectory [j] is selected.

    Arguments:
    ----------
    state:
        current state, it is used both for exploration and exploitation
    horizon:
        horizon of random shooting
    num_paths:
        number of trajectories of random shooting
    model:
        model to predict the next states for each trajectory
    epistemic_uncertainty:
        boolean for code compatibility of the different models
    ensemble:
        boolean for code compatibility of the different models
    env:
        environment the agent interacts with, used to sample the actions
    cost_fn:
        cost function of the environment, it is passed to trajectory_cost_fn

    Returns:
    --------
    act_list[0][j]: 
        First action of the best trajectory
    """
    obs, obs_list, act_list = [], [], []
    [obs.append(state) for _ in range(num_paths)]
    for _ in range(horizon):
        obs_list.append(obs)
        actions = []
        [actions.append(env.action_space.sample()) for _ in range(num_paths)]
        act_list.append(actions)
        data = np.concatenate((np.array(obs), np.array(actions)), axis=1)
        if ensemble:
            delta_obs, delta_obs_total, delta_obs_std = model.predict_ensemble(data) # for dropout_ens, delta_obs_std is logvar but we don't need it
        elif epistemic_uncertainty:
            delta_obs, delta_obs_total, delta_obs_var = model.predict(data) # for GP delta_obs_total is nan but we don't need it
        else:
            delta_obs = model.predict(data)
        obs = delta_obs + np.array(obs)
    trajectory_cost_list = trajectory_cost_fn(cost_fn, np.array(obs_list), np.array(act_list), env)
    j = np.argmax(trajectory_cost_list)
    return act_list[0][j]

def trajectory_cost_fn(cost_fn, states, actions, env):
    """
    Function to compute the total rewards for all the trajectories

    It takes two lists as inputs: the list of all the states for all the trajectories, and the list of all the actions for all the trajectories. It computes in parallel the total rewards of the trajectories, over all the steps i of the horizon planning.

    Arguments:
    ----------
    cost_fn: 
        Cost function of the environment, it computes the cost of the current state and action, then returns -cost since the environment returns rewards, not costs
    states: list
        list of all the states of all the trajectories
    actions: list
        list of all the actions of all the trajectories

    Returns:
    --------
    trajectory_cost: array
        array which contains the total rewards for each trajectory, It has num_path shape.
    """
    trajectory_cost = 0
    for i in range(len(actions)):
        trajectory_cost += cost_fn(states[i], actions[i], env) 
    return trajectory_cost

def IG_exploration(state, model, ensemble, env):
    """
    Exploration policy based on Information Gain.

    This function defines the exploration for the models which can quantify the epistemic uncertainty. A certain number of action (here 50, but it can be set) is sampled, then the model predicts the next state, given the current state, for each different action. In addition, the model then quantify the epistemic uncertainty for each prediction. Since the epistemic uncertainty is calculated as a variance of each dimension of each state, can be directly sum up over the observation space of each prediction, in order to have a comparable value among the different states. Finally, it is selected the action with the highest uncertainty: in this way, the agent will see the true next state for that action, the tuple will be appended to the dataset the model will be trained on, so it will no longer be uncertain about it. 
    This policy allows to visit areas of the environment where the model is uncertain only, without exploring areas it is very certain about.

    Arguments:
    ----------
    state: 
        the current state to compute the next states for each action
    model:
        the model to predict the next states
    ensemble: bool
        boolen for code compatibility
    env:
        environment, it is used to sample the actions

    Returns:
    --------
    actions[j]: array 1D
        action with the highest epistemic uncertainty
    """
    actions = []
    obs = []
    [actions.append(env.action_space.sample()) for _ in range(50)]
    [obs.append(state) for _ in range(50)]
    data = np.concatenate((np.array(obs), np.array(actions)), axis=1)
    if ensemble:
        delta_obs, delta_obs_total, delta_obs_std = model.predict_ensemble(data) # for dropout_ens is logvar but we don't need it
        epistemic_uncertainty = model.uncertainty().sum(1)
    else:
        delta_obs, delta_obs_total, delta_obs_var = model.predict(data) # for GP delta_obs_total is nan but we don't need it
        epistemic_uncertainty = np.sum(model.uncertainty(), 1)
    j = np.argmax(epistemic_uncertainty)
    return actions[j]
        
            
            
            
            
            
            
            
            
            
        