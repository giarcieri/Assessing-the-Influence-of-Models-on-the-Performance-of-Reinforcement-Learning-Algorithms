import numpy as np
import math
import gym
import pybullet

def pendulum_cost_fn(state, action, env):
    def angle_normalize(x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)
    cos_theta = state[:, 0]
    thdot = state[:, 2]
    cos_theta = np.clip(cos_theta, -1, 1)
    th = np.arccos(cos_theta)
    action = action.reshape(th.shape)
    costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(action**2)
    return -costs


def half_cheetah_cost_fn(state, action, env):
    assert state.shape[1] == env.observation_space.shape[0]
    assert action.shape[1] == env.action_space.shape[0]
    reward_ctrl = -0.1 * np.square(action).sum(1)
    reward_run = state[:, 8] 
    reward = reward_run + reward_ctrl
    return reward

def inverted_pendulum_cost_fn(state, action, env):
    assert state.shape[1] == env.observation_space.shape[0]
    assert action.shape[1] == env.action_space.shape[0]
    #data_dict = {'start_state': state, 'action': action}
    xpos_ob_pos = 0
    ypos_ob_pos = 1
    ypos_target = 0.0
    xpos_coeff = 0.0
    # xpos penalty
    xpos = state[:, xpos_ob_pos]
    xpos_reward = -(xpos ** 2) * xpos_coeff

    # ypos penalty
    ypos = state[:, ypos_ob_pos]
    ypos_reward = -(ypos - ypos_target) ** 2
    return xpos_reward + ypos_reward

def reacher_cost_fn(state, action, env):
    assert state.shape[1] == env.observation_space.shape[0]
    assert action.shape[1] == env.action_space.shape[0]
    distance = state[:, [2, 3]]
    reward_dist = - np.linalg.norm(distance, axis = 1)
    reward_ctrl = - np.square(action).sum(1)
    return reward_dist + reward_ctrl

def double_inverted_pendulum_cost_fn(state, action, env):
    assert state.shape[1] == env.observation_space.shape[0]
    assert action.shape[1] == env.action_space.shape[0]
    theta_sin = state[:, 1]
    theta_sin = np.clip(theta_sin, -1, 1)
    theta = np.arcsin(theta_sin)
    gamma_sin = state[:, 2]
    gamma_sin = np.clip(gamma_sin, -1, 1)
    gamma = np.arcsin(gamma_sin)
    theta_dot = state[:, 6]
    gamma_dot = state[:, 7]
    vel_penalty = 1e-3 * theta_dot**2 + 5e-3 * gamma_dot**2
    return -(theta**2) - (gamma**2) - vel_penalty

def hopper_cost_fn(state, action, env):
    assert state.shape[1] == env.observation_space.shape[0]
    assert state.shape[1] == 11
    assert action.shape[1] == env.action_space.shape[0]
    reward_ctrl = -0.1 * np.square(action).sum(1)
    reward_run = state[:, 5] 
    reward_height = -3*(state[:, 0])**2
    reward = reward_run + reward_ctrl + reward_height + 1
    return reward

def walker2D_cost_fn(state, action, env):
    assert state.shape[1] == env.observation_space.shape[0]
    assert action.shape[1] == env.action_space.shape[0]
    reward_ctrl = -0.1 * np.square(action).sum(1)
    reward_run = state[:, 8]
    reward_height = -3 * (state[:, 0])**2
    reward = reward_run + reward_ctrl + reward_height + 1
    return reward

def ant_cost_fn(state, action, env):
    assert state.shape[1] == 27
    assert action.shape[1] == 8
    reward_ctrl = -0.1 * np.square(action).sum(1)
    reward_run = state[:, 13]
    reward_height = -3 * (state[:, 0] - 0.75)**2
    reward = reward_run + reward_ctrl + reward_height + 1
    return reward
    
    

        

    
    
    
    
    
    
    
    
    