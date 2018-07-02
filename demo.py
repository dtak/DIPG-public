#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 02:20:55 2018

@author: arjumand
"""
#%%

#Policy Gradient Code has been adapted from: https://gist.github.com/nailo2c/09c3fd3a92fe212dea8f97ac5c7a1043
#Toy 2D Grid environment adapted from: https://github.com/dtak/hip-mdp-public/blob/master/grid_simulator/grid.py
# MMD based regularization is the main contribution

import numpy as np



import matplotlib.pyplot as plt
import dipg as DIPG #Diversity-Inducing Policy Gradient
import toy_2d_domain as toy_env

#Experiment Settings
max_episode_length = 40
min_num_episodes = 500

#Algorithm Parameters
gamma = 0.99 #Discount Factor
mmd_alpha = 0.2 #DIPG Regularization 
n_policies = 4


#Environment
env = toy_env.Grid()
env.reset()
state = env.observe()


#Run Algorithm
policies, states_policies, rewards_policies, terminal_states = DIPG.mmd_multi_policy(env, mmd_alpha = mmd_alpha, n_policies = n_policies, max_episodes = min_num_episodes, max_episode_length = max_episode_length)                        


# Function to indicate which goal the episode terminated in

goals = env.goal
goal_radii = env.goal_radius
xlim = env.x_range
ylim = env.y_range

def which_goal(state):
    dist = []
    for goal, goal_radius in zip(goals, goal_radii):
        dist.append(np.linalg.norm(np.array(state) - goal) - goal_radius)
    if np.min(dist) < 0:
        return np.argmin(dist)
    else:
        return None
    
    
#State Visitation, Goals reached etc for Plotting/Analysis
all_states = []

for states_ep, rewards_ep in zip(states_policies, rewards_policies):
    for states, rewards in zip(states_ep, rewards_ep):
        all_states.append(states)

goals_reached = []
for terminal_state in terminal_states:
    goal_id = which_goal(terminal_state)
    if goal_id is not None:
        goals_reached.append(goal_id)
            


theta = np.linspace(0,2*np.pi)
plt.figure()
for goal, goal_radius in zip(goals, goal_radii): 
    
    x_circle = goal_radius*np.sin(theta) + goal[0]
    y_circle = goal_radius*np.cos(theta) + goal[1]
    plt.plot(x_circle, y_circle,'ko',markersize = 1)

plt.axis([xlim[0],xlim[1],ylim[0],ylim[1]])


for path, count in zip(all_states, range(len(all_states))):
    path = np.array(path)
    if count == 0:
        plt.plot(path[:,0], path[:,1], 'bo--', markersize = 3)
    else:
        plt.plot(path[:,0], path[:,1], 'bo--', markersize = 3)


plt.title('DIPG' + ' Performance')
plt.tight_layout()
plt.savefig('DIPG_Performance.png')

print('Number of Distinct Goals {}\n Number of Policies: {}'.format(
                        len(np.unique(goals_reached)),n_policies))

