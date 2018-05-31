#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 02:16:27 2018

@author: arjumand

"""
#Adapted from https://gist.github.com/nailo2c/09c3fd3a92fe212dea8f97ac5c7a1043

import argparse
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()



state_dim = 2 
action_dim = 4

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(state_dim, 128)
        self.affine2 = nn.Linear(128, action_dim)

        self.saved_log_probs = []
        self.rewards = []
        self.states = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

def exp_kernel(x,y,h = 1.0, vec = True):
    if x.ndim == 1:
        x = x.reshape((1,-1))
    if y.ndim == 1:
        y = y.reshape((1,-1))
    return np.exp(-1.0*np.sum((x - y)**2,1)/h**2/2.0)


def mmd_multi_policy(env, n_policies = 4, mmd_alpha = 0.2, max_episodes = 100, max_episode_length = 20, test_episodes = 5):

    terminal_states = []
    known_trajectories_x1 = []
    known_trajectories_x2 = []

    
    policies = []
    
    def states_kernel():
        ic = len(policy.states)
        
        kernel_vals = []
        p1s = []
        p2s = []
        for policy_x1, policy_x2 in zip(known_trajectories_x1, known_trajectories_x2):
            ip = len(policy_x1)
            
            p1 = np.zeros((np.max([ip,ic]),2))
            p2 = np.zeros((np.max([ip,ic]),2))
            
            p1[:,0] = policy_x1[-1]
            p1[:,1] = policy_x2[-1]
        
            p2[:,0] = policy.states[-1][0]
            p2[:,1] = policy.states[-1][1]
            
            p1[0:len(policy_x1),0] = np.array(policy_x1)
            p1[0:len(policy_x2),1] = np.array(policy_x2)
            
            p2[0:len(policy.states),:] = np.array(policy.states)
            kernel_vals.append(exp_kernel(p1.flatten(), p2.flatten())/len(known_trajectories_x1))
            
            p1s.append(p1)
            p2s.append(p2)
            
        
        return p1s,p2s, kernel_vals

    def select_action(state, epsilon = 0):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        probs = policy(state)
        m = Categorical(probs)
        action = m.sample()
        policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def finish_episode():
        R = 0
        policy_loss = []
        rewards = []
        for r in policy.rewards[::-1]:
            R = r + args.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        
            
        for log_prob, reward in zip(policy.saved_log_probs, rewards):
            policy_loss.append(-log_prob*reward)
        if len(known_trajectories_x1) > 0:
            p1s, p2s, kernels = states_kernel()
            idx = np.argmax(kernels)
            s1 = p1s[idx]
            s2 = p2s[idx]
            kss = kernels[idx]
            
            kernel_terms = np.exp(-1*np.sum((s1 - s2)**2,1))
            kernel_list = []
            for kt in kernel_terms:
                kernel_list.append(torch.tensor(kt))
                
            kernel_list = torch.tensor(kernel_list)
            kernel_list = (kernel_list - kernel_list.mean())/(kernel_list.std() + eps)
            for log_prob, k_term in zip(policy.saved_log_probs, kernel_list):        
                policy_loss.append(log_prob*(2-2*k_term)*mmd_alpha)
            
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
    
        del policy.rewards[:]
        del policy.saved_log_probs[:]
    
    states_policies = []
    rewards_policies = []
    
    N_REWARD_LIST = []
    for i in range(n_policies):
        
        policy = Policy()
        optimizer = optim.Adam(policy.parameters(), lr=1e-2)
        eps = np.finfo(np.float32).eps.item()
        
        running_reward = 0
        KD_running = 0
        
        REWARD_LIST = []
        for i_episode in range(max_episodes):
            env.reset()
            state = env.observe()
            policy.states.append(state)
            reward_list = []
            for t in range(max_episode_length): 
                action = select_action(state)                
                state, reward, done = env.perform_action(action)
                policy.states.append(state)
                policy.rewards.append(reward)
                reward_list.append(reward)
                if done:
                    break
            running_reward = (i_episode*running_reward  + reward)/(i_episode + 1)
            finish_episode()
            REWARD_LIST.append(reward_list)
            if i_episode % args.log_interval == 0:
                if len(known_trajectories_x1) > 0:
                    p1s, p2s, kernels = states_kernel()
                    idx = np.argmax(kernels)
                    s1 = p1s[idx]
                    s2 = p2s[idx]
                    kss = kernels[idx]
                    kernel_mean = np.mean( np.exp(-1*np.sum((s1 - s2)**2,1)))
                    KD_running = 0.95*KD_running + 0.05*kernel_mean
                    print('Episode {}\tLast length: {:5d}\tAverage award: {:.2f}\tAverage Kernel: {:.2f}'.format(
                        i_episode, t, running_reward, kernel_mean))
                else:
                    print('Episode {}\tLast length: {:5d}\tAverage award: {:.2f}\tNo Kernel'.format(
                        i_episode, t, running_reward ))
    
        
            if i_episode > max_episodes:
                break
            del policy.states[:]
        
        N_REWARD_LIST.append(REWARD_LIST)
        
    
        states_list_ep = []
        rewards_list_ep = []
        

        for i in range(test_episodes):
            x1 = []
            x2 = [] 
            env.reset()
            state = env.observe()
            x1.append(state[0])
            x2.append(state[1])
            
            states_i = []
            rewards_i = []
            for t in range(max_episode_length):  # Don't infinite loop while learning
                action = select_action(state)
                states_i.append(state)
                state, reward, done = env.perform_action(action)
                rewards_i.append(reward)
                x1.append(state[0])
                x2.append(state[1])
                policy.rewards.append(reward)
                if done:
                    terminal_states.append(state)
                    break

        states_list_ep.append(states_i)
        rewards_list_ep.append(rewards_i)
        
        policies.append(policy)
        known_trajectories_x1.append(np.array(x1))
        known_trajectories_x2.append(np.array(x2))
        
        states_policies.append(states_list_ep)
        rewards_policies.append(rewards_list_ep)
        
    return policies, states_policies, rewards_policies, terminal_states

