#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 10:22:26 2018

@author: arjumand
"""


import numpy as np


#Toy 2D Grid environment adapted from: https://github.com/dtak/hip-mdp-public/blob/master/grid_simulator/grid.py

goals = [[5.0, 0],
         [-5.0,0],
         [0,5.0],
         [0,-5.0]]

goal_radii = [1.0,1.0,1.0,1.0]
step_size = 0.4

class Grid(object):
    """
    This is a 2D Grid environment for simple RL tasks    
    """

    def __init__(self, start_state = [0,0], step_size = step_size, **kw):
        """
        Initialize the environment: creating the box, setting a start and goal region
        Later -- might include other obstacles etc
        """  
        
        self.num_actions = 4
        self.x_range = [-7,7]
        self.y_range = [-7,7]
#        self.goal_radius = 0.5
#        self.goal = [1.0,0]
        self.reset(start_state, step_size,**kw)
        
    def reset(self, start_state = [0,0], step_size = step_size, goal = goals, goal_radius = goal_radii, x_std = 0, y_std = 0, **kw):
        """
        Reset Environment
        """
        self.t = 0
        self.step_size = step_size
        self.start_state = start_state
        self.state = start_state
        self.goal_radius = goal_radius
        self.goal = goal
        self.x_std = x_std
        self.y_std = y_std
        
    def observe(self):
        return self.state
    
    def get_action_effect(self, action):
        """
        Set the effect direction of the action -- actual movement will involve step size and possibly involve action error
        """
        if action == 0:
            return [1,0]
        elif action == 1:
            return [-1,0]
        elif action == 2:
            return [0,1]
        elif action == 3:
            return [0,-1]
        
    def get_next_state(self, state, action):
        """
        Take action from state, and return the next state
        
        """
        action_effect = self.get_action_effect(action)
        new_x = state[0] +  (self.step_size * action_effect[0]) + np.random.normal(0, self.x_std)
        new_y = state[1] + (self.step_size * action_effect[1]) + np.random.normal(0, self.y_std)
        
        
        next_state = [new_x, new_y]
        return next_state
    
    def _valid_crossing(self, state=None, next_state=None, action = None):
        if state is None:
            state = self.state
            action = self.action
        if next_state is None:
            next_state = self.get_next_state(state, action)
            
        #Check for moving out of box in x direction 
        if next_state[0] < np.min(self.x_range) or next_state[0] > np.max(self.x_range):
#            print "Going out of x bounds"
            return False
        elif next_state[1] < np.min(self.y_range) or next_state[1] > np.max(self.y_range):
#            print "Going out of y bounds"
            return False
        else:
            return True
        
    def _in_goal(self, state = None):
        if state is None:
            state = self.state  
        each_goal = []
        for goal_i, radius_i in zip(self.goal, self.goal_radius):
            if (np.linalg.norm(np.array(state) - np.array(goal_i)) < radius_i):
                each_goal.append(1)
            else:
                each_goal.append(0)
        if np.sum(each_goal) >= 1:
            return True
        else:
            return False
            
                
    def calc_reward(self, state = None, action = None, next_state = None, **kw):
        if state is None:
            state = self.state
            action = self.action
        if next_state is None:
            next_state = self.get_next_state(state,action)
        if self._valid_crossing(state = state, next_state = next_state, action = action) and self._in_goal(state = next_state):
            return 1000
        elif self._valid_crossing(state = state, next_state = next_state, action = action) and not self._in_goal(state = next_state):
            return -0.1
        else: 
            return -5
                
    def perform_action(self, action, **kw):
        self.t += 1
        self.action = action
        reward = self.calc_reward()
        if self._valid_crossing():
            self.state = self.get_next_state(self.state, action)
        return self.observe(), reward, self._in_goal()
    
            


