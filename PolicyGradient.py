import math
import random
import numpy as np
from scipy.integrate import odeint
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 0. gridworld environment

class Tank():

    def __init__(self):
        pass
    
    def differential_eq(self, h, t, V_in, uncertainty):

        "Differential equation that governs the tank"

        r = 0.5
        A = math.pi * r ** 2
        k = 1/10
        V_out = k * h + uncertainty
        dhdt = 1/A * (V_in - V_out)

        return dhdt

    def ODE_Solver(self, h_initial, V_in, t):

        "use odeint from scipy to solve differential eq for time after 1 seconds"

        t = np.linspace(t, t+1, 2)
        # uncertainty = np.random.normal(0,0.5,1)
        uncertainty = 0
        h = odeint(self.differential_eq, h_initial, t, args = (V_in, uncertainty))[1]
        h = np.round(h, decimals = 1)

        return h
    
# 1. agent: state, action, reward

class Agent(Tank): 

    def __init__(self,state_list, action_list, start_state, terminal_state, discount_rate): 
        
        "Inherits class Tank to use ODE_Solver"
        "Makes dictionaries of state and action indicies"

        Tank.__init__(self)
        self.state_list = np.round(state_list, decimals = 1)
        self.action_list = np.round(action_list, decimals = 1)
        self.start_state = start_state
        self.terminal_state = terminal_state
        self.discount_rate = discount_rate

    def next_state_generator(self,state,action,t): 
         
        "Uses ODE Solver to generate Next State"

        state_max, state_min = np.max(self.state_list), np.min(self.state_list)
        next_state = self.ODE_Solver(state,action,t)[0]

        # To prevent next_state to go over limit (bouncing back rule)
        if next_state > state_max: 
            next_state = state_max
        elif next_state < state_min: 
            next_state = state_min
        else: 
            pass

        return next_state

    
# 2. NN: 2 layers with 24 units each in ReLU + 1 layer with 4 units each in softmax + loss function as policy gradient

class NN_Model(nn.Module):
  
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(1,7)
    self.layer_2 = nn.Linear(7,7)
    self.layer_3 = nn.Linear(7,7)
    self.relu = nn.ReLU()

  def forward(self, x : torch.Tensor) -> torch.Tensor:
    out = self.relu(self.layer_1(x))
    out = self.relu(self.layer_2(out))
    out = nn.functional.softmax(self.layer_3(out), dim=0)
    return out

# a = NN_Model()
# state = 13
# episode = []
# policy_at_state = a.forward(torch.Tensor([state]))
# action = random.choices([1,2,3,4,5,6,7],policy_at_state)[0]
# print(action)

# 3. Runcode that makes an episode and gives it to NN to train.

class PolicyGradient_Agent(Agent, NN_Model):

    def __init__(self, state_list, action_list, start_state, terminal_state, discount_rate):
        Agent.__init__(self,state_list, action_list, start_state, terminal_state, discount_rate)
        NN_Model.__init__(self)

    def episode_generator(self): # Need to fix this
            
        "Produces an episode (state, action, reward) based on policy"

        t=0
        state = self.start_state 
        episode = []
        epsilon = 1e-6

        while abs(state - self.terminal_state) > epsilon:
            
            policy_at_state = self.forward(torch.Tensor([state]))
            action = random.choices(self.action_list,policy_at_state)[0] 
            
            if action > 1.5:
                reward = - (10 - state) ** 2 - 6 * (action) ** 2 
            else:
                reward = - (10 - state) ** 2

            if t >= 10:
                reward += -100
                episode.append([state, action, reward])
                break
            else:
                pass

            episode.append([state, action, reward])
            next_state = self.next_state_generator(state,action,t) 
            state = next_state
            t=t+1
                 
        return np.array(episode)

    def training(self, episode):
        return
        
a = PolicyGradient_Agent(state_list=np.linspace(5,15,101),
                         action_list=np.linspace(0,3,7),
                         start_state=13,
                         terminal_state=10,
                         discount_rate=1)

episode = a.episode_generator()
print(a.state_dict())
print(episode)

# 4. Plot 2 graphs: Loss graph vs episode, mean squared error vs episode