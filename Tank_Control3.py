import math
import random
import numpy as np
from scipy.integrate import odeint
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt

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
  
# 1. NN: 2 layers with 24 units each in ReLU + 1 layer with 4 units each in softmax + loss function as policy gradient

class PolicyNetwork():
    
    def __init__(self,state_list,action_list, n_hidden, lr):
        
        self.state_list = state_list
        self.action_list = action_list

        n_state = 1
        n_action= len(action_list)

        self.model = nn.Sequential(
            nn.Linear(n_state,n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_action),
            nn.Softmax(dim=0),
        )

        self.optimizer = torch.optim.SGD(self.model.parameters(),lr)

    def predict(self,s):
        return self.model(torch.Tensor([s]))
    
    def update(self,returns,log_probs):

        policy_gradient = []
        
        for log_prob, Gt in zip(log_probs, returns):
            policy_gradient.append(log_prob * Gt)
        loss = torch.stack(policy_gradient, dim=0).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self,s): # NEEDS Modification

        probs = self.predict(s)
        action_index = torch.multinomial(probs,1).item()

        log_prob = torch.log(probs[action_index])
        action = self.action_list[action_index]

        return action, log_prob
    
    def save(self):
        FILE = "model.pth"
        torch.save(self.model.state_dict(), FILE)


# 3. Runcode that makes an episode and gives it to NN to train.

class PolicyGradient(Tank): 

    def __init__(self,state_list, action_list, start_state, terminal_state, discount_rate): 
        
        "Inherits class Tank to use ODE_Solver"
        "Makes dictionaries of state and action indicies"

        Tank.__init__(self)
        self.state_list = np.round(state_list, decimals = 1)
        self.action_list = np.round(action_list, decimals = 1)
        self.start_state = start_state
        self.terminal_state = terminal_state
        self.discount_rate = discount_rate

    def next_state_transition(self,state,action,t): 
         
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

        if action > 1.5:
            reward = - (10 - state) ** 2 - 6 * (action) ** 2 
        else:
            reward = - (10 - state) ** 2

        if t >= 20:
            reward += -100
        else:
            pass

        return next_state, reward

    def reinforce(self, NN_model, n_episode, gamma):
        return_episode = [0] * n_episode
        print(NN_model.predict(13))
        for episode in range(n_episode):
            log_probs = []
            rewards = []
            actions = []
            state = 13

            t = 0

            while True:
                action, log_prob = NN_model.get_action(state)
                actions.append(action)
                next_state, reward = self.next_state_transition(state, action, t)
                t += 1

                return_episode[episode] += reward
                log_probs.append(log_prob)
                rewards.append(reward)
                # print(f"t")

                if next_state == 10 or t >= 10:
                    returns = []
                    Gt = 0
                    pw = 0

                    for reward in rewards[::-1]:
                        Gt = gamma ** pw * reward
                        pw += 1
                        returns.append(Gt)
                    
                    returns = torch.Tensor(returns[::-1])
                    # returns = (returns - returns.mean()/(returns.std() + 1e-9))
                    NN_model.update(returns, log_probs)
                    print(f"Episode: {episode+1} | Return: {return_episode[episode]}")
                    print(f"PROB: {NN_model.predict(13)}")
                    print(f"actions taken: {actions}")
                    break

                state = next_state

        return return_episode
    
a = PolicyNetwork(state_list=np.linspace(5,15,101),
                  action_list=np.linspace(0,3,7),
                  n_hidden=50,
                  lr=0.00001)

b = PolicyGradient(state_list=np.linspace(5,15,101),
                         action_list=np.linspace(0,3,7),
                         start_state=13,
                         terminal_state=10,
                         discount_rate=1)

return_episode = b.reinforce(NN_model=a,
                             n_episode=7,
                             gamma=1)

# # 4. Plot 2 graphs: Loss graph vs episode, mean squared error vs episode

df1 = pd.DataFrame(return_episode)
with pd.ExcelWriter('output.xlsx') as excel_writer:
    df1.to_excel(excel_writer, sheet_name='Sheet1', index=False)

plt.plot(return_episode)
plt.show()

# 5. test model


