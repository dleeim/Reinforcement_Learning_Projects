import math
import random
import numpy as np
from scipy.integrate import odeint
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

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
        uncertainty = np.random.normal(0,0.5,1)
        uncertainty = 0
        h = odeint(self.differential_eq, h_initial, t, args = (V_in, uncertainty))[1]
        h = np.round(h, decimals = 1)

        return h
    
    
class MonteCarloAgent(Tank): 
    def __init__(self,state_list, action_list, start_state, terminal_state, discount_rate): 
        
        "Inherits class Tank to use ODE_Solver"
        "Makes dictionaries of state and action indicies"

        Tank.__init__(self)
        self.state_list = np.round(state_list, decimals = 1)
        self.action_list = np.round(action_list, decimals = 1)
        self.start_state = start_state
        self.terminal_state = terminal_state
        self.discount_rate = discount_rate

        self.stateindex_dict = {} 
        i=0
        for state in self.state_list:
            self.stateindex_dict[state] = (i) 
            i = i + 1
    
        self.actionindex_dict = {} 
        i=0
        for action in self.action_list:
            self.actionindex_dict[action] = (i)
            i = i + 1

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

    def episode_generator(self,policy_table): 
         
        "Produces an episode based on policy"

        t=0
        state = self.start_state 
        state_history = []
        epsilon = 1e-6

        while abs(state - self.terminal_state) > epsilon:
            
            "There can be a mistake in using next_state_generator"
            stateindex = self.stateindex_dict[state]
            action = random.choices(self.action_list, policy_table[stateindex])[0] 
            if action > 1.5:
                reward = - (10 - state) ** 2 - 6 * (action) ** 2 
            else:
                reward = - (10 - state) ** 2
            state_history.append([state, action, reward])
            
            next_state = self.next_state_generator(state,action,t) 
            state = next_state
            t=t+1

        state_history = np.array(state_history) 
        return state_history

    def optmialepisode_generator(self,q_table): 
        
        "Produces an optimal episode based on policy"

        t=0
        state = self.start_state 
        state_history = np.array([state])

        while state != self.terminal_state: # Condition needs to be done like this for comparison between float numbers
            
            state_index = self.stateindex_dict[state]
            q_max = np.max(q_table[state_index],axis=None)
            maxaction_indices = np.where(q_table[state_index] == q_max)[0]
            maxaction_index = np.random.choice(maxaction_indices)


            for i, j in self.actionindex_dict.items(): 
                if j == maxaction_index:
                    episode_action = i

            next_state = self.next_state_generator(state,episode_action,t) 
            state = next_state
            t = t + 1
            state_history = np.append(state_history,state)
        
        return state_history 
    
    def policyargumentmaximum(self,q_table,policy_table,state): 
         
        "Improves policy table at a state based on action value table" 
        
        if state != self.terminal_state:

            stateindex = self.stateindex_dict[state] # Finds row index of a table for a state
            policy_table[stateindex,:] = 0 # Initialize Policy for that state
            q_max = np.max(q_table[stateindex])
            columnindicesthathasqmax_list = np.argwhere(q_table[stateindex] == q_max)
            probability = np.divide(1,len(columnindicesthathasqmax_list))
            policy_table[stateindex,columnindicesthathasqmax_list] = probability

        return policy_table

    def policy_imporvement(self,q_table,policy_table):

        "Improves policy table based on action value table"

        for state in self.state_list:
            policy_table = self.policyargumentmaximum(q_table,policy_table,state)
        
        return policy_table 
    
    def epsilongreedypolicy_generator(self,state,q_table,policy_table,epsilon):

        "Makes the policy at a state to be epsilon - greedy"

        state_index = self.stateindex_dict[state]
        q_max = np.max(q_table[state_index],axis=None)
        maxaction_indices = np.where(q_table[state_index] == q_max)[0]

        for action in self.action_list:
            action_index = self.actionindex_dict[action]
        
            if np.any(maxaction_indices == action_index): 
                policy_table[state_index,action_index] = (1 - epsilon)/len(maxaction_indices) + epsilon/len(self.action_list) 
            else:
                policy_table[state_index,action_index] = epsilon/len(self.action_list) 
        
        return policy_table
    
class Offpolicy_MCcontrol(MonteCarloAgent):
    
    def __init__(self,state_list, action_list, start_state, terminal_state, discount_rate):
        MonteCarloAgent.__init__(self,state_list, action_list, start_state, terminal_state, discount_rate) 
    
    def Tables_Initialization(self):
        q_table = np.full((self.state_list.shape[0],self.action_list.shape[0]),-300,dtype=float)
        q_table[self.stateindex_dict[self.terminal_state]] = 0
        
        c_table = np.zeros((self.state_list.shape[0],self.action_list.shape[0]),dtype=float)
        
        targetpolicy_table = np.zeros((self.state_list.shape[0],self.action_list.shape[0]))
        targetpolicy_table = self.policy_imporvement(q_table, policy_table = targetpolicy_table)
        
        randomprobability = 1/self.action_list.shape[0]
        behaviourpolicy_table = np.full((self.state_list.shape[0],self.action_list.shape[0]),randomprobability)
        
        return q_table, c_table, targetpolicy_table, behaviourpolicy_table

    def Offpolicy_GPI(self, numberofepisode, epsilongreedy: bool):
        
        "Performs Monte Carlo Methods to find optimal action value and policy"
        
        q_table, c_table, targetpolicy_table, behaviourpolicy_table = self.Tables_Initialization() 
        
        for i in range(1,numberofepisode+1):
    
            episode = self.episode_generator(policy_table = behaviourpolicy_table) 
            G=0
            W=1
        
            for event in np.flip(episode, axis = 0):

                state = event[0]
                action = event[1]
                reward = event[2]

                stateindex = self.stateindex_dict[state]
                actionindex = self.actionindex_dict[action]

                G = self.discount_rate * G + reward
                c_table[stateindex,actionindex] = c_table[stateindex,actionindex] + W
                q_table[stateindex,actionindex] = q_table[stateindex,actionindex] + \
                    (W/c_table[stateindex,actionindex]) * (G - q_table[stateindex,actionindex])
                
                targetpolicy_table = self.policyargumentmaximum(q_table,targetpolicy_table,state)
                
                if targetpolicy_table[stateindex,actionindex] == 0: 
                    break
                else: 
                    pass
                
                W = W / behaviourpolicy_table[stateindex,actionindex]
                
                if epsilongreedy == True:
                    behaviourpolicy_table = self.epsilongreedypolicy_generator(state, q_table, policy_table = behaviourpolicy_table, epsilon=0.1) 
                else:
                    pass
            
            print(f"\rProgress: {i/numberofepisode*100} %", end="")
        print(f"\n")
        
        return q_table, behaviourpolicy_table, targetpolicy_table

# Runcode for Off Policy Method with random behaviour policy
state_list= np.linspace(5,15,101)
action_list= np.linspace(0,3,7)
start_state = 13
terminal_state= 10
discount_rate = 1
a = Offpolicy_MCcontrol(state_list= np.linspace(5,15,101), action_list= np.linspace(0,3,7), start_state = 13, terminal_state= 10, discount_rate = 1)
numberofepisode_list = [25,50,75,100]
offpolicy_notgreedy_list = []
offpolicy_greedy_list = []
std_offpolicy_notgreedy_list = []
std_offpolicy_greedy_list = []
meanq_offpolicy_notgreedy_list = []
meanq_offpolicy_greedy_list = []

for numberofepisode in numberofepisode_list: 

    # For offpolicy being random
    print("Offpolicy without greedy")
    mean=0
    mean_q = 0
    var = 0
    for i in range(1,31):
    
        q_table, behaviourpolicy_table, targetpolicy_table = a.Offpolicy_GPI(numberofepisode,epsilongreedy=False)
        optimalepisode_list = a.optmialepisode_generator(q_table)
        total_state_visited = len(optimalepisode_list)
        oldmean = np.copy(mean)
        mean = (i-1)/(i) * mean + 1/i * (total_state_visited)
        var = 1/i * ((i-1) * var + (total_state_visited-mean) * (total_state_visited-oldmean))
        mean_q = (i-1)/(i) * q_table[a.stateindex_dict[13],0] + 1/i * (mean_q)

    offpolicy_notgreedy_list.append(mean)
    std_offpolicy_notgreedy_list.append(var ** 0.5)
    meanq_offpolicy_notgreedy_list.append(mean_q)

    # For offpolicy being greedy
    print("Off policy with greedy") 
    mean = 0
    mean_q = 0
    var = 0
    for i in range(1,31):
        q_table, behaviourpolicy_table, targetpolicy_table = a.Offpolicy_GPI(numberofepisode,epsilongreedy=True)
        optimalepisode_list = a.optmialepisode_generator(q_table)
        total_state_visited = len(optimalepisode_list)
        oldmean = np.copy(mean)
        mean = (i-1)/(i) * mean + 1/i * (total_state_visited)
        var = 1/i * ((i-1) * var + (total_state_visited-mean) * (total_state_visited-oldmean))
        mean_q = (i-1)/(i) * q_table[a.stateindex_dict[13],0] + 1/i * (mean_q)
        
    offpolicy_greedy_list.append(mean)
    std_offpolicy_greedy_list.append(var ** 0.5)
    meanq_offpolicy_greedy_list.append(mean_q)

plt.figure(1)
plt.plot(numberofepisode_list, offpolicy_notgreedy_list,'bo-',label='Off policy without greedy')
plt.plot(numberofepisode_list, offpolicy_greedy_list, 'ro-',label='Off policy with greedy')

plt.errorbar(numberofepisode_list, offpolicy_notgreedy_list, yerr=std_offpolicy_notgreedy_list, linestyle='None', capsize=5, color='blue')
plt.errorbar(numberofepisode_list, offpolicy_greedy_list, yerr=std_offpolicy_greedy_list, linestyle='None', capsize=5, color='red')

plt.title("Average number of state visited vs Number of episode created")
plt.xlabel("Number of episodes created")
plt.ylabel("Average number of state visited")
plt.legend()

plt.figure(2)
plt.plot(numberofepisode_list, meanq_offpolicy_notgreedy_list,'bo-',label='Off policy without greedy')
plt.plot(numberofepisode_list, meanq_offpolicy_greedy_list, 'ro-',label='Off policy with greedy')

plt.title("Mean Action value for start state = 13 vs Number of episode created")
plt.xlabel("Number of episodes created")
plt.ylabel("Mean Action value for start state = 13")
plt.legend()

plt.show() 
