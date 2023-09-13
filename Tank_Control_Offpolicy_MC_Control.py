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
        # uncertainty = np.random.normal(0,0.5,1)
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
    
    def episode_generator(self, policy_table): 
         
        "Produces an episode (state, action, reward) based on policy"

        t=0
        state = self.start_state 
        state_history = []
        epsilon = 1e-6

        while abs(state - self.terminal_state) > epsilon:
            
            stateindex = self.stateindex_dict[state]
            action = random.choices(self.action_list, policy_table[stateindex])[0] 
            
            if action > 1.5:
                reward = - (10 - state) ** 2 - 6 * (action) ** 2 
            else:
                reward = - (10 - state) ** 2

            if t >= 20:
                reward += -100
                state_history.append([state, action, reward])
                break
            else:
                pass

            state_history.append([state, action, reward])
            next_state = self.next_state_generator(state,action,t) 
            state = next_state
            t=t+1
                
        state_history = np.array(state_history) 
        return state_history

    def maxpolicy_generator(self,q_table,policy_table,state): 
         
        "Improves policy table at a state based on action value table" 
    
        if state != self.terminal_state:

            stateindex = self.stateindex_dict[state] # Finds row index of a table for a state
            policy_table[stateindex,:] = 0 # Initialize Policy for that state
            q_max = np.max(q_table[stateindex]) # Finds maximum q value in the state
            columnforq_max_list = np.argwhere(q_table[stateindex] == q_max)
            probability = np.divide(1,len(columnforq_max_list))
            policy_table[stateindex,columnforq_max_list] = probability
        
        else:
            pass

        return policy_table
    
    def policy_imporvement(self,q_table,policy_table):

        "Improves policy table based on action value table"

        for state in self.state_list:
            policy_table = self.maxpolicy_generator(q_table,policy_table,state)

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

    def optimalepisode_generator(self, policy_table): 
         
        "Produces an episode (state, action, reward) based on policy"

        t=0
        state = self.start_state
        state_history = [] 
        epsilon = 1e-6

        while abs(state - self.terminal_state) > epsilon:
            
            stateindex = self.stateindex_dict[state]
            action = random.choices(self.action_list, policy_table[stateindex])[0] 

            state_history.append([state])
            next_state = self.next_state_generator(state,action,t) 
            state = next_state
            t=t+1

            if t >= 1000:
                return state_history  
            else:
                pass

        state_history = np.array(state_history) 
        return state_history

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
 
    def Offpolicy_GPI(self, q_table, c_table, episode, targetpolicy_table, behaviourpolicy_table, epsilongreedy: bool):
        
        "Performs Monte Carlo Methods to find optimal action value and policy"
        
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
            
            targetpolicy_table = self.maxpolicy_generator(q_table,targetpolicy_table,state)
            
            if targetpolicy_table[stateindex,actionindex] == 0: 
                break
            else: 
                pass
            
            W = W / behaviourpolicy_table[stateindex,actionindex]
            
            if epsilongreedy == True:
                behaviourpolicy_table = self.epsilongreedypolicy_generator(state, q_table, policy_table = behaviourpolicy_table, epsilon=0.1) 
            else:
                pass
    
        return q_table, behaviourpolicy_table, targetpolicy_table
    
    def training(self,num_episode):
        q_table, c_table, targetpolicy_table, behaviourpolicy_table = self.Tables_Initialization()
        num_state_visited = []
        count = []

        for i in range(num_episode):
            episode = self.episode_generator(policy_table=behaviourpolicy_table)
            q_table, behaviourpolicy_table, targetpolicy_table = self.Offpolicy_GPI(q_table, 
                                                                                    c_table, 
                                                                                    episode,
                                                                                    targetpolicy_table, 
                                                                                    behaviourpolicy_table, 
                                                                                    epsilongreedy= False)
        
            # Find how many state visited til termination
            state_history = self.optimalepisode_generator(policy_table=targetpolicy_table)
            num_state_visited.append(len(state_history))
            count.append(i)


            if i % 100 == 0:
                print(f"Episode: {i} | Time required: {len(state_history)}")
            else:
                pass

        # Convert each dataset dictionary into a DataFrame
        df1 = pd.DataFrame(q_table,index=a.state_list, columns=a.action_list)
        df2 = pd.DataFrame(behaviourpolicy_table,index=a.state_list, columns=a.action_list)
        df3 = pd.DataFrame(targetpolicy_table,index=a.state_list, columns=a.action_list)

        # Write each DataFrame to a different sheet in the Excel file
        with pd.ExcelWriter('output.xlsx') as excel_writer:
            df1.to_excel(excel_writer, sheet_name='q_table')
            df2.to_excel(excel_writer, sheet_name='behaviourpolicy_table')
            df3.to_excel(excel_writer, sheet_name='targetpolicy_table')

        return count, num_state_visited


a = Offpolicy_MCcontrol(state_list=np.linspace(5,15,101),
                        action_list=np.linspace(0,3,7),
                        start_state=13,
                        terminal_state=10,
                        discount_rate=1)

count, num_state_visited = a.training(num_episode=1000)

plt.plot(count, num_state_visited)
plt.title("Performance")
plt.ylabel("Time required to reach to 10m")
plt.xlabel("Episodes")
plt.show()





