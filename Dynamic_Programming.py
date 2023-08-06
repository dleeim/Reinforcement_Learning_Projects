import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

class Dynamic_Programming():

    def __init__(self,state_table,terminal_state,policy_table,v_table,r_table,discount_factor):

        self.terminal_state = terminal_state
        self.policy_table = policy_table
        self.v_table = v_table
        self.next_state_v_table = np.zeros((np.shape(r_table)))
        self.r_table = r_table
        self.discount_factor = discount_factor

        self.state_table = state_table
        self.state_axis = {}

        for i in np.nditer(self.state_table):
            if not np.isnan(i):
                row, column = np.where(self.state_table == i)
                self.state_axis[i.item()] = (row.item(), column.item())

    def next_state_generator(self,state,action):
        
        (row,column) = self.state_axis[state]
        row_max, column_max = np.shape(self.state_table)[0]-1, np.shape(self.state_table)[1]-1
        
        if action == 0 and row > 0 :
            next_state = self.state_table[row - 1, column]
        elif action == 1 and row < row_max:
            next_state = self.state_table[row + 1, column]
        elif action == 2 and column < column_max:
            next_state = self.state_table[row, column + 1]
        elif action == 3 and column > 0:
            next_state = self.state_table[row, column - 1]
        else:
            next_state = state 

        if np.isnan(next_state):
            next_state = state
        
        return next_state

    def Bellman_Equation(self,state,action):

        next_state = self.next_state_generator(state.item(),action)
        (next_row,next_column) = self.state_axis[next_state]
        new_state_value = self.policy_table[state,action] * (self.r_table[state,action] + self.discount_factor  * self.v_table[next_row,next_column])
    
        return new_state_value

    def policy_evaluation(self):
        threshold = 0.01
        next_v_table = np.zeros(np.shape(self.state_table))
        delta = 1
        while delta > threshold:
            delta = 0

            for state in np.nditer(self.state_table):

                if np.all(state != self.terminal_state):
                    (row,column) = self.state_axis[state.item()]
                        
                    for action in range(4):
                        next_v_table[row,column] += self.Bellman_Equation(state,action)

                    delta = np.maximum(delta, np.absolute(self.v_table[row,column] \
                                                                - next_v_table[row,column]))

            self.v_table = np.copy(next_v_table)
            next_v_table = np.zeros((4,4))
       
        self.v_table = np.around(self.v_table,2)

        print(f"value table: \n {self.v_table}") 
        return self.v_table

    def Bellman_Optimality_Equation(self,state,action,next_row,next_column):
        
        next_v = (self.r_table[state,action] + self.discount_factor \
                    * self.v_table[next_row,next_column])
        
        return next_v

    def value_max_evaulation(self):
        threshold = 0.01
        next_v_table = np.zeros((4,4))
        delta = 1

        while delta > threshold:
            delta = 0

            for state in np.nditer(self.state_table):
                next_v_max = -100

                if np.all(state != self.terminal_state):
                    (row,column) = self.state_axis[state.item()]

                    for action in range(4):
                        next_state = self.next_state_generator(state.item(),action)
                        (next_row,next_column) = self.state_axis[next_state]
                        next_v = self.Bellman_Optimality_Equation(state,action,next_row,next_column)
                        next_v_max = np.maximum(next_v_max,next_v)
                    
                    next_v_table[row,column] = next_v_max

                    delta = np.maximum(delta, np.absolute(self.v_table[row,column] - next_v_table[row,column]))

            self.v_table = np.copy(next_v_table)
            next_v_table = np.zeros((4,4))

        self.v_table = np.around(self.v_table,2)

        print(f"value table: \n {self.v_table}") 
        return self.v_table
    
    def policy_improvement(self):
        self.policy_table[:] = 0

        for state in np.nditer(self.state_table):

            if np.all(state != self.terminal_state):
                (row,column) = self.state_axis[state.item()]
                
                for action in range(4):
                    next_state = self.next_state_generator(state.item(),action)
                    (next_row,next_column) = self.state_axis[next_state]
                    self.next_state_v_table[state,action] = self.v_table[next_row,next_column]

                next_state_v_max = np.max(self.next_state_v_table[state])
                action_max = np.argwhere(self.next_state_v_table[state] == next_state_v_max)
                probability = np.divide(1,len(action_max))

                self.policy_table[state,action_max] = probability

        print(f"policy table: \n {self.policy_table}") 
        return self.policy_table
    
    def Iteration(self,dopolicyiteration,dovalueiteration):

        if dopolicyiteration == True:
            policy_stable = False

            while policy_stable == False:
                old_policy_table = np.copy(self.policy_table)
                self.policy_evaluation()
                self.policy_improvement()

                if np.all(old_policy_table == self.policy_table):
                    policy_stable = True
                else:
                    pass

        elif dovalueiteration == True:
            self.value_max_evaulation()
            self.policy_improvement()
        else:
            pass





# Policy Iteration and Value Iteration Run Code
state_table = np.array([[0,1,2,3],
                        [4,5,6,7],
                        [8,9,10,11],
                        [12,13,14,15]])
terminal_state = np.array([0,15])
policy_table = np.full((16,4),0.25)
v_table = np.zeros((4,4))
r_table = np.full((16,4), -1)
discount_factor = 1
a = Dynamic_Programming(state_table,terminal_state,policy_table,v_table,r_table,discount_factor)

# You could do Policy Iteration or Value Iteration by changing the boolean (True, False):
a.Iteration(dopolicyiteration=False,dovalueiteration=True)

# # If you want to just do policy evaluation for policy iteration or value iteration use the following code:
# a.policy_evaluation()
# a.value_max_evaulation()



## From here, the code is for making diagram
# Pandas
a.v_table = pd.DataFrame(a.v_table)

# Matplotlib 
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(8, 4))

# Hide the axes
ax1.axis('off')
ax1.axis('tight')

# Create table for state value
table = ax1.table(cellText=a.v_table.values, loc='center', cellLoc = 'center')
table.scale(1, 4)

# Matplotlib diagram for policy
ax2.yaxis.set_visible(False)
ax2.xaxis.set_visible(False)

## Draw table
for i in range(a.v_table.shape[0]+1):
    ax2.axhline(y=-i,color='k', linestyle='-')
for i in range(a.v_table.shape[1]+1):
    ax2.axvline(x=i,color='k', linestyle='-')

## Draw arrows for optimum policy
optimalstatelist, optimalactionlist = np.nonzero(a.policy_table)
for i in range(len(optimalstatelist)):
        optimalstate = optimalstatelist[i]
        optimalaction = optimalactionlist[i]

        if np.all(optimalstate != np.array(a.terminal_state)):
            (row,column) = a.state_axis[optimalstate]
            if optimalaction == 0:
                ax2.arrow(x=column+0.5, y=-row+0.5, dx=0, dy = 0.2, width = 0.05 ,color='red')
            elif optimalaction == 1:
                ax2.arrow(x=column+0.5, y=-row+0.5, dx=0, dy = -0.2, width = 0.05 ,color='red')
            elif optimalaction == 2:
                ax2.arrow(x=column+0.5, y=-row+0.5, dx=0.2, dy = 0, width = 0.05 ,color='red')
            elif optimalaction == 3:
                ax2.arrow(x=column+0.5, y=-row+0.5, dx=-0.2, dy = 0, width = 0.05 ,color='red')
            else:
                pass

ax2.set_xlim(0, 4)
ax2.set_ylim(1, -3)
ax2.invert_yaxis()

# Adjust spacing between subplots

# Display the figure
ax1.title.set_text('State Value')
ax2.title.set_text('Policy')

plt.show()

