import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

class PolicyNetwork():
    
    def __init__(self,n_state,n_action, n_hidden = 50, lr = 0.001):
        self.model = nn.Sequential(
            nn.Linear(n_state,n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_action),
            nn.Softmax(),
        )

        self.optimizer = torch.optim.SGD(self.model.parameters(),lr)

    def predict(self,s):
        return self.model(torch.Tensor(s))
    
    def update(self,returns,log_probs):
        
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, returns):
            policy_gradient.append(-log_prob * Gt)
        
        loss = torch.stack(policy_gradient).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def get_action(self,s):

        probs = self.predict(s)
        action = torch.multinomial(probs,1).item()
        log_prob = torch.log(probs[action])
        return action, log_prob
    
def reinforce(env, estimator, n_episode, gamma=1.0):

    for episode in range(n_episode):
        log_probs = []
        rewards = []
        total_reward_episode = [0] * n_episode
        state = env.reset()

        while True:
            action, log_prob = estimator.get_action(state)
            next_state, reward, is_done, _= env.step(action)
            total_reward_episode[episode] += reward
            log_probs.append(log_prob)
            rewards.append(reward)

            if is_done:
                returns = []
                Gt = 0
                pw = 0
                for reward in rewards[::-1]:
                    Gt += gamma ** pw * reward
                    pw += 1
                    returns.append(Gt)
                returns = returns[::-1]
                returns = torch.tensor(returns)
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
                estimator.update(returns, log_probs)
                print("Episode: {}, total reward: {}'.format(episode, return_episode[episode])")
                break
            
            state = next_state

    return total_reward_episode

n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_hidden = 128
lr = 0.003
policy_net = PolicyNetwork(n_state,n_action,n_hidden,lr)
gamma = 0.9

n_episode = 500
total_reward_episode = reinforce(env, policy_net, n_episode,gamma)
plt.plot(total_reward_episode)
plt.title('Episode reward over time')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()


