#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.autograd as autograd
from collections import namedtuple
import torch.nn.functional as F
import random
import torch.optim as optim
import gym
from gym import wrappers
import csv
import math
import argparse

import matplotlib.pyplot as plt
import pickle
import os 
from tqdm import tqdm_notebook as tqdm
import numpy as np


# In[2]:


WEIGHT_DIR = 'weights'
os.makedirs(WEIGHT_DIR, exist_ok=True)


# In[3]:


# History
HISTORY_DIR = 'history'
os.makedirs(HISTORY_DIR, exist_ok=True)

def save_history(data, file_name='history'):
    if '.pickle' not in file_name:
        file_name += '.pickle'
    # Store data (serialize)
    with open(os.path.join(HISTORY_DIR, file_name), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_history(file_name='history'):
    if '.pickle' not in file_name:
        file_name += '.pickle'
    # Load data (deserialize)
    with open(os.path.join(HISTORY_DIR, file_name), 'rb') as handle:
        unserialized_data = pickle.load(handle)
    return unserialized_data

def plot_hisotry(history, block=10, save=False):
    scores = history['Rewards']
    if block > 0:
        scores = [np.mean(scores[low:low+block]) for low in range(0, len(scores), block)]
    plt.plot(scores)
    plt.xlabel("# %d game" %(block))
    if save:
        file_name = "%d.png"%( len(scores) * block) 
        plt.savefig(os.path.join(HISTORY_DIR, file_name))
        plt.clf() 
        plt.cla()
    else:
        plt.show()


# In[4]:


# GPU usage
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


# In[5]:


def Variable(data, *args, **kwargs):
    if use_cuda:
        data = data.cuda()
    return autograd.Variable(data,*args, **kwargs)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward','done'))


# In[6]:


# Experience replay
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# In[7]:


# DQN
class DQN(nn.Module):
    def __init__(self, env, replayMemory):
        super(DQN, self).__init__()
        self.action_dim = env.action_space.n
        self.state_dim =  env.observation_space.shape[0]
        self.model = nn.Sequential(nn.Linear(self.state_dim, 32),
                                nn.ReLU(),
                                nn.Linear(32, self.action_dim))
        
        
        self.memory = replayMemory
        self.epsilon = EPS_START
        self.action_dim = env.action_space.n
        self.state_dim =  env.observation_space.shape[0]
        self.model = nn.Sequential(nn.Linear(self.state_dim, 32),
                                nn.ReLU(),
                                nn.Linear(32, self.action_dim))
        self.targetModel = nn.Sequential(nn.Linear(self.state_dim,32),
                                nn.ReLU(),
                                nn.Linear(32,self.action_dim))
        self.updateTargetModel()
        self.targetModel.eval()
        print(self.model)

    def forward(self,x):
        return self.model(x)
    
    def target_forward(self,x):
        return self.targetModel(x)

    def updateTargetModel(self):
        #Assign weight to the target model
        self.targetModel.load_state_dict(self.model.state_dict())

    #epsilon greedy policy to select action
    def egreedy_action(self,state):
        global steps_done
        if self.epsilon >= EPS_END:
            self.epsilon *= EPS_DECAY
        steps_done += 1
        if random.random() > self.epsilon:
            return self.action(state)
        else:
            return LongTensor([[random.randrange(self.action_dim)]])
        
    def action(self,state):
        return self.forward(Variable(state)).detach().data.max(1)[1].view(1, 1)

    def loss(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        minibatch = Transition(*zip(*transitions))

        state_batch = Variable(torch.cat(minibatch.state))
        action_batch = Variable(torch.cat(minibatch.action))
        reward_batch = Variable(torch.cat(minibatch.reward))
        done_batch = Variable(torch.cat(minibatch.done))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.forward(state_batch).gather(1, action_batch)
        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor),volatile=True)
        non_final_next_states = Variable(torch.cat([s for t,s in enumerate(minibatch.next_state) if done_batch[t]==0]))
        next_state_values[done_batch == 0] = self.forward(non_final_next_states).max(1)[0].detach()
        
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        # Undo volatility (which was used to prevent unnecessary gradients)
        expected_state_action_values = Variable(expected_state_action_values.data,volatile=False)

        
        criterion = nn.MSELoss()
        loss = criterion(torch.squeeze(state_action_values), expected_state_action_values)
        return loss

    def push(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

    def saveModel(self, name):
        torch.save(self.model.state_dict(), name)
    
    def loadModel(self, name):
        self.model.load_state_dict(torch.load(name))
        self.updateTargetModel()


# In[8]:


def train(dqn, env, optimizer, STEP, num_episodes):
    best_testReward = 0
    pbar = tqdm(total=num_episodes)
    history = {'Step':[], 'Loss':[], "Rewards":[]}
    
    for episode in range(1, num_episodes+1):
        state = env.reset()
        state = torch.from_numpy(state.reshape((-1,4))).float()
        total_reward = 0
        total_loss = 0
        for t in range(1, STEP+1):
            action = dqn.egreedy_action(state)
            next_state,reward,done,_ = env.step(int(action[0,0].data.item()))
            next_state = torch.from_numpy(next_state.reshape((-1,4))).float()
            total_reward += reward
            reward = Tensor([reward])
            final = LongTensor([done])
            dqn.push(state,action,next_state,reward,final)
            state = next_state
            loss = dqn.loss()
            
            # Backward
            if loss is not None:
                optimizer.zero_grad()
                loss.backward()
                for param in dqn.model.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
                total_loss += loss.data.item()
            if steps_done % TARGETQ_UPDATE == 0:
                dqn.updateTargetModel()
            if done:
                break
                
        pbar.set_postfix({"Step":str(t), "Loss":'%.4f'%(total_loss/t), "Reward":str(total_reward)})
        history['Step'] += [t]
        history['Loss'] += [(total_loss/t)]
        #history['Rewards'] += [total_reward]
        pbar.update()

        # Evaluate Model
        if (episode) % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                state = torch.from_numpy(state.reshape((-1, 4))).float()
                for j in range(STEP):
                    action = dqn.action(state)
                    state,reward,done,_ = env.step(int(action[0,0].data.item()))
                    state = torch.from_numpy(state.reshape((-1, 4))).float()
                    total_reward += reward
                    if done:
                        break
            avg_reward = total_reward / TEST
            print('Episode: {} Evaluation Average Reward: {}'.format(episode, avg_reward))
            if avg_reward >= best_testReward:
                print('Save best model on episode {}'.format(episode))
                dqn.saveModel(os.path.join(WEIGHT_DIR, 'DQN_best.pth'))
                best_testReward = avg_reward
            print("\n")
            history['Rewards'] += [avg_reward]
    save_history(history)
    plot_hisotry(history, block=1, save=True)
    dqn.saveModel(os.path.join(WEIGHT_DIR, 'DQN_final.pth'))
    return history


# In[9]:


# Hyper-parameters
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
REPLAY_SIZE = 5000
BATCH_SIZE = 128
TARGETQ_UPDATE = 50
num_episodes = 1000
STEP = 250 # Max : 200
TEST = 10
steps_done = 0


# In[10]:


env = gym.make('CartPole-v0')
memory = ReplayMemory(REPLAY_SIZE)
dqn = DQN(env,memory)
optimizer = optim.Adam(dqn.model.parameters(), lr=5e-4)

if use_cuda:
    dqn.model.cuda()
    dqn.targetModel.cuda()


# In[11]:


history = train(dqn, env, optimizer, STEP, num_episodes)


# In[ ]:


def plot_hisotry(history, block=10, save=False):
    scores = history['Rewards']
    if block > 0:
        scores = [np.mean(scores[low:low+block]) for low in range(0, len(scores), block)]
    plt.plot(scores)
    plt.xlabel("# %d game" %(block))
    plt.title("Test Rewards (average over 10 games)")
    if save:
        file_name = "%d.png"%( len(scores) * block) 
        plt.savefig(os.path.join(HISTORY_DIR, file_name))
        plt.clf() 
        plt.cla()
    else:
        plt.show()


# In[12]:


history = load_history()
plot_hisotry(history, block=10, save=False)


# In[13]:


def eval_dqn(dqn, env, num_test):
    total_reward = 0
    for i in tqdm(range(num_test)):
        state = env.reset()
        state = torch.from_numpy(state.reshape((-1, 4))).float()
        for j in range(STEP):
            action = dqn.action(state)
            state,reward,done,_ = env.step(int(action[0,0].data.item()))
            state = torch.from_numpy(state.reshape((-1, 4))).float()
            total_reward += reward
            if done:
                break
    avg_reward = total_reward / num_test
    return avg_reward

num_test = 100
avg_reward = eval_dqn(dqn, env, num_test)
print(avg_reward)


# In[ ]:


# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
# columns of actions taken
state_action_values = self.forward(state_batch).gather(1, action_batch)
# Compute V(s_{t+1}) for all next states.
next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor),volatile=True)
non_final_next_states = Variable(torch.cat([s for t,s in enumerate(minibatch.next_state) if done_batch[t]==0]))
next_state_values[done_batch == 0] = self.forward(non_final_next_states).max(1)[0].detach()



expected_state_action_values = (next_state_values * discount) + reward_batch
criterion = nn.MSELoss()
loss = criterion(torch.squeeze(state_action_values), expected_state_action_values)

