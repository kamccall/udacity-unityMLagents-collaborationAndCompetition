import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = .0001    # L2 weight decay

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Agent():
    def __init__(self, n_agents, state_size, action_size, random_seed):      
        self.n_agents = n_agents
        self.state_size = state_size
        self.action_size = action_size
        
        # actor networks local and target
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = LR_ACTOR)
        
        ## critic networks local and target
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = LR_CRITIC)
        
        # create noise and replaybuffer objects (based on classes defined below)
        self.noise = OUNoise(action_size, random_seed)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def act(self,state, add_noise = True):
        state = torch.from_numpy(state).float().to(device) # convert state from numpy array to a tensor
        
        self.actor_local.eval() # put network in evaluation mode, no trainig
        
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
            
        self.actor_local.train()
        
        if add_noise:
            action += self.noise.sample()
            
        return np.clip(action, -1, 1)  # ensure action space is in valid range of -1.0 to 1.0
        
    def step(self, state, action, reward, next_state, done): 
        for i in range(self.n_agents):                             # used for 20 agent run
            self.memory.add(state[i,:], action[i,:], reward[i], next_state[i,:], done[i])
            
        # self.memory.add(state, action, reward, next_state, done) # used for single agent run
        
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
    
    def learn(self, experiences, gamma):  
        # obtain all the necessary contents from the sampled replay buffer
        # print('calling learn...')  # KM TEST
        states, actions, rewards, next_states, dones = experiences        
        
        # get Q_target for next_states to compute Q_target for current state
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1-dones))
        
        # compute the actual Q_target based on current critic_local network
        Q_expected = self.critic_local(states, actions)
        Q_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        Q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) # KM important for stable training
        self.critic_optimizer.step()
        
        # train actor_local to maximize Q
        actions_pred = self.actor_local(states)
        Q_value = - self.critic_local(states, actions_pred).mean()
    
        self.actor_optimizer.zero_grad()
        Q_value.backward()
        self.actor_optimizer.step()
        
        # soft update target params
        self.soft_update( self.actor_local, self.actor_target, TAU)
        self.soft_update( self.critic_local, self.critic_target, TAU)       
    
    def soft_update(self, local_model, target_model, tau):
        # periodically move values between local and target networks
        for target_params, local_params in zip(target_model.parameters(), local_model.parameters()):
            target_params.data.copy_( tau * local_params.data + (1-tau) * target_params.data)
    
    def reset(self):
        self.noise.reset()
    
class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.05):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)