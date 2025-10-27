# agent_immediate.py
import torch
import torch.nn as nn
import numpy as np
import random
from models import DQN, ReplayMemory


class DQNAgentImmediate:
    """Agent avec mise à jour IMMÉDIATE (à chaque step)"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Un seul réseau pour l'approche immédiate
        self.policy_net = DQN(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), 
                                        lr=self.learning_rate)
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.policy_net(state)
        return np.argmax(q_values.detach().numpy())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Mise à jour IMMÉDIATE à chaque appel"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Q-values actuelles
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Q-values cibles (utilise le même réseau pour l'approche immédiate)
        next_q_values = self.policy_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Calcul de la loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Backpropagation IMMÉDIATE
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Mise à jour de l'epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)
    
    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))