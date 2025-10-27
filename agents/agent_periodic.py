# agent_periodic.py
import torch
import torch.nn as nn
import numpy as np
import random
from models import DQN, ReplayMemory

class DQNAgentPeriodic:
    """Agent avec mise à jour PÉRIODIQUE (pendant une période)"""
    
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
        
        # Paramètres de mise à jour périodique
        self.update_frequency = 4  # Mise à jour tous les 4 steps
        self.step_count = 0
        self.target_update_frequency = 100  # Mise à jour du réseau cible
        
        # Deux réseaux pour l'approche périodique
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), 
                                        lr=self.learning_rate)
        
        # Synchronisation initiale
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.policy_net(state)
        return np.argmax(q_values.detach().numpy())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Mise à jour PÉRIODIQUE - seulement tous les N steps"""
        self.step_count += 1
        
        # Vérifier si c'est le moment de mettre à jour
        if self.step_count % self.update_frequency != 0:
            return
        
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
        
        # Q-values cibles (utilise le réseau cible)
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Calcul de la loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Backpropagation PÉRIODIQUE
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Mise à jour de l'epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Mise à jour périodique du réseau cible
        if self.step_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filename):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'epsilon': self.epsilon
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_count = checkpoint['step_count']
        self.epsilon = checkpoint['epsilon']