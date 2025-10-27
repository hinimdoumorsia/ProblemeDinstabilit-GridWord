# agent_simple.py
import torch
import torch.nn as nn
import numpy as np
import random

class DQNSimple:
    """Agent DQN SANS replay memory - Mise à jour immédiate uniquement"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Pas de replay memory
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Un seul réseau
        self.policy_net = self._build_model(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
    
    def _build_model(self, state_size, action_size):
        """Construction du réseau de neurones"""
        return nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
    
    def act(self, state):
        """Sélection d'action avec politique ε-greedy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.policy_net(state_tensor)
        return np.argmax(q_values.detach().numpy())
    
    def train_step(self, state, action, reward, next_state, done):
        """Mise à jour IMMÉDIATE sans mémoire de replay"""
        
        # Conversion en tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        # Q-values courantes
        current_q_values = self.policy_net(state_tensor)
        current_q = current_q_values[0, action]
        
        # Q-values cibles
        with torch.no_grad():
            next_q_values = self.policy_net(next_state_tensor)
            max_next_q = torch.max(next_q_values)
            
        target_q = reward
        if not done:
            target_q += self.gamma * max_next_q
        
        # Calcul de la loss
        loss = self.loss_fn(current_q, torch.tensor([target_q]))
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Décay de l'epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, filename):
        """Sauvegarde du modèle"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
    
    def load(self, filename):
        """Chargement du modèle"""
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']