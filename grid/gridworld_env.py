import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.agent_pos = [0, 0]
        self.goal_pos = [size-1, size-1]
    
    def reset(self):
        """Réinitialise l'environnement"""
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size-1, self.size-1]
        return self.get_state()
    
    def get_state(self):
        """Retourne l'état actuel sous forme de coordonnées normalisées"""
        state = np.array([self.agent_pos[0] / (self.size-1), 
                         self.agent_pos[1] / (self.size-1),
                         self.goal_pos[0] / (self.size-1),
                         self.goal_pos[1] / (self.size-1)])
        return state
    
    def step(self, action):
        """
        Actions: 0=haut, 1=droite, 2=bas, 3=gauche
        """
        reward = -0.1  # pénalité pour chaque pas
        done = False
        
        # Sauvegarder l'ancienne position
        old_pos = self.agent_pos.copy()
        
        # Appliquer l'action
        if action == 0:  # haut
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # droite
            self.agent_pos[1] = min(self.size-1, self.agent_pos[1] + 1)
        elif action == 2:  # bas
            self.agent_pos[0] = min(self.size-1, self.agent_pos[0] + 1)
        elif action == 3:  # gauche
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        
        # Vérifier si l'agent a atteint le but
        if self.agent_pos == self.goal_pos:
            reward = 10.0
            done = True
        
        return self.get_state(), reward, done
    
    def move_goal(self):
        """Déplace le but à une position aléatoire"""
        self.goal_pos = [random.randint(0, self.size-1), 
                        random.randint(0, self.size-1)]
        # S'assurer que le but n'est pas sur la position de l'agent
        while self.goal_pos == self.agent_pos:
            self.goal_pos = [random.randint(0, self.size-1), 
                            random.randint(0, self.size-1)]
    
    def render(self):
        """Affiche la grille dans la console"""
        print(f"\nGrille {self.size}x{self.size}:")
        print(f"Agent: {self.agent_pos}, But: {self.goal_pos}")
        
        for i in range(self.size):
            line = ""
            for j in range(self.size):
                if [i, j] == self.agent_pos:
                    line += " A "
                elif [i, j] == self.goal_pos:
                    line += " G "
                else:
                    line += " . "
            print(line)
        print()

class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

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
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
    
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
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
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

def train_immediate_agent():
    """Entraînement avec mise à jour IMMÉDIATE"""
    print("=== AGENT AVEC MISE À JOUR IMMÉDIATE ===")
    
    env = GridWorld(size=5)
    state_size = 4
    action_size = 4
    
    agent = DQNAgentImmediate(state_size, action_size)
    episodes = 800
    scores = []
    moving_avg = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 100:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            # Mise à jour IMMÉDIATE à chaque step
            agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # Déplacer le but pour le prochain épisode
        env.move_goal()
        
        scores.append(total_reward)
        
        # Calcul de la moyenne mobile
        if episode >= 100:
            avg_score = np.mean(scores[-100:])
            moving_avg.append(avg_score)
        else:
            moving_avg.append(np.mean(scores))
        
        if episode % 100 == 0:
            avg = np.mean(scores[-100:]) if episode >= 100 else np.mean(scores)
            print(f"Épisode {episode}, Score: {total_reward:.2f}, Moyenne: {avg:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent, scores, moving_avg

def train_periodic_agent():
    """Entraînement avec mise à jour PÉRIODIQUE"""
    print("\n=== AGENT AVEC MISE À JOUR PÉRIODIQUE ===")
    
    env = GridWorld(size=5)
    state_size = 4
    action_size = 4
    
    agent = DQNAgentPeriodic(state_size, action_size)
    episodes = 800
    scores = []
    moving_avg = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 100:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            # Mise à jour PÉRIODIQUE (seulement tous les N steps)
            agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # Déplacer le but pour le prochain épisode
        env.move_goal()
        
        scores.append(total_reward)
        
        # Calcul de la moyenne mobile
        if episode >= 100:
            avg_score = np.mean(scores[-100:])
            moving_avg.append(avg_score)
        else:
            moving_avg.append(np.mean(scores))
        
        if episode % 100 == 0:
            avg = np.mean(scores[-100:]) if episode >= 100 else np.mean(scores)
            print(f"Épisode {episode}, Score: {total_reward:.2f}, Moyenne: {avg:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent, scores, moving_avg

def test_agent(agent, env, agent_name, episodes=3):
    """Teste un agent spécifique"""
    print(f"\n--- Test Agent {agent_name} ---")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        trajectory = [env.agent_pos.copy()]
        
        print(f"\nÉpisode {episode + 1}:")
        print(f"Départ: Agent {env.agent_pos}, But {env.goal_pos}")
        env.render()  # ✅ MAINTENANT ÇA FONCTIONNE !
        
        while not done and steps < 20:
            # Utilise la politique greedy (epsilon=0)
            action = agent.act(state)
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            trajectory.append(env.agent_pos.copy())
            
            action_names = ["Haut", "Droite", "Bas", "Gauche"]
            print(f"  Step {steps}: {action_names[action]} -> Agent {env.agent_pos}")
            
            if done:
                print(f"  🎉 BUT ATTEINT!")
                env.render()
        
        print(f"Score final: {total_reward:.2f}, Steps: {steps}")
        print(f"Trajectoire complète: {trajectory}")
        
        # Déplacer le but pour le prochain test
        if episode < episodes - 1:
            env.move_goal()

def compare_agents():
    """Compare les deux agents et affiche les résultats"""
    
    # Entraînement des deux agents
    immediate_agent, immediate_scores, immediate_avg = train_immediate_agent()
    periodic_agent, periodic_scores, periodic_avg = train_periodic_agent()
    
    # Visualisation comparative
    plt.figure(figsize=(15, 10))
    
    # Graphique 1: Scores comparés
    plt.subplot(2, 2, 1)
    plt.plot(immediate_scores, alpha=0.3, color='blue', label='Immédiat (raw)')
    plt.plot(immediate_avg, color='blue', linewidth=2, label='Immédiat (moyenne)')
    plt.plot(periodic_scores, alpha=0.3, color='red', label='Périodique (raw)')
    plt.plot(periodic_avg, color='red', linewidth=2, label='Périodique (moyenne)')
    plt.xlabel('Épisodes')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Comparaison des Scores')
    plt.grid(True, alpha=0.3)
    
    # Graphique 2: Moyennes mobiles seulement
    plt.subplot(2, 2, 2)
    plt.plot(immediate_avg, color='blue', linewidth=2, label='Mise à jour Immédiate')
    plt.plot(periodic_avg, color='red', linewidth=2, label='Mise à jour Périodique')
    plt.xlabel('Épisodes')
    plt.ylabel('Score Moyen')
    plt.legend()
    plt.title('Moyennes Mobiles (100 épisodes)')
    plt.grid(True, alpha=0.3)
    
    # Graphique 3: Distribution des scores
    plt.subplot(2, 2, 3)
    plt.hist(immediate_scores, bins=50, alpha=0.7, color='blue', label='Immédiat')
    plt.hist(periodic_scores, bins=50, alpha=0.7, color='red', label='Périodique')
    plt.xlabel('Score')
    plt.ylabel('Fréquence')
    plt.legend()
    plt.title('Distribution des Scores')
    plt.grid(True, alpha=0.3)
    
    # Graphique 4: Scores des 100 derniers épisodes
    plt.subplot(2, 2, 4)
    last_100_imm = immediate_scores[-100:] if len(immediate_scores) >= 100 else immediate_scores
    last_100_per = periodic_scores[-100:] if len(periodic_scores) >= 100 else periodic_scores
    
    plt.boxplot([last_100_imm, last_100_per], labels=['Immédiat', 'Périodique'])
    plt.ylabel('Score')
    plt.title('Performance Finale (100 derniers épisodes)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistiques comparatives
    print("\n" + "="*50)
    print("STATISTIQUES COMPARATIVES")
    print("="*50)
    
    print(f"Mise à jour Immédiate - Score moyen final: {np.mean(last_100_imm):.2f} ± {np.std(last_100_imm):.2f}")
    print(f"Mise à jour Périodique - Score moyen final: {np.mean(last_100_per):.2f} ± {np.std(last_100_per):.2f}")
    
    # Test des agents entraînés
    print("\n" + "="*50)
    print("TEST DES AGENTS ENTRAÎNÉS")
    print("="*50)
    
    test_env = GridWorld(size=5)
    test_agent(immediate_agent, test_env, "Immédiat")
    test_agent(periodic_agent, test_env, "Périodique")

if __name__ == "__main__":
    # Comparaison complète des deux approches
    compare_agents()