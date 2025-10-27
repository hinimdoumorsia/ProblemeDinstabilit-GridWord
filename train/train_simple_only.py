# train_simple_only.py
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from gridworld_env import GridWorld

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

def train_simple_agent():
    """Entraînement de l'agent SANS replay memory"""
    print("=" * 70)
    print("🎯 ENTRAÎNEMENT AGENT SIMPLE - SANS REPLAY MEMORY")
    print("=" * 70)
    
    # Configuration
    env = GridWorld(size=5)
    state_size = 4
    action_size = 4
    
    # Création de l'agent
    agent = DQNSimple(state_size, action_size)
    episodes = 1000
    scores = []
    losses = []
    moving_avg = []
    epsilon_history = []
    
    print("Début de l'entraînement...")
    print("Caractéristiques de l'agent:")
    print("   ✅ Pas de replay memory")
    print("   ✅ Mise à jour immédiate à chaque step")
    print("   ✅ Un seul réseau neuronal")
    print("   ✅ Pas de réseau cible")
    print()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        episode_loss = 0
        step_count = 0
        
        while not done and steps < 100:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            # Mise à jour IMMÉDIATE sans mémoire
            loss = agent.train_step(state, action, reward, next_state, done)
            episode_loss += loss
            step_count += 1
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # Déplacer le but pour le prochain épisode
        env.move_goal()
        
        # Enregistrement des métriques
        scores.append(total_reward)
        losses.append(episode_loss / max(step_count, 1))
        epsilon_history.append(agent.epsilon)
        
        # Calcul de la moyenne mobile
        if episode >= 100:
            avg_score = np.mean(scores[-100:])
            moving_avg.append(avg_score)
        else:
            moving_avg.append(np.mean(scores))
        
        # Affichage des progrès
        if episode % 100 == 0:
            avg = np.mean(scores[-100:]) if episode >= 100 else np.mean(scores)
            current_loss = losses[-1] if losses else 0
            print(f"Épisode {episode:4d} | Score: {total_reward:6.2f} | Moyenne: {avg:6.2f} | Loss: {current_loss:6.3f} | Epsilon: {agent.epsilon:.3f}")
        
        # Arrêt précoce si performance excellente
        if episode >= 300 and np.mean(scores[-100:]) > 9.0:
            print(f"🎉 Performance excellente atteinte à l'épisode {episode}!")
            break
    
    # Sauvegarde du modèle
    agent.save("simple_agent.pth")
    print(f"\n💾 Modèle sauvegardé: simple_agent.pth")
    
    # Analyse finale
    final_avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
    success_rate = (sum(1 for s in scores[-100:] if s > 9) / min(100, len(scores))) * 100
    
    print(f"\n📊 PERFORMANCE FINALE:")
    print(f"   Score moyen (100 derniers): {final_avg:.2f}")
    print(f"   Taux de succès: {success_rate:.1f}%")
    print(f"   Epsilon final: {agent.epsilon:.3f}")
    
    # Visualisation des résultats
    plot_training_results(scores, losses, moving_avg, epsilon_history)
    
    return agent, scores, losses, moving_avg

def plot_training_results(scores, losses, moving_avg, epsilon_history):
    """Visualisation des résultats d'entraînement"""
    plt.figure(figsize=(16, 12))
    
    # Graphique 1: Scores
    plt.subplot(2, 2, 1)
    plt.plot(scores, alpha=0.3, color='blue', label='Scores par épisode')
    plt.plot(moving_avg, color='red', linewidth=2, label='Moyenne mobile (100)')
    plt.xlabel('Épisodes')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Agent Simple - Évolution des Scores')
    plt.grid(True, alpha=0.3)
    
    # Graphique 2: Loss
    plt.subplot(2, 2, 2)
    plt.plot(losses, color='orange', alpha=0.7)
    plt.xlabel('Épisodes')
    plt.ylabel('Loss Moyenne')
    plt.title('Évolution de la Loss')
    plt.grid(True, alpha=0.3)
    
    # Graphique 3: Distribution des scores
    plt.subplot(2, 2, 3)
    plt.hist(scores, bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Score')
    plt.ylabel('Fréquence')
    plt.title('Distribution des Scores')
    plt.grid(True, alpha=0.3)
    
    # Graphique 4: Epsilon et performance
    plt.subplot(2, 2, 4)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Epsilon
    ax1.plot(epsilon_history, color='red', label='Epsilon')
    ax1.set_ylabel('Epsilon', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.set_ylim(0, 1.1)
    
    # Scores (moyenne mobile)
    ax2.plot(moving_avg, color='blue', label='Score moyen')
    ax2.set_ylabel('Score Moyen', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    plt.xlabel('Épisodes')
    plt.title('Exploration vs Performance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_agent_training.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_simple_agent(agent, test_episodes=5):
    """Test de l'agent entraîné"""
    print("\n" + "=" * 70)
    print("🧪 TEST DE L'AGENT SIMPLE ENTRAÎNÉ")
    print("=" * 70)
    
    env = GridWorld(size=5)
    
    for episode in range(test_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        trajectory = [env.agent_pos.copy()]
        
        print(f"\n--- Épisode de Test {episode + 1} ---")
        print(f"Départ: Agent {env.agent_pos}, But {env.goal_pos}")
        env.render()
        
        # Mode exploitation pure
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0
        
        while not done and steps < 20:
            action = agent.act(state)
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            trajectory.append(env.agent_pos.copy())
            
            action_names = ["↑ Haut", "→ Droite", "↓ Bas", "← Gauche"]
            print(f"Step {steps:2d}: {action_names[action]} → Agent {env.agent_pos} (Reward: {reward:5.1f})")
            
            if done:
                print("🎉 BUT ATTEINT!")
                env.render()
                break
        
        # Restaurer epsilon
        agent.epsilon = original_epsilon
        
        print(f"📊 Résultat: Score {total_reward:.2f}, Steps {steps}")
        print(f"🛣️  Trajectoire: {trajectory}")
        
        if episode < test_episodes - 1:
            env.move_goal()

def performance_evaluation(agent, num_episodes=100):
    """Évaluation quantitative de la performance"""
    print("\n" + "=" * 70)
    print("📈 ÉVALUATION QUANTITATIVE SUR 100 ÉPISODES")
    print("=" * 70)
    
    env = GridWorld(size=5)
    scores = []
    steps_to_goal = []
    success_count = 0
    
    # Mode exploitation pure
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 50:
            action = agent.act(state)
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
        
        scores.append(total_reward)
        if done:
            steps_to_goal.append(steps)
            success_count += 1
        
        env.move_goal()
    
    # Restaurer epsilon
    agent.epsilon = original_epsilon
    
    success_rate = (success_count / num_episodes) * 100
    avg_score = np.mean(scores)
    avg_steps = np.mean(steps_to_goal) if steps_to_goal else float('inf')
    std_score = np.std(scores)
    
    print(f"📊 Performance sur {num_episodes} épisodes:")
    print(f"   ✅ Taux de succès: {success_rate:.1f}%")
    print(f"   📈 Score moyen: {avg_score:.2f} ± {std_score:.2f}")
    print(f"   ⏱️  Steps moyens: {avg_steps:.1f}" if steps_to_goal else "   ⏱️  Aucun but atteint")
    print(f"   🏆 Meilleur score: {np.max(scores):.2f}")
    print(f"   📉 Pire score: {np.min(scores):.2f}")
    print(f"   🔄 Épisodes réussis: {success_count}/{num_episodes}")
    
    return success_rate, avg_score

if __name__ == "__main__":
    # Entraînement complet
    agent, scores, losses, moving_avg = train_simple_agent()
    
    # Test de l'agent
    test_simple_agent(agent)
    
    # Évaluation quantitative
    success_rate, avg_score = performance_evaluation(agent)
    
    print("\n" + "=" * 70)
    print("✅ ENTRAÎNEMENT AGENT SIMPLE TERMINÉ AVEC SUCCÈS!")
    print("=" * 70)
    print(f"Résumé final:")
    print(f"   🎯 Score moyen: {avg_score:.2f}")
    print(f"   ✅ Taux de succès: {success_rate:.1f}%")
    print(f"   💾 Modèle sauvegardé: simple_agent.pth")
    print("=" * 70)