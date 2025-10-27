# train_immediate.py
import numpy as np
import matplotlib.pyplot as plt
from gridworld_env import GridWorld
from agent_immediate import DQNAgentImmediate

def train_immediate_agent():
    """Entraînement avec mise à jour IMMÉDIATE"""
    print("=== ENTRAÎNEMENT AVEC MISE À JOUR IMMÉDIATE ===")
    
    # Configuration
    env = GridWorld(size=5)
    state_size = 4
    action_size = 4
    
    # Création de l'agent
    agent = DQNAgentImmediate(state_size, action_size)
    episodes = 800
    scores = []
    moving_avg = []
    
    # Entraînement
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
    
    # Sauvegarde du modèle
    agent.save("immediate_agent.pth")
    
    # Visualisation des résultats
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores, alpha=0.3, label='Score par épisode')
    plt.plot(moving_avg, label='Moyenne mobile (100 épisodes)', linewidth=2)
    plt.xlabel('Épisodes')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Mise à Jour Immédiate - Performance')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(scores, bins=50, alpha=0.7, color='blue')
    plt.xlabel('Score')
    plt.ylabel('Fréquence')
    plt.title('Distribution des Scores')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('immediate_training.png')
    plt.show()
    
    return agent, scores, moving_avg

if __name__ == "__main__":
    agent, scores, moving_avg = train_immediate_agent()
    print(f"\nEntraînement terminé! Score moyen final: {np.mean(scores[-100:]):.2f}")