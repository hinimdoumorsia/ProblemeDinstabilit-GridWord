# test_agents.py
import numpy as np
from gridworld_env import GridWorld
from agent_immediate import DQNAgentImmediate
from agent_periodic import DQNAgentPeriodic

def test_single_agent(agent, env, agent_name, episodes=3):
    """Teste un agent spécifique"""
    print(f"\n{'='*60}")
    print(f"TEST DE L'AGENT: {agent_name}")
    print(f"{'='*60}")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        trajectory = [env.agent_pos.copy()]
        
        print(f"\n🎯 Épisode {episode + 1}")
        print(f"📍 Départ: Agent {env.agent_pos}, But {env.goal_pos}")
        env.render()
        
        while not done and steps < 20:
            # Sauvegarder l'epsilon original et le mettre à 0 pour le test
            original_epsilon = agent.epsilon
            agent.epsilon = 0.0
            
            action = agent.act(state)
            
            # Restaurer l'epsilon
            agent.epsilon = original_epsilon
            
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            trajectory.append(env.agent_pos.copy())
            
            action_names = ["↑ Haut", "→ Droite", "↓ Bas", "← Gauche"]
            print(f"  Step {steps}: {action_names[action]} → Agent {env.agent_pos} (Reward: {reward:.1f})")
            
            if done:
                print(f"  🎉 BUT ATTEINT!")
                env.render()
        
        print(f"📊 Score final: {total_reward:.2f}, Steps: {steps}")
        print(f"🛣️  Trajectoire: {trajectory}")
        
        # Déplacer le but pour le prochain test
        if episode < episodes - 1:  # Ne pas déplacer après le dernier épisode
            env.move_goal()

def test_performance(agent, agent_name, num_episodes=50):
    """Test quantitatif de la performance"""
    env = GridWorld(size=5)
    scores = []
    steps_to_goal = []
    success_count = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        # Mettre epsilon à 0 pour l'évaluation
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0
        
        while not done and steps < 50:
            action = agent.act(state)
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
        
        # Restaurer l'epsilon
        agent.epsilon = original_epsilon
        
        scores.append(total_reward)
        if done:
            steps_to_goal.append(steps)
            success_count += 1
        
        env.move_goal()
    
    success_rate = (success_count / num_episodes) * 100
    avg_steps = np.mean(steps_to_goal) if steps_to_goal else float('inf')
    avg_score = np.mean(scores)
    
    print(f"\n📈 Performance {agent_name}:")
    print(f"   Score moyen: {avg_score:.2f} ± {np.std(scores):.2f}")
    print(f"   Taux de succès: {success_rate:.1f}%")
    print(f"   Steps moyens pour atteindre le but: {avg_steps:.1f}" if steps_to_goal else "   Aucun but atteint")
    print(f"   Meilleur score: {np.max(scores):.2f}")
    print(f"   Pire score: {np.min(scores):.2f}")
    
    return avg_score, success_rate

def compare_agents():
    """Compare les deux agents entraînés"""
    
    # Configuration
    env = GridWorld(size=5)
    state_size = 4
    action_size = 4
    
    # Chargement des agents
    print("Chargement des agents...")
    
    # Agent immédiat
    immediate_agent = DQNAgentImmediate(state_size, action_size)
    try:
        immediate_agent.load("immediate_agent.pth")
        print("✅ Agent immédiat chargé avec succès")
    except FileNotFoundError:
        print("❌ Fichier immediate_agent.pth non trouvé")
        return
    except Exception as e:
        print(f"❌ Erreur lors du chargement de l'agent immédiat: {e}")
        return
    
    # Agent périodique
    periodic_agent = DQNAgentPeriodic(state_size, action_size)
    try:
        periodic_agent.load("periodic_agent.pth")
        print("✅ Agent périodique chargé avec succès")
    except FileNotFoundError:
        print("❌ Fichier periodic_agent.pth non trouvé")
        return
    except Exception as e:
        print(f"❌ Erreur lors du chargement de l'agent périodique: {e}")
        return
    
    # Test des agents
    test_single_agent(immediate_agent, env, "MISE À JOUR IMMÉDIATE")
    
    # Créer un nouvel environnement pour le deuxième agent
    env2 = GridWorld(size=5)
    test_single_agent(periodic_agent, env2, "MISE À JOUR PÉRIODIQUE")
    
    # Test de performance quantitative
    print(f"\n{'='*60}")
    print("TEST DE PERFORMANCE QUANTITATIVE")
    print(f"{'='*60}")
    
    imm_score, imm_success = test_performance(immediate_agent, "Immédiat")
    peri_score, peri_success = test_performance(periodic_agent, "Périodique")
    
    # Comparaison finale
    print(f"\n{'='*60}")
    print("COMPARAISON FINALE")
    print(f"{'='*60}")
    print(f"🔵 Immédiat  - Score: {imm_score:.2f}, Succès: {imm_success:.1f}%")
    print(f"🔴 Périodique - Score: {peri_score:.2f}, Succès: {peri_success:.1f}%")
    
    if imm_score > peri_score:
        print("🎖️  L'agent IMMÉDIAT performe mieux!")
    elif peri_score > imm_score:
        print("🎖️  L'agent PÉRIODIQUE performe mieux!")
    else:
        print("⚖️  Les deux agents ont des performances similaires!")

if __name__ == "__main__":
    compare_agents()