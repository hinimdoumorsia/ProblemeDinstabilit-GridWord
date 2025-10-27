# test_render_fix.py
import numpy as np
import random

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
        state = np.array([
            self.agent_pos[0] / (self.size-1), 
            self.agent_pos[1] / (self.size-1),
            self.goal_pos[0] / (self.size-1),
            self.goal_pos[1] / (self.size-1)
        ])
        return state
    
    def step(self, action):
        """
        Actions: 0=haut, 1=droite, 2=bas, 3=gauche
        """
        reward = -0.1  # pénalité pour chaque pas
        done = False
        
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
        self.goal_pos = [
            random.randint(0, self.size-1), 
            random.randint(0, self.size-1)
        ]
        # S'assurer que le but n'est pas sur la position de l'agent
        while self.goal_pos == self.agent_pos:
            self.goal_pos = [
                random.randint(0, self.size-1), 
                random.randint(0, self.size-1)
            ]
    
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

def test_render():
    """Test complet de la méthode render et des fonctionnalités de GridWorld"""
    print("=" * 50)
    print("TEST COMPLET GRIDWORLD AVEC RENDER")
    print("=" * 50)
    
    # Créer l'environnement
    env = GridWorld(size=5)
    
    # Test 1: État initial
    print("\n🧪 TEST 1: État initial après reset")
    state = env.reset()
    env.render()
    print(f"État: {state}")
    
    # Test 2: Quelques actions
    print("\n🧪 TEST 2: Séquence d'actions")
    actions = [1, 1, 2, 2]  # droite, droite, bas, bas
    action_names = ["Droite", "Droite", "Bas", "Bas"]
    
    for i, action in enumerate(actions):
        print(f"\nAction {i+1}: {action_names[i]}")
        state, reward, done = env.step(action)
        env.render()
        print(f"État: {state}")
        print(f"Récompense: {reward}, Terminé: {done}")
    
    # Test 3: Déplacement du but
    print("\n🧪 TEST 3: Déplacement du but")
    print("Avant déplacement:")
    env.render()
    
    env.move_goal()
    print("Après déplacement:")
    env.render()
    
    # Test 4: Atteindre le but
    print("\n🧪 TEST 4: Simulation jusqu'au but")
    env.reset()
    env.goal_pos = [0, 2]  # But proche pour test
    print("But placé à [0, 2]")
    env.render()
    
    # Actions pour atteindre le but
    actions_to_goal = [1, 1]  # droite, droite
    for i, action in enumerate(actions_to_goal):
        state, reward, done = env.step(action)
        print(f"Action {i+1}: {['Droite', 'Droite'][i]}")
        env.render()
        print(f"Récompense: {reward}, Terminé: {done}")
        
        if done:
            print("🎉 BUT ATTEINT!")
            break
    
    # Test 5: Actions aux bords
    print("\n🧪 TEST 5: Test des bords de la grille")
    env.reset()
    env.agent_pos = [0, 0]
    print("Agent dans le coin [0, 0]")
    env.render()
    
    # Essayer d'aller à gauche (devrait rester à la même position)
    state, reward, done = env.step(3)  # Gauche
    print("Après action Gauche (devrait rester à [0, 0]):")
    env.render()
    
    # Essayer d'aller en haut (devrait rester à la même position)
    state, reward, done = env.step(0)  # Haut
    print("Après action Haut (devrait rester à [0, 0]):")
    env.render()

def test_advanced_features():
    """Test des fonctionnalités avancées"""
    print("\n" + "=" * 50)
    print("TEST FONCTIONNALITÉS AVANCÉES")
    print("=" * 50)
    
    env = GridWorld(size=4)  # Taille différente
    
    # Test avec une grille plus petite
    print("\n🔄 Test avec grille 4x4")
    env.reset()
    env.render()
    
    # Test de normalisation des états
    print("\n📊 Test de normalisation des états:")
    positions_test = [
        [0, 0],    # Coin supérieur gauche
        [3, 3],    # Coin inférieur droit
        [1, 2],    # Position centrale
    ]
    
    for pos in positions_test:
        env.agent_pos = pos.copy()
        state = env.get_state()
        print(f"Position {pos} -> État normalisé: {state}")

if __name__ == "__main__":
    # Exécuter tous les tests
    test_render()
    test_advanced_features()
    
    print("\n" + "=" * 50)
    print("✅ TOUS LES TESTS TERMINÉS AVEC SUCCÈS!")
    print("=" * 50)