# Résultats des Tests des Agents RL dans GridWorld

## Chargement des Agents
- ✅ Agent immédiat chargé avec succès
- ✅ Agent périodique chargé avec succès
- ✅ Agent simple (sans replay memory) chargé avec succès

---

## 1️⃣ Test de l'Agent : Mise à Jour Immédiate

### Épisodes 1 à 3
- **Départ** : Agent `[0, 0]`, But `[4, 4]`
- **Trajectoire** : `[0,0] → [0,1] → [0,2] → [0,3] → [1,3] → [2,3] → [3,3] → [3,4] → [4,4]`
- **Récompenses** : -0.1 par étape sauf dernière étape (+10)
- **Steps** : 8
- **Score final** : 9.30
- 🎉 **But atteint !**

> Observation : L’agent immédiat atteint le but de manière optimale en 8 étapes à chaque épisode.

---

## 2️⃣ Test de l'Agent : Mise à Jour Périodique (Replay Memory)

### Épisodes 1 à 3
- **Départ** : Agent `[0, 0]`, But `[4, 4]`
- **Trajectoire** : `[0,0] → [1,0] → [2,0] → [3,0] → [4,0] → [4,1] → [4,2] → [4,3] → [4,4]`
- **Récompenses** : -0.1 par étape sauf dernière étape (+10)
- **Steps** : 8
- **Score final** : 9.30
- 🎉 **But atteint !**

> Observation : L’agent périodique utilise une **replay memory** pour consolider ses expériences et stabiliser l’apprentissage.

---

## 3️⃣ Test de l'Agent : Simple (Sans Replay Memory)

### Épisodes 1 à 5
- **Départ** : Agent `[0, 0]`, But `[4, 4]`
- **Trajectoire** : `[0,0] → [1,0] → [2,0] → [3,0] → [4,0] → [4,1] → [4,2] → [4,3] → [4,4]`
- **Récompenses** : -0.1 par étape sauf dernière étape (+10)
- **Steps** : 8
- **Score final** : 9.30
- 🎉 **But atteint !**

> Observation : L’agent simple **sans replay memory** met à jour ses Q-values à chaque étape. Malgré l’absence de mémoire, il atteint le but efficacement dans ce GridWorld simple.  

---

## 4️⃣ Performance Quantitative

| Agent                   | Score moyen | Succès | Steps moyens | Meilleur score | Pire score |
|-------------------------|------------|--------|--------------|----------------|------------|
| Immédiat                | 9.30 ± 0.00 | 100%   | 8.0          | 9.30           | 9.30       |
| Périodique (Replay)     | 9.30 ± 0.00 | 100%   | 8.0          | 9.30           | 9.30       |
| Simple (Sans Replay)    | 9.30 ± 0.00 | 100%   | 8.0          | 9.30           | 9.30       |

> Tous les agents atteignent le but de manière optimale dans ce GridWorld 5x5.

---

## 5️⃣ Comparaison Finale
- 🔵 **Immédiat** : Score 9.30, Succès 100%
- 🔴 **Périodique avec replay memory** : Score 9.30, Succès 100%
- 🟢 **Simple sans replay memory** : Score 9.30, Succès 100%
- ⚖️ **Conclusion** : Les performances sont similaires ici, mais la **replay memory** reste un avantage pour des environnements plus complexes.

---

## 6️⃣ Visualisations

### 1️⃣ Courbe des scores par épisode - Agent immédiat
![Courbe Immédiat](images/immediate_training.png)

### 2️⃣ Courbe des scores par épisode - Agent périodique
![Courbe Périodique](images/periodic_training.png)

### 3️⃣ Courbe des scores par épisode - Agent simple sans replay memory
![Courbe Simple](images/simple_agent_training.png)

> Les visualisations permettent de comparer la stabilité et la répétabilité des trois approches.

---

### 📝 Résumé
- L’**agent immédiat** privilégie un chemin horizontal puis vertical.  
- L’**agent périodique avec replay memory** consolide ses expériences pour stabiliser l’apprentissage.  
- L’**agent simple** met à jour ses Q-values directement, sans mémoire, et réussit efficacement dans ce GridWorld simple.  
- Les visualisations confirment la constance des scores et le succès de chaque épisode.
---
