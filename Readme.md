# GridWorld DQN — Apprentissage par Renforcement Profond

Ce projet illustre la différence entre **deux stratégies d’apprentissage DQN (Deep Q-Network)** appliquées à un environnement simple **GridWorld** :  
- **Mise à jour immédiate** : le réseau est mis à jour à chaque étape.  
- **Mise à jour périodique** : le réseau est mis à jour après un certain nombre d’étapes, avec un **réseau cible** pour plus de stabilité.

---

## 🚀 Objectif du projet

Dans cette partie du projet, nous allons essayer d’implémenter **des approches différentes** en utilisant un agent dans un **GridWorld** avec un **but qui se déplace**.  
L’objectif est de **mettre en exergue le problème d’instabilité** rencontré dans le Deep Reinforcement Learning.  

En premier temps, nous implémentons le **DQN classique de DeepMind avec mise à jour immédiate**, puis nous comparons avec une **approche utilisant une mise à jour périodique**, qui représente la **solution proposée par DeepMind pour résoudre le problème de divergence**.

---

## ⚙️ Principe du Bootstrapping

Le **bootstrapping** consiste à **mettre à jour une estimation à partir d’une autre estimation** plutôt que de s’appuyer uniquement sur une récompense finale.  
Autrement dit, l’agent apprend non seulement à partir des récompenses immédiates, mais aussi à partir des **valeurs estimées des états futurs**.

La mise à jour Q-learning classique repose sur :

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \Big[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \Big]
$$

- $Q(s, a)$ : valeur actuelle de l’action $a$ dans l’état $s$  
- $r$ : récompense immédiate  
- $\gamma$ : facteur de discount  
- $s'$ : nouvel état après avoir pris l’action  
- $\max_{a'} Q(s', a')$ : estimation bootstrapée de la valeur future  

Le terme $r + \gamma \max_{a'} Q(s', a')$ est appelé **cible bootstrapée**.

---

## ⚠️ Problème d’instabilité détecté par DeepMind

DeepMind a constaté que l’utilisation du bootstrapping dans un réseau de neurones profond crée une **instabilité** pour plusieurs raisons :

1. **Corrélation forte entre les échantillons successifs** : les transitions d’expérience $(s, a, r, s')$ ne sont pas indépendantes.
2. **Cibles mouvantes** : comme $Q(s', a')$ est lui-même produit par le même réseau en cours d’apprentissage, la cible change constamment.
3. **Propagation d’erreurs** : une petite erreur dans la prédiction de $Q(s', a')$ peut se propager et s’amplifier lors des mises à jour successives.

---

## 💡 Solutions proposées par DeepMind

Pour stabiliser le bootstrapping dans DQN, DeepMind a introduit deux mécanismes majeurs :

### 1. **Experience Replay**
Les expériences sont stockées dans une mémoire $D = \{(s, a, r, s')\}$.  
Pendant l’entraînement, on échantillonne **aléatoirement** un mini-batch de transitions depuis cette mémoire pour briser les corrélations temporelles.

### 2. **Target Network**
On utilise un **second réseau $Q_{\text{target}}$**, copie périodique du réseau principal $Q_{\text{online}}$.  
Ce réseau fournit des cibles stables pour le bootstrapping :

$$
y = r + \gamma \max_{a'} Q_{\text{target}}(s', a'; \theta^-)
$$

Les poids $\theta^-$ du réseau cible ne sont mis à jour qu’occasionnellement, réduisant les oscillations et améliorant la stabilité.

---

## 🧠 En résumé

| Concept | Description |
|----------|--------------|
| **Bootstrapping** | Utiliser les valeurs estimées d’états futurs pour mettre à jour les valeurs actuelles. |
| **Problème** | Cibles mouvantes et corrélations entre les échantillons → instabilité. |
| **Solution DeepMind** | Experience Replay + Target Network. |

Grâce à ces techniques, **DQN** a pu combiner **le bootstrapping**, **l’apprentissage profond** et **l’exploration** pour battre les humains dans plusieurs jeux Atari.

---

## 🧩 Architecture du projet

# Architecture du projet GridWorld DQN

📦 GridWorld-DQN  
├── gridworld_env.py        # Environnement GridWorld (agent, but, actions)  
├── models.py               # Réseau DQN et mémoire d’expérience (ReplayBuffer)  
├── agent_immediate.py      # Agent DQN avec mise à jour immédiate  
├── agent_periodic.py       # Agent DQN avec mise à jour périodique  
├── train_immediate.py      # Script d’entraînement (agent immédiat)  
├── train_periodic.py       # Script d’entraînement (agent périodique)  
├── test_agents.py          # Tests et comparaison des agents entraînés  
└── requirements.txt        # Dépendances Python
