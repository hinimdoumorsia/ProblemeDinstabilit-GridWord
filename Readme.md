# GridWorld DQN — Apprentissage par Renforcement Profond

Ce projet illustre la différence entre **trois stratégies d’apprentissage DQN (Deep Q-Network)** appliquées à un environnement simple **GridWorld** :  

- **Agent simple (DQN sans replay memory)** : mise à jour immédiate à chaque étape, pas de mémoire d’expérience.  
- **Mise à jour immédiate (DQN avec Experience Replay)** : le réseau est mis à jour à chaque étape mais à partir d’un mini-batch extrait de la mémoire.  
- **Mise à jour périodique (DQN avec Experience Replay et Target Network)** : le réseau est mis à jour après un certain nombre d’étapes, avec un réseau cible pour plus de stabilité.

---

## 🚀 Objectif du projet

Ce projet vise à **comprendre les problèmes liés à l’apprentissage par renforcement profond (RL)** : instabilité, divergence et oscillations lors de l’entraînement des agents DQN.  
Nous implémentons différentes stratégies pour **mettre en évidence ces problèmes et la solution proposée par DeepMind**.

---

## ⚙️ Principe du Bootstrapping

Le **bootstrapping** consiste à **mettre à jour une estimation à partir d’une autre estimation** plutôt que de s’appuyer uniquement sur la récompense finale.  
Formule générale du Q-learning :

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

## 🧠 Agents et étapes mathématiques

### 1️⃣ Agent simple (sans Replay Memory)

- **Principe** : mise à jour immédiate à chaque étape, pas de mini-batch.  
- **Formule mathématique** :

$$
y = 
\begin{cases}
r & \text{si état terminal} \\
r + \gamma \max_{a'} Q(s', a'; \theta) & \text{sinon}
\end{cases}
$$

- **Loss** :  

$$
L(\theta) = \big( Q(s, a; \theta) - y \big)^2
$$

- **Backpropagation** : mise à jour directe des poids du réseau après chaque transition.

- **Exploration** : politique $\epsilon$-greedy avec décroissance progressive de $\epsilon$.

> Cet agent illustre directement **le problème des cibles mouvantes**, car il met à jour le réseau avec sa propre prédiction sans stabilisation.

---

### 2️⃣ Agent avec mise à jour immédiate + Replay Memory

- **Principe** : mise à jour immédiate, mais sur un mini-batch aléatoire issu de la mémoire d’expérience.  
- **Formule mathématique** :

$$
y = r + \gamma \max_{a'} Q(s', a'; \theta)
$$

- **Loss** :  

$$
L(\theta) = \frac{1}{N} \sum_{i=1}^{N} \big( Q(s_i, a_i; \theta) - y_i \big)^2
$$

- **Avantage** : briser les corrélations temporelles entre transitions pour stabiliser l’apprentissage.

---

### 3️⃣ Agent avec mise à jour périodique + Target Network

- **Principe** : mise à jour tous les $N$ steps, avec un **réseau cible** $Q_{\text{target}}$ pour calculer les cibles.  
- **Formule mathématique** :

$$
y = r + \gamma \max_{a'} Q_{\text{target}}(s', a'; \theta^-)
$$

- **Loss** :  

$$
L(\theta) = \frac{1}{N} \sum_{i=1}^{N} \big( Q_{\text{policy}}(s_i, a_i; \theta) - y_i \big)^2
$$

- **Mise à jour du réseau cible** : tous les $M$ steps, les poids $\theta^-$ sont remplacés par ceux du réseau principal $\theta$.

> Cette approche réduit fortement l’instabilité et représente **la solution DeepMind pour stabiliser le DQN**.

---

## ⚠️ Problème d’instabilité détecté par DeepMind

- **Corrélation forte entre les transitions** : les états successifs sont dépendants, ce qui fausse l’apprentissage.  
- **Cibles mouvantes** : le réseau apprend sur des cibles qui évoluent en permanence.  
- **Propagation d’erreurs** : une petite erreur dans la prédiction d’un état futur peut se propager et amplifier l’erreur.

---

## 💡 Solutions proposées

| Solution | Description |
|----------|-------------|
| **Experience Replay** | Stocker les expériences et échantillonner un mini-batch aléatoire pour casser les corrélations. |
| **Target Network** | Réseau cible fixe temporairement pour fournir des cibles stables. |

---

## 🧩 Architecture du projet

📦 GridWorld-DQN  
├── `gridworld_env.py`        # Environnement GridWorld (agent, but, actions)  
├── `models.py`               # Réseau DQN et Replay Memory  
├── `agent_simple.py`         # Agent DQN simple (sans Replay Memory)  
├── `agent_immediate.py`      # Agent DQN avec mise à jour immédiate + Replay Memory  
├── `agent_periodic.py`       # Agent DQN avec mise à jour périodique + Target Network  
├── `train_immediate.py`      # Script d’entraînement (agent immédiat)  
├── `train_periodic.py`       # Script d’entraînement (agent périodique)  
├── `train_simple.py`         # Script d’entraînement (agent simple)  
├── `test_agents.py`          # Tests et comparaison des agents entraînés  
└── `requirements.txt`        # Dépendances Python

---

## 📌 Conclusion

Ce projet permet de **comparer les performances et la stabilité de différents agents DQN** :  

- L’**agent simple** montre l’instabilité maximale.  
- L’**agent immédiat avec Replay Memory** améliore légèrement la stabilité.  
- L’**agent périodique avec Target Network** montre la meilleure stabilité, illustrant **la solution de DeepMind** pour le Deep Reinforcement Learning.  

Ainsi, le projet est une **démonstration pédagogique complète** des **problèmes d’apprentissage profond avec bootstrapping** et des **solutions pour stabiliser les DQN**.
