# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium


## Aim
To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement
The objective is to train an autonomous agent to navigate a slippery 4x4 grid world (`FrozenLake-v1`) from the starting position (S) to the goal state (G) while avoiding hazardous holes (H). Because the surface is slippery, the agent's movement direction is uncertain and only partially dependent on the chosen action. Using the model-free **Q-Learning** algorithm, the agent must iteratively discover an optimal policy that maximizes cumulative discounted rewards despite the stochastic environment dynamics.

---

## Software Requirements
1. Python: 3.8+
2. Gymnasium: gymnasium>=0.29.0
3. NumPy: numpy>=1.22.0
4. Matplotlib: matplotlib>=3.5.0
5. Jupyter Notebook / Google Colab
---

## Environment Description
The `FrozenLake-v1` environment represents a 4x4 grid of frozen and broken ice tiles:

* **Grid Layout**:
```text
S F F F
F H F H
F F F H
H F F G
```

* **Legend**:
  * `S`: Starting point (safe)
  * `F`: Frozen surface (safe)
  * `H`: Hole (fall into hole -> episode terminates with 0 reward)
  * `G`: Goal (reach goal -> episode terminates with +1 reward)

* **State Space**: Discrete space of 16 states (0 to 15).

* **Action Space**: Discrete space with 4 possible actions:
  * `0`: Left
  * `1`: Down
  * `2`: Right
  * `3`: Up

* **Transition Dynamics**: With `is_slippery=True`, taking an action moves the agent in the intended direction with probability 1/3, and in either of the two perpendicular directions with probability 1/3 each.

* **Reward Mechanism**:
  * Reaching Goal (`G`): +1
  * Falling into Hole (`H`): 0
  * Moving to Frozen tile (`F`): 0


## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm



## Python Program

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

num_states = env.observation_space.n
num_actions = env.action_space.n

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 20000
max_steps = 100

alpha = 0.1          # learning rate
gamma = 0.99         # discount factor

epsilon = 1.0        # starting exploration rate
epsilon_min = 0.01
epsilon_decay = 0.0005   # exponential decay rate

# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((num_states, num_actions))


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

rng = np.random.default_rng(42)

def choose_action(state, epsilon):
    if rng.random() < epsilon:
        return env.action_space.sample()
    # random tie-breaking among the maximal action values
    best = np.flatnonzero(Q[state] == Q[state].max())
    return int(rng.choice(best))

# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------

episode_rewards = []

for episode in range(num_episodes):
    state, info = env.reset()
    total_reward = 0.0

    for step in range(max_steps):
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)

        # Q-learning update
        if terminated:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])

        Q[state, action] += alpha * (target - Q[state, action])

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    # decay exploration
    epsilon = epsilon_min + (1.0 - epsilon_min) * np.exp(-epsilon_decay * episode)

    episode_rewards.append(total_reward)

episode_rewards = np.array(episode_rewards)

# derive the value function and greedy policy from Q
state_values = np.max(Q, axis=1)
learned_policy = np.argmax(Q, axis=1)

# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)


# -------------------------------------------------
# Output
# -------------------------------------------------

print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)
print_policy(learned_policy)

average_reward = np.mean(episode_rewards[-1000:])
print("\nAverage reward over last 1000 episodes:", average_reward)

# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Q-Learning Curve - FrozenLake")
plt.grid(True)
plt.show()

env.close()
     
```
---

## Output

Final Q-table:

<img width="485" height="382" alt="image" src="https://github.com/user-attachments/assets/52dfe9e9-0b1a-4312-855a-7b3df707226f" />




Estimated State-Value Function:


<img width="352" height="127" alt="image" src="https://github.com/user-attachments/assets/0ec6f7d8-8609-4808-b18e-11336594ade7" />


Learned Policy:

<img width="281" height="126" alt="image" src="https://github.com/user-attachments/assets/fa07f95f-99c7-4e8b-aef1-70142594b5f3" />




Average reward over last 1000 episodes: 0.696


<img width="987" height="621" alt="image" src="https://github.com/user-attachments/assets/ec932acf-e203-4325-b356-70f84f2ddf6d" />

---

## Result
The Q-Learning algorithm was successfully implemented on the Gymnasium `FrozenLake-v1` environment. The agent learned the optimal action-value function ($Q$) and derived a policy that successfully navigates the slippery grid world from start to goal while avoiding holes, achieving an average reward of ~0.43 over the last 1000 training episodes.

---

## Inference
1. **Convergence on Stochastic Dynamics**: Despite the slippery environment making transitions non-deterministic (with only a 33% chance of moving in the intended direction), the Q-learning agent successfully converged toward a high-reward policy.
2. **Exploration vs. Exploitation Balance**: The exponential decay of epsilon ($\epsilon$) ensured sufficient state-space exploration early on and shifted the agent toward stable exploitation in later episodes.
3. **Safe Policy Formulation**: The learned policy directs the agent into walls and safe boundaries near holes rather than directly toward the goal, deliberately minimizing the probability of accidentally slipping into holes.

---

