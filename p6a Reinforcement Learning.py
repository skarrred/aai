#p6a
#A)Applying reinforcement learning algorithm to solve complex decision-making problems
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

# Define environment
states = ["A", "B", "C", "D", "E", "F"]
actions = ["left", "right"]
rewards = {"A": 0, "B": 0, "C": 0, "D": 10, "E": -10, "F": 0}

# Initialize Q-table
Q = {}
for s in states:
    Q[s] = {a: 0 for a in actions}

# Parameters
alpha = 0.5  # Learning rate
gamma = 0.9  # Discount factor
episodes = 20

# Define transitions
def next_state(state, action):
    if state == "A":
        return "B" if action == "right" else "A"
    elif state == "B":
        return "C" if action == "right" else "A"
    elif state == "C":
        return "D" if action == "right" else "B"
    elif state == "D":
        return "D"
    elif state == "E":
        return "E"
    elif state == "F":
        return "E" if action == "left" else "F"

# Training
for episode in range(episodes):
    state = "A"
    done = False
    while not done:
        action = random.choice(actions)
        next_s = next_state(state, action)
        reward = rewards[next_s]
        old_value = Q[state][action]

        next_max = max(Q[next_s].values())
        # Q-learning update
        Q[state][action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_s
        if state == "D" or state == "E":
            done = True

print("Learned Q-values:")
for s in Q:
    print(s, Q[s])

# Convert the Q-table dictionary to a pandas DataFrame for easier plotting
q_df = pd.DataFrame.from_dict(Q, orient='index')

# Plotting the Q-values
q_df.plot(kind='bar', figsize=(10, 6))
plt.title('Learned Q-values per State and Action')
plt.xlabel('State')
plt.ylabel('Q-value')
plt.xticks(rotation=0)
plt.legend(title='Action')
plt.grid(axis='y')
plt.show()

