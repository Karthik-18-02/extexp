import numpy as np

# Define the grid world environment
class GridWorld:
    def __init__(self, grid_size, rewards, terminal_states):
        self.grid_size = grid_size
        self.rewards = rewards
        self.terminal_states = terminal_states
        self.actions = ['up', 'down', 'left', 'right']
        self.action_prob = 1.0  # Deterministic transitions

    def step(self, state, action):
        if state in self.terminal_states:
            return state, self.rewards[state]
        
        x, y = state
        if action == 'up':
            x = max(0, x - 1)
        elif action == 'down':
            x = min(self.grid_size[0] - 1, x + 1)
        elif action == 'left':
            y = max(0, y - 1)
        elif action == 'right':
            y = min(self.grid_size[1] - 1, y + 1)
        
        next_state = (x, y)
        return next_state, self.rewards[next_state]

# Initialize the environment
grid_size = (4, 4)
rewards = np.zeros(grid_size)
rewards[3, 3] = 1  # Goal state
terminal_states = [(3, 3)]
env = GridWorld(grid_size, rewards, terminal_states)

# Initialize the policy and value function
policy = {}
value_function = np.zeros(grid_size)
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        policy[(i, j)] = np.random.choice(env.actions)

def policy_evaluation(policy, env, value_function, discount_factor=1.0, theta=1e-6):
    while True:
        delta = 0
        for state in policy.keys():
            if state in env.terminal_states:
                continue
            v = value_function[state]
            new_v = 0
            action = policy[state]
            next_state, reward = env.step(state, action)
            new_v = reward + discount_factor * value_function[next_state]
            value_function[state] = new_v
            delta = max(delta, abs(v - new_v))
        if delta < theta:
            break
    return value_function

def policy_improvement(policy, env, value_function, discount_factor=1.0):
    policy_stable = True
    for state in policy.keys():
        if state in env.terminal_states:
            continue
        old_action = policy[state]
        action_values = []
        for action in env.actions:
            next_state, reward = env.step(state, action)
            action_values.append(reward + discount_factor * value_function[next_state])
        best_action = env.actions[np.argmax(action_values)]
        policy[state] = best_action
        if old_action != best_action:
            policy_stable = False
    return policy, policy_stable

def policy_iteration(env, discount_factor=1.0):
    policy = {}
    for i in range(env.grid_size[0]):
        for j in range(env.grid_size[1]):
            policy[(i, j)] = np.random.choice(env.actions)
    value_function = np.zeros(env.grid_size)

    while True:
        value_function = policy_evaluation(policy, env, value_function, discount_factor)
        policy, policy_stable = policy_improvement(policy, env, value_function, discount_factor)
        if policy_stable:
            break
    return policy, value_function

# Run policy iteration
optimal_policy, optimal_value_function = policy_iteration(env)
print("Optimal Policy:")
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        print(optimal_policy[(i, j)], end=" ")
    print()
print("Optimal Value Function:")
print(optimal_value_function)
