import numpy as np

np.random.seed(0)

class GridWorld:
    def __init__(self):
        self.size = 6
        self.states = [(i, j) for i in range(self.size) for j in range(self.size)]
        self.actions = ['left', 'right', 'up', 'down']
        self.gamma = 0.9
        self.theta = 0.001

        # 规则定义
        self.rules1 = {
            (0, 0): ((0, 5), 1),  # A 到 B
            (0, 5): ((5, 5), 1),  # B 到 C
            (5, 5): ((5, 0), 1),  # C 到 D
            (5, 0): ((0, 0), 1)   # D 到 A
        }
        self.rules2 = {
            (1, 1): ((1, 2), 5),  # E 到 F
            (1, 4): ((1, 3), 5),  # G 到 H
            (4, 2): ((4, 3), 10), # I 到 J
            (4, 3): ((4, 2), 10)  # J 到 I
        }

    def step(self, state, action):
        if state in self.rules2 and (
            (state == (1, 1) and action == 'right') or
            (state == (1, 4) and action == 'left') or
            (state == (4, 2) and action == 'right') or
            (state == (4, 3) and action == 'left')
        ):
            next_state, reward = self.rules2[state]
            return next_state, reward

        if state in self.rules1:
            next_state, reward = self.rules1[state]
            return next_state, reward

        x, y = state
        if action == 'left':
            y = max(y - 1, 0)
        elif action == 'right':
            y = min(y + 1, self.size - 1)
        elif action == 'up':
            x = max(x - 1, 0)
        elif action == 'down':
            x = min(x + 1, self.size - 1)

        if (x, y) == state:
            return (x, y), -1  # 碰到边界
        else:
            return (x, y), 0  # 正常移动

# 策略迭代
def policy_iteration(grid_world):
    V = np.zeros((grid_world.size, grid_world.size))
    policy = np.random.choice(grid_world.actions, (grid_world.size, grid_world.size))
    is_policy_stable = False

    while not is_policy_stable:
        # 评估当前策略
        while True:
            delta = 0
            for x in range(grid_world.size):
                for y in range(grid_world.size):
                    v = V[x, y]
                    next_state, reward = grid_world.step((x, y), policy[x, y])
                    V[x, y] = reward + grid_world.gamma * V[next_state]
                    delta = max(delta, abs(v - V[x, y]))
            if delta < grid_world.theta:
                break

        # 改进策略
        is_policy_stable = True
        for x in range(grid_world.size):
            for y in range(grid_world.size):
                old_action = policy[x, y]
                action_values = []
                for action in grid_world.actions:
                    next_state, reward = grid_world.step((x, y), action)
                    action_values.append(reward + grid_world.gamma * V[next_state])
                best_action = grid_world.actions[np.argmax(action_values)]
                policy[x, y] = best_action
                if old_action != best_action:
                    is_policy_stable = False

    return V, policy

if __name__ == "__main__":
    grid_world = GridWorld()
    optimal_value_table, optimal_policy = policy_iteration(grid_world)

    print("The optimal value table for this Grid world is:")
    for row in optimal_value_table:
        print(" ".join(f"{val:.1f}" for val in row))
