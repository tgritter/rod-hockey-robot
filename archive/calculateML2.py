import math
import random
import numpy as np

# Constants
PUCK_X_RANGE = (260, 270)
PUCK_Y_RANGE = (570, 585)
GOAL_X_RANGE = (180, 330)
GOAL_Y = 0

# Discrete steps for each action component
VALUES = [round(i * 0.1, 2) for i in range(11)]
ACTION_SPACE = [
    (ld, ls, ra, rs)
    for ld in VALUES
    for ls in VALUES
    for ra in VALUES
    for rs in VALUES
]

class Puck:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0

    def move(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.95
        self.vy *= 0.95

    def collide(self, player):
        dx = self.x - player.x
        dy = self.y - player.y
        distance = math.hypot(dx, dy)
        if distance < 30:
            speed = player.linear_speed_norm * 10
            angle = player.rotation_angle_norm * math.pi
            self.vx = speed * math.cos(angle)
            self.vy = -abs(speed * math.sin(angle))

    def is_goal(self):
        return GOAL_X_RANGE[0] <= self.x <= GOAL_X_RANGE[1] and self.y <= GOAL_Y

class Player:
    def __init__(self):
        self.x = 250
        self.y = 600
        self.linear_distance_norm = 0
        self.linear_speed_norm = 0
        self.rotation_angle_norm = 0
        self.rotation_speed_norm = 0

    def do_action(self, action):
        ld, ls, ra, rs = action
        self.linear_distance_norm = ld
        self.linear_speed_norm = ls
        self.rotation_angle_norm = ra
        self.rotation_speed_norm = rs

        dx = (ld - 0.5) * 100
        self.x += dx

        da = (ra - 0.5) * math.pi
        self.y -= abs(da) * 5

def simulate_action(action_idx, puck_x, puck_y):
    action = ACTION_SPACE[action_idx]

    player = Player()
    puck = Puck(puck_x, puck_y)

    player.do_action(action)
    puck.collide(player)

    for _ in range(100):
        puck.move()
        if puck.is_goal():
            return 1
        if abs(puck.vx) < 0.1 and abs(puck.vy) < 0.1:
            break
    return 0

def simulate_custom(action_idx, puck_x, puck_y):
    action = ACTION_SPACE[action_idx]

    player = Player()
    puck = Puck(puck_x, puck_y)

    player.do_action(action)
    puck.collide(player)

    for _ in range(100):
        puck.move()
        if puck.is_goal():
            return True
        if abs(puck.vx) < 0.1 and abs(puck.vy) < 0.1:
            break
    return False

def train_q_learning(episodes=5000):
    Q_table = np.zeros(len(ACTION_SPACE))
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.2

    for _ in range(episodes):
        puck_x = random.uniform(*PUCK_X_RANGE)
        puck_y = random.uniform(*PUCK_Y_RANGE)

        if random.random() < epsilon:
            action_idx = random.randint(0, len(ACTION_SPACE) - 1)
        else:
            action_idx = np.argmax(Q_table)

        reward = simulate_action(action_idx, puck_x, puck_y)
        Q_table[action_idx] = Q_table[action_idx] + alpha * (reward - Q_table[action_idx])

    return Q_table

if __name__ == "__main__":
    Q_table = train_q_learning()
    best_idx = np.argmax(Q_table)
    print("Best action:", ACTION_SPACE[best_idx])

    # Custom inference
    custom_x = 266
    custom_y = 578
    success = simulate_custom(best_idx, custom_x, custom_y)
    print(f"Custom puck ({custom_x}, {custom_y}) goal?", "✅" if success else "❌")
