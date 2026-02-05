"""
Bubble Hockey - Q-Learning Version
- Discretized 4D action space (11 steps per component)
- Random puck position within small range
- Learns to hit puck into goal using Q-learning
"""

import pygame
import sys
import math
import random
import numpy as np
import pickle

# --- Constants ---
SCALE = 25
WIDTH = int(18 * SCALE)
HEIGHT = int(33.25 * SCALE)
PUCK_DIAMETER = 1 * SCALE
PLAYER_DIAMETER = 1 * SCALE
STICK_LENGTH = 1.75 * SCALE
GOAL_WIDTH = 3.75 * SCALE
TOP_GOAL_LINE_Y = 5.5 * SCALE
BOTTOM_GOAL_LINE_Y = HEIGHT - (5.5 * SCALE)

# Discretization steps
ACTION_STEPS = np.linspace(0.0, 1.0, 11)
ACTION_SPACE = [(a, b, c, d) for a in ACTION_STEPS for b in ACTION_STEPS[5:] for c in ACTION_STEPS for d in ACTION_STEPS[5:]]

# Q-learning settings
EPISODES = 5000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1

# Fixed puck position range
PUCK_X_RANGE = (260, 270)
PUCK_Y_RANGE = (570, 585)

# Simple state (fixed to 1 bin)
STATE = 0
NUM_ACTIONS = len(ACTION_SPACE)
Q_table = np.zeros((1, NUM_ACTIONS))

# --- Classes for Player and Puck (simplified for training only) ---
class Player:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = (HEIGHT / 2) + PLAYER_DIAMETER / 2
        self.angle = -math.pi / 2
        self.radius = PLAYER_DIAMETER / 2
        self.max_linear_speed = 5
        self.max_rotation_speed = 0.5

    def do_action(self, action):
        dist, dist_speed, angle_norm, angle_speed = action
        range_limit = 9 * SCALE
        min_y = (HEIGHT / 2) + self.radius
        max_y = min(min_y + range_limit, BOTTOM_GOAL_LINE_Y - self.radius)
        target_y = min_y + (dist * (max_y - min_y))
        self.y = target_y

        angle_change = angle_norm * math.pi
        self.angle += angle_change
        self.angle = math.fmod(self.angle, 2 * math.pi)
        if self.angle > math.pi:
            self.angle -= 2 * math.pi
        elif self.angle < -math.pi:
            self.angle += 2 * math.pi

class Puck:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = PUCK_DIAMETER / 2
        self.vx = 0
        self.vy = 0
        self.friction = 0.9

    def move(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= self.friction
        self.vy *= self.friction

    def collide(self, player):
        dx = math.cos(player.angle)
        dy = math.sin(player.angle)
        stick_x = player.x + dx * (player.radius + STICK_LENGTH)
        stick_y = player.y + dy * (player.radius + STICK_LENGTH)

        dist = math.hypot(self.x - stick_x, self.y - stick_y)
        if dist < self.radius + 3:
            angle = math.atan2(self.y - stick_y, self.x - stick_x)
            self.vx = math.cos(angle) * 15
            self.vy = math.sin(angle) * 15

    def is_goal(self):
        goal_left = WIDTH / 2 - GOAL_WIDTH / 2
        goal_right = WIDTH / 2 + GOAL_WIDTH / 2
        return (self.y + self.radius >= BOTTOM_GOAL_LINE_Y and goal_left < self.x < goal_right)

# --- Training loop ---
def simulate(action_idx):
    action = ACTION_SPACE[action_idx]
    puck_x = random.uniform(*PUCK_X_RANGE)
    puck_y = random.uniform(*PUCK_Y_RANGE)

    player = Player()
    puck = Puck(puck_x, puck_y)

    player.do_action(action)
    puck.collide(player)

    for _ in range(100):
        puck.move()
        if puck.is_goal():
            return 1  # Reward
        if abs(puck.vx) < 0.1 and abs(puck.vy) < 0.1:
            break
    return 0

print("Starting Q-learning...")
for episode in range(EPISODES):
    if random.random() < EPSILON:
        action_idx = random.randint(0, NUM_ACTIONS - 1)
    else:
        action_idx = np.argmax(Q_table[STATE])

    reward = simulate(action_idx)
    best_next = np.max(Q_table[STATE])
    Q_table[STATE][action_idx] += ALPHA * (reward + GAMMA * best_next - Q_table[STATE][action_idx])

    if (episode + 1) % 500 == 0:
        print(f"Episode {episode + 1}, max Q: {np.max(Q_table):.3f}")

print("Training complete. Best action:")
best_idx = np.argmax(Q_table[STATE])
print("Action:", ACTION_SPACE[best_idx])

with open('qtable.pkl', 'wb') as f:
    pickle.dump(Q_table, f)
print("Q-table saved to qtable.pkl")

# Visualize result (optional): integrate with original pygame loop to show the best action
# For now, you could copy ACTION_SPACE[best_idx] into your original brute-force visualizer.