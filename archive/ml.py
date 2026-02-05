import pygame
import sys
import math
import random
import copy
import argparse
import numpy as np
import pickle
import os
from collections import deque

# Parse optional puck coordinates from the command line
parser = argparse.ArgumentParser(description="Run Bubble Hockey simulation with ML.")
parser.add_argument("--puck_x", type=float, help="Initial puck x-coordinate in pixels")
parser.add_argument("--puck_y", type=float, help="Initial puck y-coordinate in pixels")
parser.add_argument("--train", action='store_true', help="Train the model before predicting")
parser.add_argument("--train_samples", type=int, default=200, help="Number of training samples to collect")
parser.add_argument("--model_file", type=str, default="hockey_model.pkl", help="File to save/load model")
args = parser.parse_args()

pygame.init()

# Define scale (pixels per inch)
SCALE = 25  # This gives a good size on screen

# Game dimensions based on real measurements
GOAL_WIDTH = 3.75 * SCALE  # 3 3/4 inches
HALF_FIELD_LENGTH = 16.75 * SCALE  # 13 1/2 inches from goal to center
PLAYER_DIAMETER = 1 * SCALE  # 1 inch
STICK_LENGTH = 1.75 * SCALE  # 1 3/4 inches
PUCK_DIAMETER = 1 * SCALE  # 1 inch
PLAYER_MIN_DIST_FROM_GOAL = 4 * SCALE  # 4 inches from goal
BEHIND_GOAL_SPACE = 5.5 * SCALE  # 3 inches of space behind each goal

# Calculate full dimensions
WIDTH = 18 * SCALE  # Width of the field (approximate based on aspect ratio)
HEIGHT = 31.5 * SCALE  # Add behind-goal space

# Goal positions (moved in from the edges)
TOP_GOAL_Y = BEHIND_GOAL_SPACE
BOTTOM_GOAL_Y = HEIGHT - BEHIND_GOAL_SPACE - SCALE / 2

screen = pygame.display.set_mode((int(WIDTH), int(HEIGHT)))
pygame.display.set_caption("Bubble Hockey - ML Training")
clock = pygame.time.Clock()

WHITE = (255, 255, 255)
BLUE = (66, 135, 245)
RED = (245, 66, 66)
BLACK = (0, 0, 0)
LIGHT_GRAY = (200, 200, 200)

class SimpleNeuralNetwork:
    """A simple neural network for predicting actions from puck positions."""
    
    def __init__(self, input_size=2, hidden_size=64, output_size=4):
        # Initialize weights with Xavier initialization
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, hidden_size))
        self.w3 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros((1, output_size))
        
        self.training_data = []
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, x):
        """Forward pass through the network."""
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.relu(self.z2)
        
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        
        # Apply appropriate activation to each output
        # linear_distance: sigmoid (0-1)
        # linear_speed: sigmoid scaled to (0.5-1)
        # rotation_angle: tanh (-1 to 1)
        # rotation_speed: sigmoid scaled to (0.5-1)
        output = np.zeros_like(self.z3)
        output[:, 0] = self.sigmoid(self.z3[:, 0])
        output[:, 1] = 0.5 + 0.5 * self.sigmoid(self.z3[:, 1])
        output[:, 2] = self.tanh(self.z3[:, 2])
        output[:, 3] = 0.5 + 0.5 * self.sigmoid(self.z3[:, 3])
        
        return output
    
    def predict(self, puck_x, puck_y):
        """Predict action for given puck position."""
        # Normalize inputs
        x_norm = puck_x / WIDTH
        y_norm = puck_y / HEIGHT
        x = np.array([[x_norm, y_norm]])
        
        output = self.forward(x)
        return output[0].tolist()
    
    def add_training_sample(self, puck_x, puck_y, action):
        """Add a successful action to training data."""
        self.training_data.append((puck_x, puck_y, action))
    
    def save(self, filename):
        """Save the model to a file."""
        model_data = {
            'w1': self.w1,
            'b1': self.b1,
            'w2': self.w2,
            'b2': self.b2,
            'w3': self.w3,
            'b3': self.b3,
            'training_data': self.training_data
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")
    
    def load(self, filename):
        """Load the model from a file."""
        if not os.path.exists(filename):
            return False
        
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.w1 = model_data['w1']
        self.b1 = model_data['b1']
        self.w2 = model_data['w2']
        self.b2 = model_data['b2']
        self.w3 = model_data['w3']
        self.b3 = model_data['b3']
        self.training_data = model_data['training_data']
        print(f"Model loaded from {filename} ({len(self.training_data)} training samples)")
        return True
    
    def train(self, epochs=100, learning_rate=0.01, batch_size=32):
        """Train the network on collected data."""
        if len(self.training_data) == 0:
            print("No training data available!")
            return
        
        print(f"Training on {len(self.training_data)} samples...")
        
        # Prepare training data
        X = np.array([[x/WIDTH, y/HEIGHT] for x, y, _ in self.training_data])
        y = np.array([action for _, _, action in self.training_data])
        
        n_samples = len(X)
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            total_loss = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                # Forward pass
                z1 = np.dot(batch_X, self.w1) + self.b1
                a1 = self.relu(z1)
                
                z2 = np.dot(a1, self.w2) + self.b2
                a2 = self.relu(z2)
                
                z3 = np.dot(a2, self.w3) + self.b3
                
                # Apply activations
                predictions = np.zeros_like(z3)
                predictions[:, 0] = self.sigmoid(z3[:, 0])
                predictions[:, 1] = 0.5 + 0.5 * self.sigmoid(z3[:, 1])
                predictions[:, 2] = self.tanh(z3[:, 2])
                predictions[:, 3] = 0.5 + 0.5 * self.sigmoid(z3[:, 3])
                
                # Loss (MSE)
                loss = np.mean((predictions - batch_y) ** 2)
                total_loss += loss
                
                # Backward pass
                d_output = 2 * (predictions - batch_y) / len(batch_y)
                
                # Gradient through activations
                d_z3 = np.zeros_like(z3)
                sig0 = self.sigmoid(z3[:, 0])
                d_z3[:, 0] = d_output[:, 0] * sig0 * (1 - sig0)
                sig1 = self.sigmoid(z3[:, 1])
                d_z3[:, 1] = d_output[:, 1] * 0.5 * sig1 * (1 - sig1)
                d_z3[:, 2] = d_output[:, 2] * (1 - self.tanh(z3[:, 2]) ** 2)
                sig3 = self.sigmoid(z3[:, 3])
                d_z3[:, 3] = d_output[:, 3] * 0.5 * sig3 * (1 - sig3)
                
                d_w3 = np.dot(a2.T, d_z3)
                d_b3 = np.sum(d_z3, axis=0, keepdims=True)
                d_a2 = np.dot(d_z3, self.w3.T)
                
                d_z2 = d_a2 * (z2 > 0)
                d_w2 = np.dot(a1.T, d_z2)
                d_b2 = np.sum(d_z2, axis=0, keepdims=True)
                d_a1 = np.dot(d_z2, self.w2.T)
                
                d_z1 = d_a1 * (z1 > 0)
                d_w1 = np.dot(batch_X.T, d_z1)
                d_b1 = np.sum(d_z1, axis=0, keepdims=True)
                
                # Update weights
                self.w3 -= learning_rate * d_w3
                self.b3 -= learning_rate * d_b3
                self.w2 -= learning_rate * d_w2
                self.b2 -= learning_rate * d_b2
                self.w1 -= learning_rate * d_w1
                self.b1 -= learning_rate * d_b1
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / max(1, (n_samples / batch_size))
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

# Global neural network
neural_net = SimpleNeuralNetwork()

class Player:
    def __init__(self, x, y, team, is_goalie=False, min_y=None, max_y=None):
        self.x = x
        self.y = y
        self.start_x = x - SCALE * 0.5
        self.start_y = y
        self.radius = PLAYER_DIAMETER / 2
        self.team = team
        self.angle = -math.pi / 2
        self.is_goalie = is_goalie

        self.min_y = min_y if min_y is not None else BEHIND_GOAL_SPACE
        self.max_y = max_y if max_y is not None else HEIGHT - BEHIND_GOAL_SPACE
        self.range_limit = HALF_FIELD_LENGTH - BEHIND_GOAL_SPACE - PLAYER_MIN_DIST_FROM_GOAL

        self.target_x = self.start_x
        self.target_y = y
        self.target_angle = self.angle
        self.movement_in_progress = False
        self.position_reached = False

        self.max_linear_speed = 5
        self.max_rotation_speed = 0.5
        self.has_rotated = False
        self.has_hit_puck = False

    def draw(self, screen):
        color = BLUE if self.team == 'blue' else RED
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), int(self.radius))
        dx = math.cos(self.angle)
        dy = math.sin(self.angle)
        start_x = self.x + dx * self.radius * 0.5
        start_y = self.y + dy * self.radius * 0.5
        end_x = self.x + dx * (self.radius * 0.5 + STICK_LENGTH)
        end_y = self.y + dy * (self.radius * 0.5 + STICK_LENGTH)
        pygame.draw.line(screen, BLACK, (start_x, start_y), (end_x, end_y), 3)

    def do_action(self, action):
        linear_distance, linear_speed, rotation_angle, rotation_speed = action
        self.has_rotated = False
        self.has_hit_puck = False
        
        self.x = self.start_x
        self.y = HEIGHT // 2
        self.angle = -math.pi / 2

        if not self.is_goalie:
            if linear_speed > 0:
                desired_distance = linear_distance * self.range_limit
                self.movement_speed = linear_speed * self.max_linear_speed
                target_y = self.start_y + desired_distance
                min_allowed = max(self.start_y - self.range_limit, self.min_y + self.radius)
                max_allowed = min(self.start_y + self.range_limit, self.max_y - self.radius)
                self.target_y = max(min_allowed, min(max_allowed, target_y))
                
                progress = (self.target_y - self.min_y) / (self.max_y - self.min_y)
                self.target_x = self.start_x + progress * SCALE * 0.5

            if rotation_speed > 0:
                desired_rotation = rotation_angle * 2 * math.pi
                self.rotation_speed = rotation_speed * self.max_rotation_speed
                self.target_angle = self.angle + desired_rotation

            self.movement_in_progress = True
            self.position_reached = False

        return self.x, self.y, self.angle

    def update(self):
        if not self.is_goalie and self.movement_in_progress:
            move_speed = getattr(self, 'movement_speed', 0)
            rot_speed = getattr(self, 'rotation_speed', 0)

            if not self.position_reached:
                if abs(self.y - self.target_y) > move_speed:
                    direction = 1 if self.target_y > self.y else -1
                    self.y += direction * move_speed
                    
                    progress = (self.y - self.min_y) / (self.max_y - self.min_y)
                    progress = max(0, min(1, progress))
                    self.x = self.start_x + progress * SCALE * 0.5
                else:
                    self.y = self.target_y
                    progress = (self.y - self.min_y) / (self.max_y - self.min_y)
                    progress = max(0, min(1, progress))
                    self.x = self.start_x + progress * SCALE * 0.5
                    self.position_reached = True
            elif abs(self.angle - self.target_angle) > rot_speed:
                direction = 1 if self.target_angle > self.angle else -1
                self.angle += direction * rot_speed
                self.has_rotated = True
            else:
                self.angle = self.target_angle
                self.movement_in_progress = False
                self.position_reached = False
                if self.angle != -math.pi / 2:
                    self.has_rotated = True
        return self.x, self.y, self.angle

class Puck:
    def __init__(self, x=None, y=None):
        self.x = x if x is not None else WIDTH // 2
        self.y = y if y is not None else HEIGHT // 2
        self.radius = PUCK_DIAMETER / 2
        self.vx = 0
        self.vy = 0
        self.friction = 0.9
        self.was_hit = False

    def move(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= self.friction
        self.vy *= self.friction

        if abs(self.vx) < 0.1: self.vx = 0
        if abs(self.vy) < 0.1: self.vy = 0

        if self.x - self.radius < 0 or self.x + self.radius > WIDTH:
            self.vx *= -1
            self.x = max(self.radius, min(WIDTH - self.radius, self.x))

        goal_left = WIDTH / 2 - GOAL_WIDTH / 2
        goal_right = WIDTH / 2 + GOAL_WIDTH / 2

        if self.y - self.radius < 0:
            self.vy *= -1
            self.y = self.radius
        elif self.y - self.radius < TOP_GOAL_Y:
            if not (goal_left < self.x < goal_right):
                self.vy *= -1
                self.y = TOP_GOAL_Y + self.radius

        if self.y + self.radius > HEIGHT:
            self.vy *= -1
            self.y = HEIGHT - self.radius
        elif self.y + self.radius > BOTTOM_GOAL_Y + SCALE / 2:
            if not (goal_left < self.x < goal_right):
                self.vy *= -1
                self.y = BOTTOM_GOAL_Y + SCALE / 2 - self.radius

    def draw(self, screen):
        pygame.draw.circle(screen, (50, 50, 50), (int(self.x), int(self.y)), int(self.radius))
        pygame.draw.circle(screen, BLACK, (int(self.x), int(self.y)), int(self.radius), 2)

    def collide(self, player):
        stick_dx = math.cos(player.angle)
        stick_dy = math.sin(player.angle)
        start_x = player.x + stick_dx * player.radius * 0.5
        start_y = player.y + stick_dy * player.radius * 0.5
        end_x = player.x + stick_dx * (player.radius * 0.5 + STICK_LENGTH)
        end_y = player.y + stick_dy * (player.radius * 0.5 + STICK_LENGTH)

        px, py = self.x, self.y
        lx = end_x - start_x
        ly = end_y - start_y
        line_len_squared = lx ** 2 + ly ** 2
        t = max(0, min(1, ((px - start_x) * lx + (py - start_y) * ly) / line_len_squared))
        closest_x = start_x + t * lx
        closest_y = start_y + t * ly
        dist = math.hypot(px - closest_x, py - closest_y)

        if dist < self.radius + 3:
            norm_angle = math.atan2(py - closest_y, px - closest_x)
            speed = 6
            self.vx = math.cos(norm_angle) * speed
            self.vy = math.sin(norm_angle) * speed
            self.x = closest_x + math.cos(norm_angle) * (self.radius + 3)
            self.y = closest_y + math.sin(norm_angle) * (self.radius + 3)
            self.was_hit = True
            player.has_hit_puck = True
            return True
        return False

min_y = HEIGHT / 2
max_y = HEIGHT - PLAYER_MIN_DIST_FROM_GOAL - BEHIND_GOAL_SPACE
players = [Player(WIDTH // 2, min_y, 'blue', min_y=min_y, max_y=max_y)]
selected_blue = 0

def draw_field(screen):
    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, HEIGHT), 3)
    
    center_y = HEIGHT / 2
    pygame.draw.line(screen, BLACK, (0, center_y), (WIDTH, center_y), 2)
    pygame.draw.circle(screen, BLACK, (WIDTH / 2, center_y), SCALE * 2, 2)

    pygame.draw.line(screen, RED, (0, TOP_GOAL_Y), (WIDTH, TOP_GOAL_Y), 2)
    pygame.draw.line(screen, RED, (0, BOTTOM_GOAL_Y + SCALE / 2), (WIDTH, BOTTOM_GOAL_Y + SCALE / 2), 2)

    goal_x = WIDTH / 2 - GOAL_WIDTH / 2
    pygame.draw.rect(screen, BLACK, (goal_x, TOP_GOAL_Y, GOAL_WIDTH, SCALE / 2), 2)
    pygame.draw.rect(screen, BLACK, (goal_x, BOTTOM_GOAL_Y, GOAL_WIDTH, SCALE / 2), 2)

    for i in range(1, 5):
        x = goal_x + i * (GOAL_WIDTH / 5)
        pygame.draw.line(screen, BLACK, (x, TOP_GOAL_Y), (x, TOP_GOAL_Y + SCALE / 2), 1)
        pygame.draw.line(screen, BLACK, (x, BOTTOM_GOAL_Y), (x, BOTTOM_GOAL_Y + SCALE / 2), 1)

def simulate_action(action, puck_x, puck_y):
    test_player = copy.deepcopy(players[0])
    test_puck = Puck(x=puck_x, y=puck_y)
    test_player.do_action(action)
    hit_after_rotation = False

    for _ in range(10000):
        test_player.update()
        test_puck.move()
        if test_puck.collide(test_player) and test_player.has_rotated:
            hit_after_rotation = True
        if hit_after_rotation and test_puck.y + test_puck.radius > BOTTOM_GOAL_Y + SCALE / 2:
            if WIDTH / 2 - GOAL_WIDTH / 2 < test_puck.x < WIDTH / 2 + GOAL_WIDTH / 2:
                return True
    return False

def smart_action_generator(puck_x, puck_y):
    """Generate smarter actions based on puck position."""
    # Player starts at center (WIDTH/2, HEIGHT/2)
    player_start_x = WIDTH / 2 - SCALE * 0.5
    player_start_y = HEIGHT / 2
    
    # Calculate distance to puck
    dx = puck_x - player_start_x
    dy = puck_y - player_start_y
    
    # Normalize distance to move (0 to 1 range)
    # Player can move roughly from HEIGHT/2 to HEIGHT - PLAYER_MIN_DIST_FROM_GOAL
    max_move_distance = HEIGHT - PLAYER_MIN_DIST_FROM_GOAL - BEHIND_GOAL_SPACE - player_start_y
    
    # Base distance: try to get close to puck's y position
    target_distance = dy / max_move_distance
    linear_distance = np.clip(target_distance + random.uniform(-0.15, 0.15), 0.1, 0.9)
    
    # Always use high speed to reach position quickly
    linear_speed = random.uniform(0.8, 1.0)
    
    # Calculate angle to hit puck toward goal
    # Goal is at (WIDTH/2, BOTTOM_GOAL_Y)
    goal_center_x = WIDTH / 2
    goal_y = BOTTOM_GOAL_Y
    
    # Estimate player final position
    final_player_y = player_start_y + linear_distance * max_move_distance
    final_player_x = player_start_x + (linear_distance * max_move_distance / max_move_distance) * SCALE * 0.5
    
    # Vector from player to puck
    to_puck_x = puck_x - final_player_x
    to_puck_y = puck_y - final_player_y
    
    # Vector from puck to goal
    puck_to_goal_x = goal_center_x - puck_x
    puck_to_goal_y = goal_y - puck_y
    
    # Desired angle is toward the puck, but with consideration of goal direction
    desired_angle = math.atan2(to_puck_y, to_puck_x)
    
    # Player starts pointing at -pi/2 (upward), calculate rotation needed
    rotation_needed = desired_angle - (-math.pi / 2)
    
    # Normalize to [-pi, pi]
    while rotation_needed > math.pi:
        rotation_needed -= 2 * math.pi
    while rotation_needed < -math.pi:
        rotation_needed += 2 * math.pi
    
    # Convert to [-1, 1] range (divide by 2*pi and multiply by 2)
    rotation_angle = np.clip((rotation_needed / (2 * math.pi)) + random.uniform(-0.2, 0.2), -0.8, 0.8)
    
    # Use high rotation speed
    rotation_speed = random.uniform(0.8, 1.0)
    
    return [linear_distance, linear_speed, rotation_angle, rotation_speed]

def collect_training_data(num_samples=200):
    """Collect successful actions for random puck positions."""
    print(f"Collecting {num_samples} successful training samples...")
    samples_collected = 0
    attempts = 0
    max_attempts_per_position = 1000  # Increased from 500
    
    while samples_collected < num_samples:
        # Random puck position in upper half of field
        # Focus on more reachable areas
        puck_x = random.uniform(200, 250)
        puck_y = random.uniform(530, 550)
        
        position_attempts = 0
        found_for_position = False
        
        # Try smart actions for this position
        while position_attempts < max_attempts_per_position and not found_for_position:
            attempts += 1
            position_attempts += 1
            
            # Use smart generator 90% of the time, random 10%
            if random.random() < 0.9:
                action = smart_action_generator(puck_x, puck_y)
            else:
                action = [
                    random.uniform(0.2, 0.8),
                    random.uniform(0.7, 1),
                    random.uniform(-0.8, 0.8),
                    random.uniform(0.7, 1)
                ]
            
            if simulate_action(action, puck_x, puck_y):
                neural_net.add_training_sample(puck_x, puck_y, action)
                samples_collected += 1
                found_for_position = True
                if samples_collected % 10 == 0 or samples_collected <= 5:
                    print(f"Sample {samples_collected}/{num_samples} - Position attempts: {position_attempts}, Total: {attempts}, X: {puck_x:.1f}, Y: {puck_y:.1f}")
                break
        
        if not found_for_position:
            print(f"  Skipping difficult position ({puck_x:.0f}, {puck_y:.0f}) after {position_attempts} attempts...")
    
    print(f"Training data collection complete! Total attempts: {attempts}")
    print(f"Average attempts per sample: {attempts/num_samples:.1f}")

def visualize_episode(action, puck_x, puck_y, duration=120):
    """Visualize a single episode."""
    players[0].do_action(action)
    puck = Puck(x=puck_x, y=puck_y)
    
    for frame in range(duration):
        screen.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        players[0].update()
        puck.move()
        puck.collide(players[0])

        draw_field(screen)
        players[0].draw(screen)
        puck.draw(screen)

        pygame.display.flip()
        clock.tick(60)


def find_action_with_ml(puck_x, puck_y, max_attempts=100):
    """Use ML to predict action - return first prediction only."""
    print(f"\nFinding action for puck at ({puck_x:.1f}, {puck_y:.1f})")
    
    # Try ML prediction first
    if len(neural_net.training_data) > 0:
        print("Trying ML prediction...")
        predicted_action = neural_net.predict(puck_x, puck_y)
        print(f"Predicted action: [ld={predicted_action[0]:.3f}, ls={predicted_action[1]:.3f}, ra={predicted_action[2]:.3f}, rs={predicted_action[3]:.3f}]")
        
        # Check if it would score (just for info)
        if simulate_action(predicted_action, puck_x, puck_y):
            print("✓ ML prediction successful!")
        else:
            print("✗ ML prediction didn't score (will visualize anyway)")
        
        return predicted_action
    
    print("✗ No trained model available")
    return None

# Main execution
# Try to load existing model first
model_loaded = neural_net.load(args.model_file)

if args.train or not model_loaded:
    if not model_loaded:
        print("=== No saved model found, training from scratch ===")
    else:
        print("=== TRAINING MODE (will add to existing model) ===")
    collect_training_data(args.train_samples)
    neural_net.train(epochs=500, learning_rate=0.01)
    neural_net.save(args.model_file)
    print("\n=== Training complete! ===\n")
elif model_loaded:
    print(f"=== Using pre-trained model with {len(neural_net.training_data)} samples ===\n")

# Test with provided or random puck position
if args.puck_x is not None and args.puck_y is not None:
    test_puck_x = args.puck_x
    test_puck_y = args.puck_y
    print(f"Testing with provided puck position: ({test_puck_x}, {test_puck_y})")
else:
    test_puck_x = random.uniform(WIDTH * 0.3, WIDTH * 0.7)
    test_puck_y = random.uniform(HEIGHT * 0.3, HEIGHT * 0.5)
    print(f"Testing with random puck position: ({test_puck_x:.1f}, {test_puck_y:.1f})")

action = find_action_with_ml(test_puck_x, test_puck_y)

if action:
    print("\nVisualizing successful episode...")
    visualize_episode(action, test_puck_x, test_puck_y, duration=180)

print("Program complete.")
pygame.quit()