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

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run Bubble Hockey simulation with improved ML.")
parser.add_argument("--puck_x", type=float, help="Initial puck x-coordinate in pixels")
parser.add_argument("--puck_y", type=float, help="Initial puck y-coordinate in pixels")
parser.add_argument("--train", action='store_true', help="Train the model before predicting")
parser.add_argument("--train_samples", type=int, default=500, help="Number of training samples to collect")
parser.add_argument("--model_file", type=str, default="hockey_model_v2.pkl", help="File to save/load model")
parser.add_argument("--visualize", action='store_true', help="Visualize predictions")
args = parser.parse_args()

pygame.init()

# Define scale (pixels per inch)
SCALE = 25

# Game dimensions
GOAL_WIDTH = 3.75 * SCALE
HALF_FIELD_LENGTH = 16.75 * SCALE
PLAYER_DIAMETER = 1 * SCALE
STICK_LENGTH = 1.75 * SCALE
PUCK_DIAMETER = 1 * SCALE
PLAYER_MIN_DIST_FROM_GOAL = 4 * SCALE
BEHIND_GOAL_SPACE = 5.5 * SCALE

WIDTH = 18 * SCALE
HEIGHT = 31.5 * SCALE

TOP_GOAL_Y = BEHIND_GOAL_SPACE
BOTTOM_GOAL_Y = HEIGHT - BEHIND_GOAL_SPACE - SCALE / 2

screen = pygame.display.set_mode((int(WIDTH), int(HEIGHT)))
pygame.display.set_caption("Bubble Hockey - Improved ML")
clock = pygame.time.Clock()

WHITE = (255, 255, 255)
BLUE = (66, 135, 245)
RED = (245, 66, 66)
BLACK = (0, 0, 0)
LIGHT_GRAY = (200, 200, 200)
GREEN = (66, 245, 66)

class ImprovedNeuralNetwork:
    """Improved neural network with better features and architecture."""
    
    def __init__(self, input_size=6, hidden_size=128, output_size=4):
        # Larger network with better initialization
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.w2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, hidden_size))
        
        self.w3 = np.random.randn(hidden_size, hidden_size // 2) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros((1, hidden_size // 2))
        
        self.w4 = np.random.randn(hidden_size // 2, output_size) * np.sqrt(2.0 / (hidden_size // 2))
        self.b4 = np.zeros((1, output_size))
        
        self.training_data = []
        self.training_history = []
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def extract_features(self, puck_x, puck_y):
        """Extract meaningful features from puck position."""
        # Normalize positions
        x_norm = puck_x / WIDTH
        y_norm = puck_y / HEIGHT
        
        # Distance and angle to goal center
        goal_x = WIDTH / 2
        goal_y = BOTTOM_GOAL_Y
        dx_to_goal = (goal_x - puck_x) / WIDTH
        dy_to_goal = (goal_y - puck_y) / HEIGHT
        dist_to_goal = math.sqrt(dx_to_goal**2 + dy_to_goal**2)
        angle_to_goal = math.atan2(dy_to_goal, dx_to_goal) / math.pi
        
        return np.array([[x_norm, y_norm, dx_to_goal, dy_to_goal, dist_to_goal, angle_to_goal]])
    
    def forward(self, x):
        """Forward pass through the network."""
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.relu(self.z2)
        
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = self.relu(self.z3)
        
        self.z4 = np.dot(self.a3, self.w4) + self.b4
        
        # Output activations
        output = np.zeros_like(self.z4)
        output[:, 0] = self.sigmoid(self.z4[:, 0])  # linear_distance (0-1)
        output[:, 1] = 0.5 + 0.5 * self.sigmoid(self.z4[:, 1])  # linear_speed (0.5-1)
        output[:, 2] = self.tanh(self.z4[:, 2])  # rotation_angle (-1 to 1)
        output[:, 3] = 0.5 + 0.5 * self.sigmoid(self.z4[:, 3])  # rotation_speed (0.5-1)
        
        return output
    
    def predict(self, puck_x, puck_y):
        """Predict action for given puck position."""
        x = self.extract_features(puck_x, puck_y)
        output = self.forward(x)
        return output[0].tolist()
    
    def add_training_sample(self, puck_x, puck_y, action, score=1.0):
        """Add a training sample with optional quality score."""
        weight = score ** 3  # 1.0^3=1.0, but 0.6^3=0.216
        self.training_data.append((puck_x, puck_y, action, weight))
    
    def save(self, filename):
        """Save the model to a file."""
        model_data = {
            'w1': self.w1, 'b1': self.b1,
            'w2': self.w2, 'b2': self.b2,
            'w3': self.w3, 'b3': self.b3,
            'w4': self.w4, 'b4': self.b4,
            'training_data': self.training_data,
            'training_history': self.training_history
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
        self.w4 = model_data['w4']
        self.b4 = model_data['b4']
        self.training_data = model_data['training_data']
        self.training_history = model_data.get('training_history', [])
        print(f"Model loaded from {filename} ({len(self.training_data)} training samples)")
        return True
    
    def train(self, epochs=200, learning_rate=0.001, batch_size=32):
        """Train the network with Adam optimizer."""
        if len(self.training_data) == 0:
            print("No training data available!")
            return
        
        print(f"\n{'='*60}")
        print(f"TRAINING on {len(self.training_data)} samples")
        print(f"Epochs: {epochs}, Learning Rate: {learning_rate}, Batch Size: {batch_size}")
        print(f"{'='*60}\n")
        
        # Prepare training data with features
        X = np.vstack([self.extract_features(x, y) for x, y, _, _ in self.training_data])
        y = np.array([action for _, _, action, _ in self.training_data])
        weights = np.array([score for _, _, _, score in self.training_data])
        
        n_samples = len(X)
        
        # Adam optimizer parameters
        m_w1, v_w1 = np.zeros_like(self.w1), np.zeros_like(self.w1)
        m_b1, v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        m_w2, v_w2 = np.zeros_like(self.w2), np.zeros_like(self.w2)
        m_b2, v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        m_w3, v_w3 = np.zeros_like(self.w3), np.zeros_like(self.w3)
        m_b3, v_b3 = np.zeros_like(self.b3), np.zeros_like(self.b3)
        m_w4, v_w4 = np.zeros_like(self.w4), np.zeros_like(self.w4)
        m_b4, v_b4 = np.zeros_like(self.b4), np.zeros_like(self.b4)
        
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            w_shuffled = weights[indices]
            
            total_loss = 0
            num_batches = 0
            
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                batch_w = w_shuffled[i:i+batch_size].reshape(-1, 1)
                
                # Forward pass
                z1 = np.dot(batch_X, self.w1) + self.b1
                a1 = self.relu(z1)
                
                z2 = np.dot(a1, self.w2) + self.b2
                a2 = self.relu(z2)
                
                z3 = np.dot(a2, self.w3) + self.b3
                a3 = self.relu(z3)
                
                z4 = np.dot(a3, self.w4) + self.b4
                
                predictions = np.zeros_like(z4)
                predictions[:, 0] = self.sigmoid(z4[:, 0])
                predictions[:, 1] = 0.5 + 0.5 * self.sigmoid(z4[:, 1])
                predictions[:, 2] = self.tanh(z4[:, 2])
                predictions[:, 3] = 0.5 + 0.5 * self.sigmoid(z4[:, 3])
                
                # Weighted loss
                loss = np.mean(batch_w * (predictions - batch_y) ** 2)
                total_loss += loss
                num_batches += 1
                
                # Backward pass
                d_output = 2 * batch_w * (predictions - batch_y) / len(batch_y)
                
                d_z4 = np.zeros_like(z4)
                sig0 = self.sigmoid(z4[:, 0])
                d_z4[:, 0] = d_output[:, 0] * sig0 * (1 - sig0)
                sig1 = self.sigmoid(z4[:, 1])
                d_z4[:, 1] = d_output[:, 1] * 0.5 * sig1 * (1 - sig1)
                d_z4[:, 2] = d_output[:, 2] * (1 - self.tanh(z4[:, 2]) ** 2)
                sig3 = self.sigmoid(z4[:, 3])
                d_z4[:, 3] = d_output[:, 3] * 0.5 * sig3 * (1 - sig3)
                
                d_w4 = np.dot(a3.T, d_z4)
                d_b4 = np.sum(d_z4, axis=0, keepdims=True)
                d_a3 = np.dot(d_z4, self.w4.T)
                
                d_z3 = d_a3 * (z3 > 0)
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
                
                # Adam update
                t = epoch * (n_samples // batch_size) + (i // batch_size) + 1
                
                # Update w4, b4
                m_w4 = beta1 * m_w4 + (1 - beta1) * d_w4
                v_w4 = beta2 * v_w4 + (1 - beta2) * (d_w4 ** 2)
                m_hat_w4 = m_w4 / (1 - beta1 ** t)
                v_hat_w4 = v_w4 / (1 - beta2 ** t)
                self.w4 -= learning_rate * m_hat_w4 / (np.sqrt(v_hat_w4) + epsilon)
                
                m_b4 = beta1 * m_b4 + (1 - beta1) * d_b4
                v_b4 = beta2 * v_b4 + (1 - beta2) * (d_b4 ** 2)
                m_hat_b4 = m_b4 / (1 - beta1 ** t)
                v_hat_b4 = v_b4 / (1 - beta2 ** t)
                self.b4 -= learning_rate * m_hat_b4 / (np.sqrt(v_hat_b4) + epsilon)
                
                # Update w3, b3
                m_w3 = beta1 * m_w3 + (1 - beta1) * d_w3
                v_w3 = beta2 * v_w3 + (1 - beta2) * (d_w3 ** 2)
                m_hat_w3 = m_w3 / (1 - beta1 ** t)
                v_hat_w3 = v_w3 / (1 - beta2 ** t)
                self.w3 -= learning_rate * m_hat_w3 / (np.sqrt(v_hat_w3) + epsilon)
                
                m_b3 = beta1 * m_b3 + (1 - beta1) * d_b3
                v_b3 = beta2 * v_b3 + (1 - beta2) * (d_b3 ** 2)
                m_hat_b3 = m_b3 / (1 - beta1 ** t)
                v_hat_b3 = v_b3 / (1 - beta2 ** t)
                self.b3 -= learning_rate * m_hat_b3 / (np.sqrt(v_hat_b3) + epsilon)
                
                # Update w2, b2
                m_w2 = beta1 * m_w2 + (1 - beta1) * d_w2
                v_w2 = beta2 * v_w2 + (1 - beta2) * (d_w2 ** 2)
                m_hat_w2 = m_w2 / (1 - beta1 ** t)
                v_hat_w2 = v_w2 / (1 - beta2 ** t)
                self.w2 -= learning_rate * m_hat_w2 / (np.sqrt(v_hat_w2) + epsilon)
                
                m_b2 = beta1 * m_b2 + (1 - beta1) * d_b2
                v_b2 = beta2 * v_b2 + (1 - beta2) * (d_b2 ** 2)
                m_hat_b2 = m_b2 / (1 - beta1 ** t)
                v_hat_b2 = v_b2 / (1 - beta2 ** t)
                self.b2 -= learning_rate * m_hat_b2 / (np.sqrt(v_hat_b2) + epsilon)
                
                # Update w1, b1
                m_w1 = beta1 * m_w1 + (1 - beta1) * d_w1
                v_w1 = beta2 * v_w1 + (1 - beta2) * (d_w1 ** 2)
                m_hat_w1 = m_w1 / (1 - beta1 ** t)
                v_hat_w1 = v_w1 / (1 - beta2 ** t)
                self.w1 -= learning_rate * m_hat_w1 / (np.sqrt(v_hat_w1) + epsilon)
                
                m_b1 = beta1 * m_b1 + (1 - beta1) * d_b1
                v_b1 = beta2 * v_b1 + (1 - beta2) * (d_b1 ** 2)
                m_hat_b1 = m_b1 / (1 - beta1 ** t)
                v_hat_b1 = v_b1 / (1 - beta2 ** t)
                self.b1 -= learning_rate * m_hat_b1 / (np.sqrt(v_hat_b1) + epsilon)
            
            avg_loss = total_loss / max(1, num_batches)
            self.training_history.append(avg_loss)
            
            if epoch == 0 or (epoch + 1) % 20 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.6f}")
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"Initial Loss: {self.training_history[0]:.6f}")
        print(f"Final Loss:   {self.training_history[-1]:.6f}")
        improvement = ((self.training_history[0] - self.training_history[-1]) / self.training_history[0] * 100)
        print(f"Improvement:  {improvement:.1f}%")
        print(f"{'='*60}\n")

neural_net = ImprovedNeuralNetwork()

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
            # Calculate the tangential velocity at the contact point due to rotation
            # Distance from player center to contact point
            contact_dist = math.hypot(closest_x - player.x, closest_y - player.y)
            
            # If player is rotating, calculate tangential velocity
            if player.movement_in_progress and hasattr(player, 'rotation_speed'):
                # Get rotation direction (positive = counterclockwise)
                rotation_rate = player.rotation_speed if player.angle < player.target_angle else -player.rotation_speed
                
                # Tangential velocity perpendicular to radius
                # For clockwise rotation (negative rate), tangent points "forward" along the swing
                radius_angle = math.atan2(closest_y - player.y, closest_x - player.x)
                tangent_angle = radius_angle + math.pi / 2  # Perpendicular to radius
                
                # Tangential speed = angular_velocity * radius
                tangential_speed = abs(rotation_rate) * contact_dist * 1.1  # Scaled up for effect
                
                # Adjust tangent direction based on rotation direction
                if rotation_rate < 0:  # Clockwise
                    tangent_angle -= math.pi  # Flip direction
                
                # Combine stick pointing direction with rotational velocity
                speed = 6
                stick_component = 0.3  # 30% from stick direction
                rotation_component = 0.7  # 70% from rotation
                
                self.vx = (stick_component * stick_dx * speed + 
                        rotation_component * math.cos(tangent_angle) * tangential_speed)
                self.vy = (stick_component * stick_dy * speed + 
                        rotation_component * math.sin(tangent_angle) * tangential_speed)
            else:
                # No rotation, just use stick direction
                speed = 6
                self.vx = stick_dx * speed
                self.vy = stick_dy * speed
            
            # Push puck away from collision point
            to_puck_angle = math.atan2(py - closest_y, px - closest_x)
            self.x = closest_x + math.cos(to_puck_angle) * (self.radius + 3)
            self.y = closest_y + math.sin(to_puck_angle) * (self.radius + 3)
            self.was_hit = True
            player.has_hit_puck = True
            return True
        return False

min_y = HEIGHT / 2
max_y = HEIGHT - PLAYER_MIN_DIST_FROM_GOAL - BEHIND_GOAL_SPACE
players = [Player(WIDTH // 2, min_y, 'blue', min_y=min_y, max_y=max_y)]

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

def evaluate_action(action, puck_x, puck_y, max_steps=10000):
    """Evaluate an action and return a quality score."""
    test_player = copy.deepcopy(players[0])
    test_puck = Puck(x=puck_x, y=puck_y)
    test_player.do_action(action)
    
    hit_puck = False
    hit_after_rotation = False
    closest_to_goal = float('inf')
    puck_hit_goal_line = False
    
    goal_left = WIDTH / 2 - GOAL_WIDTH / 2
    goal_right = WIDTH / 2 + GOAL_WIDTH / 2
    
    for step in range(max_steps):
        test_player.update()
        test_puck.move()
        
        if test_puck.collide(test_player):
            hit_puck = True
            if test_player.has_rotated:
                hit_after_rotation = True
        
        # Track closest distance to goal
        if hit_puck:
            dist_to_goal = abs(test_puck.y - (BOTTOM_GOAL_Y + SCALE / 2))
            closest_to_goal = min(closest_to_goal, dist_to_goal)
        
        # Check if scored
        if hit_after_rotation and test_puck.y + test_puck.radius > BOTTOM_GOAL_Y + SCALE / 2:
            if goal_left < test_puck.x < goal_right:
                return 1.0  # Perfect score
            puck_hit_goal_line = True
        
        # Early termination if puck stopped and didn't score
        if hit_puck and abs(test_puck.vx) < 0.1 and abs(test_puck.vy) < 0.1:
            break
    
    # Partial credit scoring
    if not hit_puck:
        return 0.0
    elif not hit_after_rotation:
        return 0.2  # Hit puck but not after rotation
    elif puck_hit_goal_line:
        return 0.7  # Close to scoring
    elif closest_to_goal < 100:
        return 0.4 + 0.3 * (1 - closest_to_goal / 100)  # Got close
    else:
        return 0.3  # Hit after rotation but not close

def physics_guided_action(puck_x, puck_y):
    """Generate physics-guided actions based on puck position."""
    player_start_x = WIDTH / 2 - SCALE * 0.5
    player_start_y = HEIGHT / 2
    
    # Calculate where player needs to be
    dx = puck_x - player_start_x
    dy = puck_y - player_start_y
    
    max_move_distance = HEIGHT - PLAYER_MIN_DIST_FROM_GOAL - BEHIND_GOAL_SPACE - player_start_y
    
    # Distance calculation with some randomness
    target_distance = dy / max_move_distance
    linear_distance = np.clip(target_distance + random.uniform(-0.1, 0.1), 0.2, 0.9)
    
    # High speed
    linear_speed = random.uniform(0.85, 1.0)
    
    # Calculate angle to hit puck toward goal
    goal_center_x = WIDTH / 2
    goal_y = BOTTOM_GOAL_Y
    
    # Estimate final position
    final_player_y = player_start_y + linear_distance * max_move_distance
    final_player_x = player_start_x + (linear_distance * SCALE * 0.5)
    
    # Direction to puck
    to_puck_angle = math.atan2(puck_y - final_player_y, puck_x - final_player_x)
    
    # Add slight aim toward goal
    puck_to_goal_angle = math.atan2(goal_y - puck_y, goal_center_x - puck_x)
    
    # Blend the angles (favor hitting puck direction)
    desired_angle = 0.7 * to_puck_angle + 0.3 * puck_to_goal_angle
    
    # Calculate rotation needed from initial angle (-pi/2)
    rotation_needed = desired_angle - (-math.pi / 2)
    
    # Normalize
    while rotation_needed > math.pi:
        rotation_needed -= 2 * math.pi
    while rotation_needed < -math.pi:
        rotation_needed += 2 * math.pi
    
    rotation_angle = np.clip((rotation_needed / (2 * math.pi)) + random.uniform(-0.15, 0.15), -0.9, 0.9)
    rotation_speed = random.uniform(0.85, 1.0)
    
    return [linear_distance, linear_speed, rotation_angle, rotation_speed]

def collect_training_data(num_samples=500):
    """Collect training data with progressive difficulty."""
    print(f"\n{'='*60}")
    print(f"COLLECTING {num_samples} TRAINING SAMPLES")
    print(f"{'='*60}\n")
    
    samples_collected = 0
    attempts = 0
    max_attempts = num_samples * 100
    
    # Progressive difficulty zones
    easy_zone = (WIDTH * 0.3, WIDTH * 0.7, HEIGHT * 0.55, HEIGHT * 0.7)
    medium_zone = (WIDTH * 0.2, WIDTH * 0.8, HEIGHT * 0.5, HEIGHT * 0.75)
    hard_zone = (WIDTH * 0.1, WIDTH * 0.9, HEIGHT * 0.4, HEIGHT * 0.8)
    
    while samples_collected < num_samples and attempts < max_attempts:
        attempts += 1
        
        # Progressive difficulty
        if samples_collected < num_samples * 0.3:
            # Easy samples
            zone = easy_zone
        elif samples_collected < num_samples * 0.7:
            # Medium samples
            zone = medium_zone
        else:
            # Hard samples
            zone = hard_zone
        
        puck_x = random.uniform(zone[0], zone[1])
        puck_y = random.uniform(zone[2], zone[3])
        
        # Generate action (90% physics-guided, 10% random)
        if random.random() < 0.9:
            action = physics_guided_action(puck_x, puck_y)
        else:
            action = [
                random.uniform(0.3, 0.8),
                random.uniform(0.8, 1.0),
                random.uniform(-0.8, 0.8),
                random.uniform(0.8, 1.0)
            ]
        
        # Evaluate action
        score = evaluate_action(action, puck_x, puck_y)
        
        # Accept good actions (score > 0.5 for perfect, > 0.3 for partial)
        if score >= 0.5:
            neural_net.add_training_sample(puck_x, puck_y, action, score)
            samples_collected += 1
            
            if samples_collected % 50 == 0 or samples_collected <= 10:
                print(f"Sample {samples_collected:4d}/{num_samples} | Score: {score:.2f} | " +
                      f"Pos: ({puck_x:5.1f}, {puck_y:5.1f}) | Attempts: {attempts}")
    
    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE")
    print(f"Samples: {samples_collected}, Total attempts: {attempts}")
    print(f"Success rate: {samples_collected/attempts*100:.1f}%")
    print(f"{'='*60}\n")

def visualize_prediction(puck_x, puck_y, show_ml=True):
    """Visualize ML prediction vs physics-guided action."""
    print(f"\n{'='*60}")
    print(f"VISUALIZING PREDICTIONS")
    print(f"Puck Position: ({puck_x:.1f}, {puck_y:.1f})")
    print(f"{'='*60}\n")
    
    if show_ml:
        # ML prediction
        ml_action = neural_net.predict(puck_x, puck_y)
        ml_score = evaluate_action(ml_action, puck_x, puck_y)
        print(f"ML Prediction:")
        print(f"  Action: [ld={ml_action[0]:.3f}, ls={ml_action[1]:.3f}, " +
              f"ra={ml_action[2]:.3f}, rs={ml_action[3]:.3f}]")
        print(f"  Score: {ml_score:.3f}")
        
        # Visualize ML action
        print(f"\nVisualizing ML action...")
        visualize_episode(ml_action, puck_x, puck_y, duration=300)
    
    # Physics-guided action
    physics_action = physics_guided_action(puck_x, puck_y)
    physics_score = evaluate_action(physics_action, puck_x, puck_y)
    print(f"\nPhysics-Guided Action:")
    print(f"  Action: [ld={physics_action[0]:.3f}, ls={physics_action[1]:.3f}, " +
          f"ra={physics_action[2]:.3f}, rs={physics_action[3]:.3f}]")
    print(f"  Score: {physics_score:.3f}")
    
    print(f"\nVisualizing physics-guided action...")
    visualize_episode(physics_action, puck_x, puck_y, duration=300)
    
    print(f"\n{'='*60}")

def visualize_episode(action, puck_x, puck_y, duration=200):
    """Visualize a single episode."""
    players[0].do_action(action)
    puck = Puck(x=puck_x, y=puck_y)
    
    scored = False
    hit_puck = False
    
    for frame in range(duration):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        players[0].update()
        puck.move()
        if puck.collide(players[0]):
            hit_puck = True

        # Check for goal
        goal_left = WIDTH / 2 - GOAL_WIDTH / 2
        goal_right = WIDTH / 2 + GOAL_WIDTH / 2
        if puck.y + puck.radius > BOTTOM_GOAL_Y + SCALE / 2:
            if goal_left < puck.x < goal_right:
                scored = True

        screen.fill(WHITE)
        draw_field(screen)
        players[0].draw(screen)
        puck.draw(screen)
        
        # Draw status
        font = pygame.font.Font(None, 24)
        status = "GOAL!" if scored else ("HIT!" if hit_puck else "")
        if status:
            text = font.render(status, True, GREEN if scored else BLUE)
            screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(60)
        
        if scored:
            pygame.time.wait(1000)
            break

# Main execution
if __name__ == "__main__":
    # Try to load existing model
    model_loaded = neural_net.load(args.model_file)
    
    if args.train:
        print(f"\n{'='*60}")
        if model_loaded:
            print(f"TRAINING MODE (adding to existing model)")
        else:
            print(f"TRAINING MODE (creating new model)")
        print(f"{'='*60}")
        
        # Collect training data
        collect_training_data(args.train_samples)
        
        # Train the network
        neural_net.train(epochs=200, learning_rate=0.001, batch_size=32)
        
        # Save model
        neural_net.save(args.model_file)
    
    # Test prediction
    if args.puck_x is not None and args.puck_y is not None:
        test_x, test_y = args.puck_x, args.puck_y
    else:
        # Random test position
        test_x = random.uniform(WIDTH * 0.3, WIDTH * 0.7)
        test_y = random.uniform(HEIGHT * 0.5, HEIGHT * 0.7)
    
    print(f"\n{'='*60}")
    print(f"TESTING MODEL")
    print(f"{'='*60}")
    
    if args.visualize or args.train:
        visualize_prediction(test_x, test_y, show_ml=model_loaded)
    else:
        # Just show prediction
        if model_loaded:
            ml_action = neural_net.predict(test_x, test_y)
            ml_score = evaluate_action(ml_action, test_x, test_y)
            print(f"\nTest Position: ({test_x:.1f}, {test_y:.1f})")
            print(f"ML Prediction: {ml_action}")
            print(f"Predicted Score: {ml_score:.3f}")
            print(f"\nUse --visualize to see the action in motion")
        else:
            print(f"No model found. Use --train to create one.")
    
    print(f"\n{'='*60}")
    print(f"PROGRAM COMPLETE")
    print(f"{'='*60}\n")