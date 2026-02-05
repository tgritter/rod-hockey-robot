"""
Q-Table Action Suggester for Bubble Hockey
Loads a trained Q-table and suggests optimal actions based on puck position
Usage: python test_qtable.py --puck_x 265 --puck_y 580
"""

import argparse
import numpy as np
import pickle
import os
import math

# --- Constants (same as training) ---
SCALE = 25
WIDTH = int(18 * SCALE)
HEIGHT = int(33.25 * SCALE)
PUCK_DIAMETER = 1 * SCALE
PLAYER_DIAMETER = 1 * SCALE
STICK_LENGTH = 1.75 * SCALE
GOAL_WIDTH = 3.75 * SCALE
TOP_GOAL_LINE_Y = 5.5 * SCALE
BOTTOM_GOAL_LINE_Y = HEIGHT - (5.5 * SCALE)

# Discretization steps (same as training)
ACTION_STEPS = np.linspace(0.0, 1.0, 11)
ACTION_SPACE = [(a, b, c, d) for a in ACTION_STEPS for b in ACTION_STEPS[5:] for c in ACTION_STEPS for d in ACTION_STEPS[5:]]

# State discretization for puck position
PUCK_X_BINS = 10
PUCK_Y_BINS = 10
PUCK_X_MIN = 200
PUCK_X_MAX = 350
PUCK_Y_MIN = 500
PUCK_Y_MAX = 650

def discretize_puck_position(puck_x, puck_y):
    """Convert puck position to discrete state"""
    # Clamp to bounds
    puck_x = max(PUCK_X_MIN, min(PUCK_X_MAX, puck_x))
    puck_y = max(PUCK_Y_MIN, min(PUCK_Y_MAX, puck_y))
    
    # Discretize
    x_bin = int((puck_x - PUCK_X_MIN) / (PUCK_X_MAX - PUCK_X_MIN) * (PUCK_X_BINS - 1))
    y_bin = int((puck_y - PUCK_Y_MIN) / (PUCK_Y_MAX - PUCK_Y_MIN) * (PUCK_Y_BINS - 1))
    
    # Convert to single state index
    state = y_bin * PUCK_X_BINS + x_bin
    return state

def load_qtable(filename='qtable.pkl'):
    """Load Q-table from file"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"Warning: {filename} not found. Using random Q-table.")
        # Return a random Q-table for demonstration
        num_states = PUCK_X_BINS * PUCK_Y_BINS
        num_actions = len(ACTION_SPACE)
        return np.random.rand(num_states, num_actions)

def suggest_action(puck_x, puck_y, qtable):
    """Suggest best action for given puck position"""
    state = discretize_puck_position(puck_x, puck_y)
    
    # Handle case where state is out of bounds
    if state >= qtable.shape[0]:
        state = 0
    
    # Get best action index
    best_action_idx = np.argmax(qtable[state])
    best_action = ACTION_SPACE[best_action_idx]
    q_value = qtable[state][best_action_idx]
    
    return best_action, best_action_idx, q_value, state

def interpret_action(action):
    """Convert action tuple to human-readable description"""
    dist, dist_speed, angle_norm, angle_speed = action
    
    # Convert to actual values
    range_limit = 9 * SCALE
    min_y = (HEIGHT / 2) + PLAYER_DIAMETER / 2
    max_y = min(min_y + range_limit, BOTTOM_GOAL_LINE_Y - PLAYER_DIAMETER / 2)
    target_y = min_y + (dist * (max_y - min_y))
    
    angle_change = angle_norm * math.pi
    final_angle = -math.pi / 2 + angle_change  # Starting from -π/2
    
    # Normalize angle
    while final_angle > math.pi:
        final_angle -= 2 * math.pi
    while final_angle < -math.pi:
        final_angle += 2 * math.pi
    
    angle_degrees = math.degrees(final_angle)
    
    return {
        'player_y_position': target_y,
        'player_angle_radians': final_angle,
        'player_angle_degrees': angle_degrees,
        'distance_factor': dist,
        'angle_factor': angle_norm
    }

def main():
    parser = argparse.ArgumentParser(description='Get Q-table action suggestion for puck position')
    parser.add_argument('--puck_x', type=float, required=True, help='Puck X position')
    parser.add_argument('--puck_y', type=float, required=True, help='Puck Y position')
    parser.add_argument('--qtable', type=str, default='qtable.pkl', help='Q-table file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed action interpretation')
    
    args = parser.parse_args()
    
    # Load Q-table
    print(f"Loading Q-table from {args.qtable}...")
    qtable = load_qtable(args.qtable)
    print(f"Q-table shape: {qtable.shape}")
    
    # Get suggestion
    action, action_idx, q_value, state = suggest_action(args.puck_x, args.puck_y, qtable)
    
    print(f"\nPuck position: ({args.puck_x}, {args.puck_y})")
    print(f"Discrete state: {state}")
    print(f"Best action index: {action_idx}")
    print(f"Q-value: {q_value:.4f}")
    print(f"Action tuple: {action}")
    
    if args.verbose:
        interpretation = interpret_action(action)
        print(f"\nAction interpretation:")
        print(f"  Player Y position: {interpretation['player_y_position']:.1f}")
        print(f"  Player angle: {interpretation['player_angle_degrees']:.1f}°")
        print(f"  Distance factor: {interpretation['distance_factor']:.3f}")
        print(f"  Angle factor: {interpretation['angle_factor']:.3f}")
        
        # Show top 5 actions for this state
        print(f"\nTop 5 actions for this state:")
        top_actions = np.argsort(qtable[state])[-5:][::-1]
        for i, action_idx in enumerate(top_actions):
            action_tuple = ACTION_SPACE[action_idx]
            q_val = qtable[state][action_idx]
            print(f"  {i+1}. Action {action_idx}: {action_tuple} (Q={q_val:.4f})")

if __name__ == "__main__":
    main()