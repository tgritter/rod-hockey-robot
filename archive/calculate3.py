import pygame
import sys
import math
import random
import copy
import argparse

# Parse optional puck coordinates from the command line
parser = argparse.ArgumentParser(description="Run Bubble Hockey simulation with optional puck coordinates.")
parser.add_argument("--puck_x", type=float, help="Initial puck x-coordinate in pixels")
parser.add_argument("--puck_y", type=float, help="Initial puck y-coordinate in pixels")
args = parser.parse_args()

pygame.init()

# Define scale (pixels per inch)
SCALE = 25  # This gives a good size on screen

# Game dimensions based on real measurements (corrected)
GOAL_WIDTH = 3.75 * SCALE  # 3 3/4 inches
PLAYER_DIAMETER = 1 * SCALE  # 1 inch
STICK_LENGTH = 1.75 * SCALE  # 1 3/4 inches
PUCK_DIAMETER = 1 * SCALE  # 1 inch

# --- CORRECTED BOARD DIMENSIONS ---
TOTAL_PLAYING_LENGTH_INCHES = 33.25
TOTAL_PLAYING_WIDTH_INCHES = 18
BEHIND_GOAL_SPACE_INCHES = 5.5 # Space from end board to goal line

# Calculate full dimensions in pixels
WIDTH = TOTAL_PLAYING_WIDTH_INCHES * SCALE
HEIGHT = TOTAL_PLAYING_LENGTH_INCHES * SCALE

# Goal positions (relative to the playing surface)
# The goal line is 5.5 inches from the top and bottom edge of the playing surface.
TOP_GOAL_LINE_Y = BEHIND_GOAL_SPACE_INCHES * SCALE
BOTTOM_GOAL_LINE_Y = HEIGHT - (BEHIND_GOAL_SPACE_INCHES * SCALE)

screen = pygame.display.set_mode((int(WIDTH), int(HEIGHT)))
pygame.display.set_caption("Bubble Hockey - ML Training (Vertical with Behind-Goal Space)")
clock = pygame.time.Clock()

WHITE = (255, 255, 255)
BLUE = (66, 135, 245)
RED = (245, 66, 66)
BLACK = (0, 0, 0)
LIGHT_GRAY = (200, 200, 200)

class Player:
    def __init__(self, x, y, team, is_goalie=False, min_y=None, max_y=None):
        self.x = x
        self.y = y
        # Player starting X is now fixed for vertical rail
        self.start_x = x
        self.start_y = y # This will be the initial Y (center line)
        self.radius = PLAYER_DIAMETER / 2
        self.team = team
        self.angle = -math.pi / 2  # Pointing upward
        self.is_goalie = is_goalie

        # Define player's vertical movement bounds
        # Player starts at the center line (HEIGHT / 2) and moves down towards their goal line (BOTTOM_GOAL_LINE_Y)
        self.min_y = min_y if min_y is not None else (HEIGHT / 2) + self.radius # Center line
        
        # --- MODIFICATION START ---
        # Calculate the new max_y based on a 9-inch movement range from min_y
        self.max_y = min_y + (9 * SCALE)
        # Ensure max_y does not exceed the actual goal line (with player radius buffer)
        self.max_y = min(self.max_y, BOTTOM_GOAL_LINE_Y - self.radius) 
        # --- MODIFICATION END ---

        # The horizontal movement and range limit logic was based on old assumptions.
        # For a fixed vertical rail, min_x and max_x should be the same as self.x
        self.min_x = self.x
        self.max_x = self.x
        
        # Player movement range (for AI actions) - This is the distance from min_y to max_y
        self.range_limit = self.max_y - self.min_y
        
        self.target_x = self.start_x
        self.target_y = y # Initial target_y is the starting y
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
        linear_distance_norm, linear_speed_norm, rotation_angle_norm, rotation_speed_norm = action
        self.has_rotated = False
        self.has_hit_puck = False
        
        # Reset to starting position for each action
        # Player starts at the center line
        self.x = WIDTH // 2 # Center X for the rail
        self.y = (HEIGHT / 2) + self.radius # Start at center line (adjusted for radius)
        self.angle = -math.pi / 2 # Pointing upward
        
        # Re-constrain player to the vertical rail after reset
        self.min_x = self.x
        self.max_x = self.x

        if not self.is_goalie:
            # linear_distance_norm (0-1) now controls movement from center line (0) to goal line (1)
            # A value of 0 means stay at min_y (center line)
            # A value of 1 means move to max_y (the new 9-inch limit)
            self.target_y = self.min_y + (linear_distance_norm * self.range_limit)
            self.target_y = max(self.min_y, min(self.max_y, self.target_y)) # Clamp for safety

            self.movement_speed = linear_speed_norm * self.max_linear_speed
            self.target_x = self.x # Stays on the rail

            # Rotation
            desired_rotation = rotation_angle_norm * math.pi # From -pi to pi (full circle relative to current angle)
            self.rotation_speed = rotation_speed_norm * self.max_rotation_speed
            self.target_angle = self.angle + desired_rotation # Target relative to current angle
            
            self.movement_in_progress = True
            self.position_reached = False

        return self.x, self.y, self.angle

    def update(self):
        if not self.is_goalie and self.movement_in_progress:
            move_speed = self.movement_speed # Directly use calculated speed
            rot_speed = self.rotation_speed   # Directly use calculated speed

            if not self.position_reached:
                # Update vertical position
                if abs(self.y - self.target_y) > move_speed:
                    direction = 1 if self.target_y > self.y else -1
                    self.y += direction * move_speed
                    # Ensure player stays on their fixed x-rail
                    self.x = self.start_x
                else:
                    self.y = self.target_y
                    self.x = self.start_x # Ensure player is on the rail when target reached
                    self.position_reached = True
            
            # Only rotate if position reached, and rotation is still needed
            if self.position_reached and abs(self.angle - self.target_angle) > rot_speed:
                direction = 1 if self.target_angle > self.angle else -1
                self.angle += direction * rot_speed
                # Ensure angle stays within a reasonable range (e.g., -pi to pi)
                self.angle = math.fmod(self.angle, 2 * math.pi)
                if self.angle > math.pi:
                    self.angle -= 2 * math.pi
                elif self.angle < -math.pi:
                    self.angle += 2 * math.pi
                self.has_rotated = True
            elif self.position_reached and abs(self.angle - self.target_angle) <= rot_speed:
                self.angle = self.target_angle
                self.has_rotated = True # Rotation is complete
                self.movement_in_progress = False # Both movement and rotation are complete
                self.position_reached = False # Reset for next action

        return self.x, self.y, self.angle

class Puck:
    def __init__(self, x=None, y=None):
        self.x = x if x is not None else WIDTH // 2
        self.y = y if y is not None else HEIGHT // 2
        self.radius = PUCK_DIAMETER / 2
        self.vx = 0
        self.vy = 0
        self.friction = 0.9 # Kept as 0.9 for this version
        self.was_hit = False

    def move(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= self.friction
        self.vy *= self.friction

        if abs(self.vx) < 0.1: self.vx = 0
        if abs(self.vy) < 0.1: self.vy = 0

        # Side wall collisions
        if self.x - self.radius < 0:
            self.vx *= -1
            self.x = self.radius
        elif self.x + self.radius > WIDTH:
            self.vx *= -1
            self.x = WIDTH - self.radius

        goal_left = WIDTH / 2 - GOAL_WIDTH / 2
        goal_right = WIDTH / 2 + GOAL_WIDTH / 2

        # Check for TOP Goal (Blue's goal)
        # Puck crosses the TOP_GOAL_LINE_Y AND is between the goal posts
        if self.y - self.radius <= TOP_GOAL_LINE_Y and (goal_left < self.x < goal_right):
            # The puck has entered the net area
            self.vx = 0 # Stop the puck
            self.vy = 0
            return 'goal_blue' # Blue scored (into the top goal)
        # If not a goal, check for collision with top wall if outside goal
        elif self.y - self.radius < TOP_GOAL_LINE_Y: # If puck hits top boundary (outside goals)
            self.vy *= -1
            self.y = TOP_GOAL_LINE_Y + self.radius

        # Check for BOTTOM Goal (Red's goal)
        # Puck crosses the BOTTOM_GOAL_LINE_Y AND is between the goal posts
        if self.y + self.radius >= BOTTOM_GOAL_LINE_Y and (goal_left < self.x < goal_right):
            # The puck has entered the net area
            self.vx = 0 # Stop the puck
            self.vy = 0
            return 'goal_red' # Red scored (into the bottom goal)
        # If not a goal, check for collision with bottom wall if outside goal
        elif self.y + self.radius > BOTTOM_GOAL_LINE_Y: # If puck hits bottom boundary (outside goals)
            self.vy *= -1
            self.y = BOTTOM_GOAL_LINE_Y - self.radius
        
        # --- END CORRECTED GOAL DETECTION LOGIC ---
        
        return None # No goal scored

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
        
        # Handle case where stick length is zero to avoid division by zero
        if line_len_squared == 0:
            return False

        t = max(0, min(1, ((px - start_x) * lx + (py - start_y) * ly) / line_len_squared))
        closest_x = start_x + t * lx
        closest_y = start_y + t * ly
        dist = math.hypot(px - closest_x, py - closest_y)

        # If puck is close to the stick and moving towards it
        if dist < self.radius + 3: # '3' is a small buffer for collision detection
            # Calculate angle from closest point on stick to puck
            norm_angle = math.atan2(py - closest_y, px - closest_x)
            
            # Determine how hard to hit based on player's stick angle relative to puck
            player_stick_angle = player.angle
            angle_diff = math.fmod(norm_angle - player_stick_angle + math.pi, 2 * math.pi) - math.pi
            
            if abs(angle_diff) < math.pi / 2:
                speed_multiplier = 1 - (abs(angle_diff) / (math.pi / 2)) # Stronger hit when more aligned
                speed = 15 * speed_multiplier # Max speed for puck hit (increased for better action in ML)
            else:
                speed = 3 # Weaker hit if hitting with back of stick or side

            self.vx = math.cos(norm_angle) * speed
            self.vy = math.sin(norm_angle) * speed
            
            # Push puck out to prevent sticking
            self.x = closest_x + math.cos(norm_angle) * (self.radius + 3)
            self.y = closest_y + math.sin(norm_angle) * (self.radius + 3)
            
            self.was_hit = True
            player.has_hit_puck = True
            return True
        return False

# Game setup - adjusted for new field dimensions
# Player is on the bottom half of the field.
# Their min_y is the center line. Their max_y is their goal line.
initial_player_y = (HEIGHT / 2) + PLAYER_DIAMETER/2 # Player starts at the center line
min_player_y_bound = (HEIGHT / 2) + PLAYER_DIAMETER/2 # Center line is the highest point

# --- MODIFICATION START ---
# Calculate the new max_player_y_bound based on a 9-inch movement range from min_player_y_bound
max_player_y_bound = min_player_y_bound + (9 * SCALE)
# Ensure max_player_y_bound does not exceed the actual goal line (with player radius buffer)
max_player_y_bound = min(max_player_y_bound, BOTTOM_GOAL_LINE_Y - PLAYER_DIAMETER/2)
# --- MODIFICATION END ---


players = [Player(WIDTH // 2, initial_player_y, 'blue', min_y=min_player_y_bound, max_y=max_player_y_bound)]
selected_blue = 0

def draw_field(screen):
    screen.fill(WHITE) # Fill background with white first

    # Draw rink boundaries (the entire playing surface)
    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, HEIGHT), 3)
    
    # Center line
    center_y = HEIGHT / 2
    pygame.draw.line(screen, BLACK, (0, center_y), (WIDTH, center_y), 2)

    # Center circle
    pygame.draw.circle(screen, BLACK, (WIDTH / 2, center_y), SCALE * 2, 2)

    # Goal lines
    pygame.draw.line(screen, RED, (0, TOP_GOAL_LINE_Y), (WIDTH, TOP_GOAL_LINE_Y), 2)
    pygame.draw.line(screen, RED, (0, BOTTOM_GOAL_LINE_Y), (WIDTH, BOTTOM_GOAL_LINE_Y), 2)

    # Top goal (blue's goal) - The area behind the goal line
    goal_x = WIDTH / 2 - GOAL_WIDTH / 2
    pygame.draw.rect(screen, LIGHT_GRAY, (goal_x, TOP_GOAL_LINE_Y - SCALE/2, GOAL_WIDTH, SCALE / 2)) # Draw a lighter background for the goal (slightly behind the line)
    pygame.draw.rect(screen, BLACK, (goal_x, TOP_GOAL_LINE_Y - SCALE/2, GOAL_WIDTH, SCALE / 2), 2) # And its outline
    
    # Bottom goal (red's goal) - The area behind the goal line
    pygame.draw.rect(screen, LIGHT_GRAY, (goal_x, BOTTOM_GOAL_LINE_Y, GOAL_WIDTH, SCALE / 2)) # Draw a lighter background for the goal
    pygame.draw.rect(screen, BLACK, (goal_x, BOTTOM_GOAL_LINE_Y, GOAL_WIDTH, SCALE / 2), 2)

    # Goal posts and nets
    for i in range(1, 5):
        x = goal_x + i * (GOAL_WIDTH / 5)
        # Top goal net
        pygame.draw.line(screen, BLACK, (x, TOP_GOAL_LINE_Y - SCALE / 2), (x, TOP_GOAL_LINE_Y), 1)
        # Bottom goal net
        pygame.draw.line(screen, BLACK, (x, BOTTOM_GOAL_LINE_Y), (x, BOTTOM_GOAL_LINE_Y + SCALE / 2), 1)


# Scores are not displayed in this ML training setup, but useful to keep in mind
score_blue = 0
score_red = 0
# font = pygame.font.Font(None, int(SCALE * 1.5)) # Not needed for ML viz

def reset_game_state(player_obj, puck_obj, initial_puck_x, initial_puck_y):
    """Resets player and puck to initial conditions for a new episode."""
    # Reset player to its initial state for the blue team (bottom half, center rail)
    player_obj.x = WIDTH // 2
    player_obj.y = (HEIGHT / 2) + PLAYER_DIAMETER/2 # Player starts at the center line
    player_obj.angle = -math.pi / 2
    player_obj.movement_in_progress = False
    player_obj.position_reached = False
    player_obj.has_hit_puck = False
    player_obj.has_rotated = False
    
    # Reset puck position and velocity
    puck_obj.x = initial_puck_x
    puck_obj.y = initial_puck_y
    puck_obj.vx = 0
    puck_obj.vy = 0
    puck_obj.was_hit = False

def visualize_successful_episode(best_action, best_puck_x, best_puck_y):
    episode_count = 0
    max_episodes = 1 # Only visualize one episode at a time as per original intent
    episode_timer = 0
    episode_duration = 120 # Frames per episode (2 seconds at 60 FPS)

    print(f"Visualizing {max_episodes} successful episode(s)...")

    # Initialize the actual game objects for visualization
    visual_player = Player(WIDTH // 2, initial_player_y, 'blue', min_y=min_player_y_bound, max_y=max_player_y_bound)
    visual_puck = Puck(x=best_puck_x, y=best_puck_y)

    visual_player.do_action(best_action) # Apply the best action to the visual player

    running_viz = True
    while running_viz:
        screen.fill(WHITE) # Clear screen for drawing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running_viz = False
                break # Exit visualization loop

        visual_player.update()
        
        # Only start moving/colliding puck once player action is complete (this was a simplification)
        # For more realism, puck always moves. Let's make it consistent.
        # Puck always moves, regardless of player action status
        visual_puck.move()
        goal_scored = visual_puck.move() # Check for goals during simulation
        visual_puck.collide(visual_player)
        
        # Check for goal after puck movement and collision
        if goal_scored == 'goal_red': # If blue (player) scored
            print("Goal scored during visualization!")
            running_viz = False # End visualization after goal.

        draw_field(screen)
        visual_player.draw(screen)
        visual_puck.draw(screen)

        pygame.display.flip()
        clock.tick(60)

        episode_timer += 1
        if episode_timer >= episode_duration:
            episode_count += 1
            if episode_count >= max_episodes:
                print("Visualization completed.")
                running_viz = False # End visualization after enough episodes
            else:
                # Reset for next visualization episode if max_episodes > 1
                reset_game_state(visual_player, visual_puck, best_puck_x, best_puck_y)
                visual_player.do_action(best_action) # Re-apply action for the next visualization episode
                episode_timer = 0

    pygame.quit() # Quit pygame after visualization
    sys.exit()

def simulate_action(action, puck_x, puck_y):
    """
    Simulates an action to determine if it results in a goal.
    This runs in a detached environment, so no drawing occurs.
    """
    # Create a deep copy of the player setup for this simulation
    sim_player = Player(WIDTH // 2, initial_player_y, 'blue', # Player starts at center line
                        min_y=min_player_y_bound, max_y=max_player_y_bound)
    sim_puck = Puck(x=puck_x, y=puck_y)
    
    sim_player.do_action(action) # Apply the action to the simulated player

    hit_after_rotation = False
    
    max_sim_steps = 200 # A reasonable number of steps for one action cycle + puck movement
    
    for step in range(max_sim_steps):
        sim_player.update() # Update player's position and angle
        
        # Puck always moves
        sim_puck.move() # <--- This is the correct place to call move once per step
        
        # Collide player and puck
        if sim_puck.collide(sim_player):
            if sim_player.has_rotated: # Check if hit happened after player's rotation is complete
                hit_after_rotation = True

        goal_status = sim_puck.move() # <--- REMOVE THIS REDUNDANT CALL

        # Check if puck scored in bottom goal (red's goal, scored by blue player)
        if goal_status == 'goal_red':
            if hit_after_rotation: # Only count as success if hit after rotation (as per original logic)
                return True
            else:
                return False # Goal, but not under the desired conditions

        # Break early if puck is too far away or stopped, to save computation
        if abs(sim_puck.vx) < 0.1 and abs(sim_puck.vy) < 0.1 and \
           math.hypot(sim_puck.x - WIDTH / 2, sim_puck.y - HEIGHT / 2) > HEIGHT / 4: # If stopped and far from center
            break
        # Also, if puck goes out of bounds on top/sides without scoring
        # NOTE: With corrected goal logic, puck should ideally stop at the goal line
        # so this check might be less critical for out of bounds on top/bottom
        if sim_puck.y - sim_puck.radius < 0 or sim_puck.x - sim_puck.radius < 0 or sim_puck.x + sim_puck.radius > WIDTH or sim_puck.y + sim_puck.radius > HEIGHT:
            break

    return False

print("Searching for successful action with rotation...")

best_action = None
best_puck_x = None
best_puck_y = None

# Try user-defined puck coordinates first, if provided
input_x = args.puck_x
input_y = args.puck_y

if input_x is not None and input_y is not None:
    print(f"Attempting to find action for user-provided puck coordinates: ({input_x}, {input_y})")
    attempts_for_user_puck = 500000 # Increased attempts for finding a good action
    for i in range(attempts_for_user_puck):
        action = [
            random.uniform(0.5, 1),       # linear_distance_norm (0=center, 1=goal line)
            random.uniform(0.5, 1),     # linear_speed_norm
            random.uniform(-1, 1),      # rotation_angle_norm (-1=CCW, 1=CW)
            random.uniform(0.5, 1)      # rotation_speed_norm
        ]
        # visualize_successful_episode([1,1,1,1], input_x, input_y)
        if simulate_action(action, input_x, input_y):
            best_action = [0.5,0.6,0.7,0.9]
            best_puck_x = input_x
            best_puck_y = input_y
            print(f"Found successful action with user puck coords after {i+1} attempts: {action}")
            visualize_successful_episode(best_action, best_puck_x, best_puck_y)
            break # Exit loop after finding the first successful action
    if best_action is None:
        print(f"Could not find a successful action for user-provided puck coordinates after {attempts_for_user_puck} attempts.")
        pygame.quit()
        sys.exit()

if best_action is None: # Only try random pucks if user didn't provide one or failed
    print("No user puck or failed to find action for user puck. Trying random pucks...")
    # Generate random puck coordinates within the blue player's half of the field
    random_puck_x = random.uniform(WIDTH / 4, WIDTH * 3 / 4)
    # Random Y between center line and bottom goal line, with some buffer from the goal line
    random_puck_y = random.uniform(HEIGHT / 2 + PUCK_DIAMETER, BOTTOM_GOAL_LINE_Y - PUCK_DIAMETER)
    
    print(f"Trying random puck coordinates: ({random_puck_x}, {random_puck_y})")
    attempts_for_random_puck = 100000
    for i in range(attempts_for_random_puck):
        action = [
            random.uniform(0, 1),
            random.uniform(0.5, 1),
            random.uniform(-1, 1),
            random.uniform(0.5, 1)
        ]
        if simulate_action(action, random_puck_x, random_puck_y):
            best_action = [0.5,0.6,0.7,0.9]
            best_puck_x = random_puck_x
            best_puck_y = random_puck_y
            print(f"Found successful action with random puck coords after {i+1} attempts: {action}")
            visualize_successful_episode(best_action, best_puck_x, best_puck_y)
            break

    if best_action is None:
        print(f"Could not find a successful action for random puck after {attempts_for_random_puck} attempts. Exiting.")
        pygame.quit()
        sys.exit()