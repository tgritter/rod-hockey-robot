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

# Progressive difficulty zones
easy_zone = (WIDTH * 0.3, WIDTH * 0.7, HEIGHT * 0.55, HEIGHT * 0.7)
medium_zone = (WIDTH * 0.2, WIDTH * 0.8, HEIGHT * 0.5, HEIGHT * 0.75)
hard_zone = (WIDTH * 0.1, WIDTH * 0.9, HEIGHT * 0.4, HEIGHT * 0.8)

screen = pygame.display.set_mode((int(WIDTH), int(HEIGHT)))
pygame.display.set_caption("Bubble Hockey - ML Training (Vertical with Behind-Goal Space)")
clock = pygame.time.Clock()

WHITE = (255, 255, 255)
BLUE = (66, 135, 245)
RED = (245, 66, 66)
BLACK = (0, 0, 0)
LIGHT_GRAY = (200, 200, 200)
GREEN = (66, 245, 135)
YELLOW = (245, 245, 66)
ORANGE = (245, 135, 66)

class Player:
    def __init__(self, x, y, team, is_goalie=False, min_y=None, max_y=None):
        self.x = x
        self.y = y
        self.start_x = x - SCALE * 0.5  # Start 1/2 inch left of provided x position
        self.start_y = y
        self.radius = PLAYER_DIAMETER / 2
        self.team = team
        self.angle = -math.pi / 2  # Pointing upward
        self.is_goalie = is_goalie

        self.min_y = min_y if min_y is not None else BEHIND_GOAL_SPACE
        self.max_y = max_y if max_y is not None else HEIGHT - BEHIND_GOAL_SPACE
        self.range_limit = HALF_FIELD_LENGTH - BEHIND_GOAL_SPACE - PLAYER_MIN_DIST_FROM_GOAL  # Player movement range

        self.target_x = self.start_x  # Initialize target_x to the offset starting position
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
        
        # Reset to starting position
        self.x = self.start_x  # Reset to left of center
        self.y = HEIGHT // 2  # Start at the center line
        self.angle = -math.pi / 2

        if not self.is_goalie:
            if linear_speed > 0:
                desired_distance = linear_distance * self.range_limit
                self.movement_speed = linear_speed * self.max_linear_speed
                target_y = self.start_y + desired_distance
                min_allowed = max(self.start_y - self.range_limit, self.min_y + self.radius)
                max_allowed = min(self.start_y + self.range_limit, self.max_y - self.radius)
                self.target_y = max(min_allowed, min(max_allowed, target_y))
                
                # Calculate horizontal position based on vertical position
                # At min_y -> start_x (1/2 inch left of center)
                # At max_y -> center (WIDTH/2)
                progress = (self.target_y - self.min_y) / (self.max_y - self.min_y)
                self.target_x = self.start_x + progress * SCALE * 0.5  # Move 1/2 inch to the right as we go down

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
                # Update vertical position
                if abs(self.y - self.target_y) > move_speed:
                    direction = 1 if self.target_y > self.y else -1
                    self.y += direction * move_speed
                    
                    # Update horizontal position based on vertical progress
                    progress = (self.y - self.min_y) / (self.max_y - self.min_y)
                    progress = max(0, min(1, progress))  # Clamp between 0 and 1
                    self.x = self.start_x + progress * SCALE * 0.5
                else:
                    self.y = self.target_y
                    # Final horizontal position based on final vertical position
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

        # Side wall collisions
        if self.x - self.radius < 0 or self.x + self.radius > WIDTH:
            self.vx *= -1
            self.x = max(self.radius, min(WIDTH - self.radius, self.x))

        goal_left = WIDTH / 2 - GOAL_WIDTH / 2
        goal_right = WIDTH / 2 + GOAL_WIDTH / 2

        # Top wall collision (with goal opening)
        if self.y - self.radius < 0:
            self.vy *= -1
            self.y = self.radius
        elif self.y - self.radius < TOP_GOAL_Y:
            if not (goal_left < self.x < goal_right):
                self.vy *= -1
                self.y = TOP_GOAL_Y + self.radius

        # Bottom wall collision (with goal opening)
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

# Game setup - adjusted for new field dimensions
min_y = HEIGHT / 2  # Center line
max_y = HEIGHT - PLAYER_MIN_DIST_FROM_GOAL - BEHIND_GOAL_SPACE  # 4 inches from goal line
players = [Player(WIDTH // 2, min_y, 'blue', min_y=min_y, max_y=max_y)]
selected_blue = 0

def draw_field(screen):
    # Behind-goal areas are white (same as main playing surface)
    
    # Draw difficulty zones (semi-transparent)
    # Easy zone (green)
    ez = easy_zone
    s = pygame.Surface((ez[1] - ez[0], ez[3] - ez[2]), pygame.SRCALPHA)
    s.fill((66, 245, 135, 50))
    screen.blit(s, (ez[0], ez[2]))
    
    # Medium zone (yellow)
    mz = medium_zone
    s = pygame.Surface((mz[1] - mz[0], mz[3] - mz[2]), pygame.SRCALPHA)
    s.fill((245, 245, 66, 50))
    screen.blit(s, (mz[0], mz[2]))
    
    # Hard zone (orange)
    hz = hard_zone
    s = pygame.Surface((hz[1] - hz[0], hz[3] - hz[2]), pygame.SRCALPHA)
    s.fill((245, 135, 66, 50))
    screen.blit(s, (hz[0], hz[2]))
    
    # Draw rink boundaries
    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, HEIGHT), 3)
    
    # Center line
    center_y = HEIGHT / 2
    pygame.draw.line(screen, BLACK, (0, center_y), (WIDTH, center_y), 2)

    # Center circle
    pygame.draw.circle(screen, BLACK, (WIDTH / 2, center_y), SCALE * 2, 2)

    # Goal lines
    pygame.draw.line(screen, RED, (0, TOP_GOAL_Y), (WIDTH, TOP_GOAL_Y), 2)
    pygame.draw.line(screen, RED, (0, BOTTOM_GOAL_Y + SCALE / 2), (WIDTH, BOTTOM_GOAL_Y + SCALE / 2), 2)

    # Top goal
    goal_x = WIDTH / 2 - GOAL_WIDTH / 2
    pygame.draw.rect(screen, BLACK, (goal_x, TOP_GOAL_Y, GOAL_WIDTH, SCALE / 2), 2)
    
    # Bottom goal
    pygame.draw.rect(screen, BLACK, (goal_x, BOTTOM_GOAL_Y, GOAL_WIDTH, SCALE / 2), 2)

    # Goal posts and nets
    for i in range(1, 5):
        x = goal_x + i * (GOAL_WIDTH / 5)
        # Top goal net
        pygame.draw.line(screen, BLACK, (x, TOP_GOAL_Y), (x, TOP_GOAL_Y + SCALE / 2), 1)
        # Bottom goal net
        pygame.draw.line(screen, BLACK, (x, BOTTOM_GOAL_Y), (x, BOTTOM_GOAL_Y + SCALE / 2), 1)

def is_in_zone(x, y, zone):
    """Check if puck is in a specific zone."""
    return zone[0] <= x <= zone[1] and zone[2] <= y <= zone[3]

def get_puck_zone(x, y):
    """Return the zone difficulty of the puck position."""
    if is_in_zone(x, y, easy_zone):
        return "easy"
    elif is_in_zone(x, y, medium_zone):
        return "medium"
    elif is_in_zone(x, y, hard_zone):
        return "hard"
    else:
        return "outside"

def visualize_two_shot_episode(setup_action, goal_action, initial_puck_x, initial_puck_y, intermediate_puck_x, intermediate_puck_y):
    """Visualize a two-shot sequence: setup shot to move puck to easy zone, then goal shot."""
    episode_count = 0
    max_episodes = 1
    episode_timer = 0
    episode_duration = 240  # Longer duration for two shots

    print(f"Visualizing {max_episodes} two-shot episode(s)...")
    print(f"Shot 1: Moving puck from ({initial_puck_x:.1f}, {initial_puck_y:.1f}) to easy zone")
    print(f"Shot 2: Scoring from easy zone at ({intermediate_puck_x:.1f}, {intermediate_puck_y:.1f})")

    shot_phase = 1  # 1 = setup shot, 2 = goal shot
    phase_timer = 0
    phase_duration = 120

    players[0].do_action(setup_action)
    puck = Puck(x=initial_puck_x, y=initial_puck_y)

    while True:
        screen.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        for player in players:
            player.update()

        puck.move()
        for player in players:
            puck.collide(player)

        draw_field(screen)

        for player in players:
            player.draw(screen)

        puck.draw(screen)

        # Draw phase indicator
        font = pygame.font.Font(None, 36)
        phase_text = f"Shot {shot_phase}: {'Setup' if shot_phase == 1 else 'Goal'}"
        text_surface = font.render(phase_text, True, BLACK)
        screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        clock.tick(60)

        phase_timer += 1
        episode_timer += 1

        # Switch to goal shot after setup phase
        if shot_phase == 1 and phase_timer >= phase_duration:
            shot_phase = 2
            phase_timer = 0
            # Reset player and puck for goal shot
            players[0] = Player(WIDTH // 2, min_y, 'blue', min_y=min_y, max_y=max_y)
            puck = Puck(x=intermediate_puck_x, y=intermediate_puck_y)
            players[0].do_action(goal_action)

        if episode_timer >= episode_duration:
            episode_count += 1
            if episode_count >= max_episodes:
                print("All episodes completed.")
                pygame.quit()
                return

            # Reset for next episode
            shot_phase = 1
            phase_timer = 0
            players[0] = Player(WIDTH // 2, min_y, 'blue', min_y=min_y, max_y=max_y)
            puck = Puck(x=initial_puck_x, y=initial_puck_y)
            players[0].do_action(setup_action)
            episode_timer = 0

def visualize_successful_episode(best_action, best_puck_x, best_puck_y):
    episode_count = 0
    max_episodes = 1
    episode_timer = 0
    episode_duration = 120

    print(f"Visualizing {max_episodes} successful episode(s)...")

    players[0].do_action(best_action)
    puck = Puck(x=best_puck_x, y=best_puck_y)

    while True:
        screen.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        for player in players:
            player.update()

        puck.move()
        for player in players:
            puck.collide(player)

        draw_field(screen)

        for player in players:
            player.draw(screen)

        puck.draw(screen)

        pygame.display.flip()
        clock.tick(60)

        episode_timer += 1
        if episode_timer >= episode_duration:
            episode_count += 1
            if episode_count >= max_episodes:
                print("All episodes completed.")
                pygame.quit()
                return

            # Reset player and puck for next episode
            players[0] = Player(WIDTH // 2, min_y, 'blue', min_y=min_y, max_y=max_y)
            puck = Puck(x=best_puck_x, y=best_puck_y)
            players[selected_blue].do_action(best_action)
            episode_timer = 0

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
        # Check if puck scored in bottom goal
        if hit_after_rotation and test_puck.y + test_puck.radius > BOTTOM_GOAL_Y + SCALE / 2:
            if WIDTH / 2 - GOAL_WIDTH / 2 < test_puck.x < WIDTH / 2 + GOAL_WIDTH / 2:
                return True
    return False

def simulate_action_get_final_pos(action, puck_x, puck_y, max_steps=10000):
    """Simulate an action and return the final puck position."""
    test_player = copy.deepcopy(players[0])
    test_puck = Puck(x=puck_x, y=puck_y)
    test_player.do_action(action)

    for _ in range(max_steps):
        test_player.update()
        test_puck.move()
        test_puck.collide(test_player)
        
        # Stop simulation when puck stops moving
        if abs(test_puck.vx) < 0.1 and abs(test_puck.vy) < 0.1:
            break

    return test_puck.x, test_puck.y

def find_setup_shot_to_easy_zone(puck_x, puck_y, attempts=50000):
    """Find an action that moves the puck from current position to easy zone."""
    print(f"Searching for setup shot to move puck to easy zone...")
    
    for i in range(attempts):
        action = [
            random.uniform(0, 1),
            random.uniform(0.5, 1),
            random.uniform(-1, 1),
            random.uniform(0.5, 1)
        ]
        
        final_x, final_y = simulate_action_get_final_pos(action, puck_x, puck_y)
        
        if is_in_zone(final_x, final_y, easy_zone):
            print(f"Found setup shot after {i+1} attempts!")
            print(f"Final puck position: ({final_x:.1f}, {final_y:.1f})")
            return action, final_x, final_y
    
    print("Could not find setup shot to easy zone.")
    return None, None, None

def find_goal_shot_from_easy_zone(puck_x, puck_y, attempts=50000):
    """Find an action that scores a goal from the easy zone."""
    print(f"Searching for goal shot from easy zone position ({puck_x:.1f}, {puck_y:.1f})...")
    
    for i in range(attempts):
        action = [
            random.uniform(0, 1),
            random.uniform(0.5, 1),
            random.uniform(-1, 1),
            random.uniform(0.5, 1)
        ]
        
        if simulate_action(action, puck_x, puck_y):
            print(f"Found goal shot after {i+1} attempts!")
            return action
    
    print("Could not find goal shot from easy zone.")
    return None

print("Searching for successful action with rotation...")

best_action = None
best_puck_x = None
best_puck_y = None

# Try user-defined puck coordinates first, if provided
input_x = args.puck_x
input_y = args.puck_y

if input_x is not None and input_y is not None:
    print(f"Trying user-provided puck coordinates: ({input_x}, {input_y})")
    puck_zone = get_puck_zone(input_x, input_y)
    print(f"Puck is in {puck_zone} zone")
    
    if puck_zone == "easy":
        # Direct shot to goal
        print("Puck in easy zone - attempting direct goal shot...")
        for _ in range(100000):
            action = [random.uniform(0, 1), random.uniform(0.5, 1), random.uniform(-1, 1), random.uniform(0.5, 1)]
            if simulate_action(action, input_x, input_y):
                best_action = action
                best_puck_x = input_x
                best_puck_y = input_y
                print(f"Found successful action: {action}")
                visualize_successful_episode(best_action, best_puck_x, best_puck_y)
                break
    
    elif puck_zone in ["medium", "hard"]:
        # Two-shot strategy: setup shot to easy zone, then goal shot
        print(f"Puck in {puck_zone} zone - attempting two-shot strategy...")
        
        # Find setup shot to move puck to easy zone
        setup_action, intermediate_x, intermediate_y = find_setup_shot_to_easy_zone(input_x, input_y)
        
        if setup_action is not None:
            # Find goal shot from easy zone position
            goal_action = find_goal_shot_from_easy_zone(intermediate_x, intermediate_y)
            
            if goal_action is not None:
                print("Two-shot sequence found successfully!")
                print(f"Setup action: {setup_action}")
                print(f"Goal action: {goal_action}")
                visualize_two_shot_episode(setup_action, goal_action, input_x, input_y, intermediate_x, intermediate_y)
            else:
                print("Failed to find goal shot from easy zone position.")
        else:
            print("Failed to find setup shot to easy zone.")
    
    else:
        print("Puck is outside defined zones - attempting direct goal shot...")
        for _ in range(100000):
            action = [random.uniform(0, 1), random.uniform(0.5, 1), random.uniform(-1, 1), random.uniform(0.5, 1)]
            if simulate_action(action, input_x, input_y):
                best_action = action
                best_puck_x = input_x
                best_puck_y = input_y
                print(f"Found successful action: {action}")
                visualize_successful_episode(best_action, best_puck_x, best_puck_y)
                break

def find_best_action(scaled_x, scaled_y, attempts=100000):
    """
    Attempts to find the best action for the given puck coordinates and renders it once if successful.

    Parameters:
    - scaled_x (float): The x-coordinate of the puck.
    - scaled_y (float): The y-coordinate of the puck.
    - attempts (int): Number of random actions to try.

    Returns:
    - best_action (list): The best action found, or None if not found.
    """
    puck_zone = get_puck_zone(scaled_x, scaled_y)
    print(f"Puck is in {puck_zone} zone")
    
    if puck_zone == "easy":
        # Direct shot to goal
        print("Attempting direct goal shot from easy zone...")
        best_action = None
        for _ in range(attempts):
            action = [
                random.uniform(0, 1),
                random.uniform(0.5, 1),
                random.uniform(-1, 1),
                random.uniform(0.5, 1)
            ]
            if simulate_action(action, scaled_x, scaled_y):
                best_action = action
                break
        
        if best_action:
            visualize_successful_episode(best_action, scaled_x, scaled_y)
        return best_action
    
    elif puck_zone in ["medium", "hard"]:
        # Two-shot strategy
        print(f"Attempting two-shot strategy from {puck_zone} zone...")
        
        setup_action, intermediate_x, intermediate_y = find_setup_shot_to_easy_zone(scaled_x, scaled_y, attempts)
        
        if setup_action is not None:
            goal_action = find_goal_shot_from_easy_zone(intermediate_x, intermediate_y, attempts)
            
            if goal_action is not None:
                visualize_two_shot_episode(setup_action, goal_action, scaled_x, scaled_y, intermediate_x, intermediate_y)
                return [setup_action, goal_action]  # Return both actions
        
        return None
    
    else:
        # Outside zones - try direct shot
        print("Puck outside zones - attempting direct shot...")
        best_action = None
        for _ in range(attempts):
            action = [
                random.uniform(0, 1),
                random.uniform(0.5, 1),
                random.uniform(-1, 1),
                random.uniform(0.5, 1)
            ]
            if simulate_action(action, scaled_x, scaled_y):
                best_action = action
                break
        
        if best_action:
            visualize_successful_episode(best_action, scaled_x, scaled_y)
        return best_action