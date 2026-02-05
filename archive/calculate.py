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

# Corrected dimensions based on 33.25 inches total length and 18 inches width
TOTAL_PLAYING_LENGTH_INCHES = 33.25
TOTAL_PLAYING_WIDTH_INCHES = 18

BEHIND_GOAL_SPACE_INCHES = 5.5 # Space from end board to goal line
PLAYER_MIN_DIST_FROM_GOAL_INCHES = 4 # This might need re-evaluating with new dims

# Calculate full dimensions in pixels
WIDTH = TOTAL_PLAYING_WIDTH_INCHES * SCALE
HEIGHT = TOTAL_PLAYING_LENGTH_INCHES * SCALE

# Goal positions (relative to the playing surface)
# The goal line is 5.5 inches from the top and bottom edge of the playing surface.
TOP_GOAL_LINE_Y = BEHIND_GOAL_SPACE_INCHES * SCALE
BOTTOM_GOAL_LINE_Y = HEIGHT - (BEHIND_GOAL_SPACE_INCHES * SCALE)

screen = pygame.display.set_mode((int(WIDTH), int(HEIGHT)))
pygame.display.set_caption("Bubble Hockey - Player Control (Vertical Movement)") # Updated caption
clock = pygame.time.Clock()

WHITE = (255, 255, 255)
BLUE = (66, 135, 245)
RED = (245, 66, 66)
BLACK = (0, 0, 0)
LIGHT_GRAY = (200, 200, 200)

class Player:
    def __init__(self, x, y, team, is_goalie=False, min_y=None, max_y=None, control_scheme='player'):
        self.x = x
        self.y = y
        self.start_x = x
        self.start_y = y
        self.radius = PLAYER_DIAMETER / 2
        self.team = team
        self.angle = -math.pi / 2  # Pointing upward (initially)
        self.is_goalie = is_goalie

        # Define player's movement bounds
        # These bounds will be relative to the new total height.
        # For the playable character (bottom player), restrict movement to their half.
        # Let's say the player can move from the bottom goal line up to the center line.
        
        # Original: self.min_y = min_y if min_y is not None else HEIGHT / 2 + self.radius
        # Original: self.max_y = max_y if max_y is not None else HEIGHT - BEHIND_GOAL_SPACE - self.radius
        
        # New player y bounds (assuming bottom half of the field for player control)
        # Player moves from BOTTOM_GOAL_LINE_Y up to the center of the field.
        # Adjust for player radius so they don't go off screen.
        self.min_y = min_y if min_y is not None else HEIGHT / 2 + self.radius # Center line
        self.max_y = max_y if max_y is not None else BOTTOM_GOAL_LINE_Y - self.radius # Up to goal line
        
        self.min_x = self.radius
        self.max_x = WIDTH - self.radius
        
        self.max_linear_speed = 3
        self.max_rotation_speed = 0.08
        self.has_hit_puck = False
        
        self.control_scheme = control_scheme # 'player' or 'ai'

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
    
    def update(self, keys):
        if self.control_scheme == 'player':
            # --- Player Input Handling ---
            # Vertical Movement (Up and Down)
            if keys[pygame.K_LEFT]: # Using LEFT for backwards (down)
                self.y += self.max_linear_speed
            if keys[pygame.K_RIGHT]: # Using RIGHT for forwards (up)
                self.y -= self.max_linear_speed
            
            # Rotation (clockwise/counter-clockwise)
            if keys[pygame.K_UP]:
                self.angle -= self.max_rotation_speed # Rotate counter-clockwise
            if keys[pygame.K_DOWN]:
                self.angle += self.max_rotation_speed # Rotate clockwise

            # Keep player within bounds (Y-axis bounds are crucial for vertical movement)
            self.x = max(self.min_x, min(self.max_x, self.x)) # Keep horizontal for now
            self.y = max(self.min_y, min(self.max_y, self.y))
            
            # Ensure angle stays within a reasonable range (e.g., -pi to pi)
            self.angle = math.fmod(self.angle, 2 * math.pi)
            if self.angle > math.pi:
                self.angle -= 2 * math.pi
            elif self.angle < -math.pi:
                self.angle += 2 * math.pi

            self.has_hit_puck = False # Reset hit status each frame


class Puck:
    def __init__(self, x=None, y=None):
        self.x = x if x is not None else WIDTH // 2
        self.y = y if y is not None else HEIGHT // 2
        self.radius = PUCK_DIAMETER / 2
        self.vx = 0
        self.vy = 0
        self.friction = 0.95
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

        # Top wall collision (with goal opening)
        if self.y - self.radius < TOP_GOAL_LINE_Y:
            if not (goal_left < self.x < goal_right):
                self.vy *= -1
                self.y = TOP_GOAL_LINE_Y + self.radius
            # If inside goal, allow to pass
            elif self.y - self.radius < 0: # Check for actual scoring past the end board
                self.vy = 0 # Stop puck
                self.vx = 0
                return 'goal_blue' # Indicate blue scored (top goal)

        # Bottom wall collision (with goal opening)
        if self.y + self.radius > BOTTOM_GOAL_LINE_Y: # Check against goal line
            if not (goal_left < self.x < goal_right):
                self.vy *= -1
                self.y = BOTTOM_GOAL_LINE_Y - self.radius
            # If inside goal, allow to pass
            elif self.y + self.radius > HEIGHT: # Check for actual scoring past the end board
                self.vy = 0 # Stop puck
                self.vx = 0
                return 'goal_red' # Indicate red scored (bottom goal)
        
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
            # This makes hits more effective when the stick is "facing" the puck
            player_stick_angle = player.angle
            angle_diff = math.fmod(norm_angle - player_stick_angle + math.pi, 2 * math.pi) - math.pi
            
            # If the puck is roughly in front of the stick, apply a stronger hit
            # You can adjust this value to control "power"
            if abs(angle_diff) < math.pi / 2:
                speed_multiplier = 1 - (abs(angle_diff) / (math.pi / 2)) # Stronger hit when more aligned
                speed = 15 * speed_multiplier # Max speed for puck hit
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

# Game setup
# Only one player for now, controlled by the user
# Player starts somewhere in their half, closer to the goal they defend.
player = Player(WIDTH // 2, BOTTOM_GOAL_LINE_Y - (5 * SCALE), 'blue', # 5 inches from their goal line
                min_y=HEIGHT/2 + PLAYER_DIAMETER/2, # Restrict to bottom half
                max_y=BOTTOM_GOAL_LINE_Y - PLAYER_DIAMETER/2) # Up to goal line (adjust for player radius)

# To constrain the player to a vertical "rail", set min_x and max_x to the same value
player.min_x = player.x
player.max_x = player.x

puck = Puck(x=args.puck_x, y=args.puck_y) # Use command line args or default

score_blue = 0
score_red = 0
font = pygame.font.Font(None, int(SCALE * 1.5)) # Larger font for scores

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

def reset_game():
    global puck, player # Need to use global for these
    puck = Puck(x=WIDTH // 2, y=HEIGHT // 2) # Reset puck to center
    player = Player(WIDTH // 2, BOTTOM_GOAL_LINE_Y - (5 * SCALE), 'blue',
                    min_y=HEIGHT/2 + PLAYER_DIAMETER/2,
                    max_y=BOTTOM_GOAL_LINE_Y - PLAYER_DIAMETER/2)
    # Re-constrain player to the vertical rail after reset
    player.min_x = player.x
    player.max_x = player.x
    
# --- Main Game Loop ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    # Update player based on keyboard input
    player.update(keys)

    # Move puck and check for goals
    goal_scored = puck.move()
    if goal_scored == 'goal_red':
        score_blue += 1
        print(f"Goal for Blue! Score: Blue {score_blue} - Red {score_red}")
        reset_game()
    elif goal_scored == 'goal_blue':
        score_red += 1
        print(f"Goal for Red! Score: Blue {score_blue} - Red {score_red}")
        reset_game()

    # Check for puck collision with player
    puck.collide(player)

    # Drawing
    draw_field(screen)
    player.draw(screen)
    puck.draw(screen)

    # Display score
    blue_score_text = font.render(f"Blue: {score_blue}", True, BLUE)
    red_score_text = font.render(f"Red: {score_red}", True, RED)
    screen.blit(blue_score_text, (WIDTH - blue_score_text.get_width() - 10, 10))
    screen.blit(red_score_text, (WIDTH - red_score_text.get_height() - 10, HEIGHT - red_score_text.get_height() - 10)) # Adjusted for bottom alignment

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()