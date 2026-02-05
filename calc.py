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
pygame.display.set_caption("Bubble Hockey - Three Player Simulation (with curved 3rd player)")
clock = pygame.time.Clock()

WHITE = (255, 255, 255)
BLUE = (66, 135, 245)
RED = (245, 66, 66)
BLACK = (0, 0, 0)

# Target zone for Player 1
TARGET_X_MIN = 160
TARGET_X_MAX = 270
TARGET_Y_MIN = 530
TARGET_Y_MAX = 600

# Target zone for Player 2
TARGET_X_MIN_P2 = 10
TARGET_X_MAX_P2 = 5.25 * SCALE  # 131.25 pixels
TARGET_Y_MIN_P2 = 530  # Same as Player 1
TARGET_Y_MAX_P2 = 775

class Player:
    def __init__(self, x, y, team, is_goalie=False, min_y=None, max_y=None):
        self.x = x
        self.y = y
        self.start_x = x
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

    # default mapping: linear horizontal shift from start_x based on progress
    def get_x_for_y(self, y):
        # progress in [0,1] across player's vertical movement range
        movement_range = max(1e-6, (self.max_y - self.min_y))
        progress = (y - self.min_y) / movement_range
        progress = max(0.0, min(1.0, progress))
        return self.start_x + progress * SCALE * 1.0

    def do_action(self, action):
        linear_distance, linear_speed, rotation_angle, rotation_speed = action
        self.has_rotated = False
        self.has_hit_puck = False
        self.x = self.start_x
        self.y = self.min_y
        self.angle = -math.pi / 2
        if not self.is_goalie:
            if linear_speed > 0:
                # Use actual player movement range (max_y - min_y).
                movement_range = self.max_y - self.min_y
                desired_distance = linear_distance * movement_range

                self.movement_speed = linear_speed * self.max_linear_speed
                target_y = self.min_y + desired_distance
                min_allowed = self.min_y + self.radius
                max_allowed = self.max_y - self.radius
                self.target_y = max(min_allowed, min(max_allowed, target_y))

                # compute target_x using the player's get_x_for_y mapping
                self.target_x = self.get_x_for_y(self.target_y)

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
                    # update x based on curve mapping
                    self.x = self.get_x_for_y(self.y)
                else:
                    self.y = self.target_y
                    self.x = self.get_x_for_y(self.y)
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

class CurvedPlayer(Player):
    """
    Player with a two-segment path:
      - segment A (top): y in [y_min_A, y_max_A] -> x around [x_min_A, x_max_A] (small wiggle)
      - segment B (behind net): y in [y_min_B, y_max_B] -> x moves horizontally from x_start_B -> x_end_B
    Uses piecewise linear interpolation between these ranges and smooths the transition.
    Coordinates provided by the user (in inches) are multiplied by SCALE to pixels.
    """

    def __init__(self, y_min_a, y_max_a, x_min_a, x_max_a,
                       y_min_b, y_max_b, x_min_b, x_max_b,
                       team='blue', start_x=None):
        # convert to pixel coordinates
        self.y_min_a = y_min_a * SCALE
        self.y_max_a = y_max_a * SCALE
        self.x_min_a = x_min_a * SCALE
        self.x_max_a = x_max_a * SCALE

        self.y_min_b = y_min_b * SCALE
        self.y_max_b = y_max_b * SCALE
        self.x_min_b = x_min_b * SCALE
        self.x_max_b = x_max_b * SCALE

        # overall min/max
        min_y = min(self.y_min_a, self.y_min_b)
        max_y = max(self.y_max_a, self.y_max_b)

        # choose a reasonable start_x if not provided
        if start_x is None:
            start_x = self.x_min_a

        super().__init__(start_x, min_y, team, is_goalie=False, min_y=min_y, max_y=max_y)
        # override start_x to first-segment center
        self.start_x = start_x
        # ensure player's displayed x,y start consistent
        self.x = self.get_x_for_y(self.y)
        self.y = min_y

    def get_x_for_y(self, y):
        # Segment A (top) remains the same
        if y <= self.y_max_a:
            denom = max(1e-6, (self.y_max_a - self.y_min_a))
            prog = (y - self.y_min_a) / denom
            prog = max(0.0, min(1.0, prog))
            return self.x_min_a + prog * (self.x_max_a - self.x_min_a)

        # Segment B (behind net)
        if y >= self.y_min_b:
            # Define two subsegments for B
            turn_threshold_y = self.y_min_b + 0.3 * (self.y_max_b - self.y_min_b)

            if y <= turn_threshold_y:
                # Horizontal move for the left turn (y roughly constant)
                prog = (y - self.y_min_b) / max(1e-6, turn_threshold_y - self.y_min_b)
                return self.x_max_b + prog * (self.x_min_b - self.x_max_b)
            else:
                # Vertical drop after the turn (x constant at x_min_b)
                return self.x_min_b

        # Transition zone between A and B
        denom = max(1e-6, (self.y_min_b - self.y_max_a))
        prog_between = (y - self.y_max_a) / denom
        prog_between = max(0.0, min(1.0, prog_between))
        return self.x_max_a + prog_between * (self.x_max_b - self.x_max_a)


class Puck:
    def __init__(self, x=None, y=None):
        self.x = x if x is not None else WIDTH // 2
        self.y = y if y is not None else HEIGHT // 2
        self.radius = PUCK_DIAMETER / 2
        self.vx = 0
        self.vy = 0
        self.friction = 0.925
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

        # keep top/bottom screen-edge bounces (so puck stays on screen)
        if self.y - self.radius < 0:
            self.vy *= -1
            self.y = self.radius
        if self.y + self.radius > HEIGHT:
            self.vy *= -1
            self.y = HEIGHT - self.radius

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

# Game setup - Two players + new curved 3rd player
min_y_p1 = 14.5 * SCALE
max_y_p1 = 23.5 * SCALE
player1_x = 8 * SCALE
player1 = Player(player1_x, min_y_p1, 'blue', min_y=min_y_p1, max_y=max_y_p1)

min_y_p2 = 15 * SCALE
max_y_p2 = 31 * SCALE
player2_x = 2 * SCALE
player2 = Player(player2_x, min_y_p2, 'red', min_y=min_y_p2, max_y=max_y_p2)

player3 = CurvedPlayer(
    y_min_a=21.5, y_max_a=28.0, x_min_a=16.0, x_max_a=16.5,
    y_min_b=28.0, y_max_b=30.0, x_min_b=5.72, x_max_b=16.5,
    team='red',
    start_x=16.0 * SCALE
)

# Player 4
min_y_p4 = 6 * SCALE
max_y_p4 = 14 * SCALE
player4_x = 5.5 * SCALE
player4 = Player(player4_x, min_y_p4, 'blue', min_y=min_y_p4, max_y=max_y_p4)

# Player 5
min_y_p5 = 1.5 * SCALE
max_y_p5 = 18 * SCALE
player5_x = 13 * SCALE
player5 = Player(player5_x, min_y_p5, 'blue', min_y=min_y_p5, max_y=max_y_p5)


players = [player1, player2, player3, player4, player5]

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

    pygame.draw.rect(screen, (255, 0, 0), (TARGET_X_MIN, TARGET_Y_MIN, TARGET_X_MAX - TARGET_X_MIN, TARGET_Y_MAX - TARGET_Y_MIN), 3)
    yellow_x = TARGET_X_MIN
    yellow_w = TARGET_X_MAX - TARGET_X_MIN
    yellow_start = TARGET_Y_MIN
    yellow_end = 420
    y_top = min(yellow_start, yellow_end)
    y_bottom = max(yellow_start, yellow_end)
    yellow_h = y_bottom - y_top
    if yellow_h > 0:
        pygame.draw.rect(screen, (255, 255, 0), (int(yellow_x), int(y_top), int(yellow_w), int(yellow_h)), 3)

    # Compute combined green box coordinates
    combined_y_top = min(TARGET_Y_MIN_P2, y_top)
    combined_y_bottom = max(TARGET_Y_MAX_P2, y_bottom)
    combined_height = combined_y_bottom - combined_y_top

    pygame.draw.rect(
        screen,
        (0, 255, 0),
        (TARGET_X_MIN_P2, combined_y_top, TARGET_X_MAX_P2 - TARGET_X_MIN_P2, combined_height),
        3
    )

    # --- Draw two blue boxes for Player 3 hitting region ---
    p3 = players[2]

    # Sample x-values along vertical range
    xs = []
    for sy in range(int(p3.min_y), int(p3.max_y) + 1, max(1, int((p3.max_y - p3.min_y) / 40))):
        xs.append(p3.get_x_for_y(sy))

    if xs:
        min_x_reach = min(xs)
        max_x_reach = max(xs)
        margin = p3.radius + STICK_LENGTH + PUCK_DIAMETER

        # Vertical movement box: full vertical range, narrow horizontal around start_x
        vert_box_x = int(p3.start_x - margin/2)
        vert_box_y = int(p3.min_y)
        vert_box_w = int(margin)
        vert_box_h = int(p3.max_y - p3.min_y)
        pygame.draw.rect(screen, (0, 0, 255), (vert_box_x, vert_box_y, vert_box_w, vert_box_h), 3)

        # Horizontal movement box: full horizontal reach, narrow vertical around max_y
        horiz_box_x = int(max(0, min_x_reach - margin))
        horiz_box_y = int(p3.max_y - margin/2)
        horiz_box_w = int(min(WIDTH, max_x_reach + margin) - horiz_box_x)
        horiz_box_h = int(margin)
        pygame.draw.rect(screen, (0, 0, 255), (horiz_box_x, horiz_box_y, horiz_box_w, horiz_box_h), 3)

    # --- Draw two blue boxes for Player 4 hitting region ---
    for p_idx in [3, 4]:  # Player 4 and 5
        p = players[p_idx]
        margin = p.radius + STICK_LENGTH + PUCK_DIAMETER

        # Vertical movement box
        vert_box_x = int(p.start_x - margin/2)
        vert_box_y = int(p.min_y)
        vert_box_w = int(margin)
        vert_box_h = int(p.max_y - p.min_y)
        pygame.draw.rect(screen, (0, 0, 255), (vert_box_x, vert_box_y, vert_box_w, vert_box_h), 3)



def calculate_distance_to_target(puck, target_player_idx, prioritize_horizontal=False):
    """Calculate distance from puck to target zone center"""
    if target_player_idx == 0:  # Player 1
        center_x = (TARGET_X_MIN + TARGET_X_MAX) / 2
        center_y = (TARGET_Y_MIN + TARGET_Y_MAX) / 2
    else:  # Player 2
        center_x = (TARGET_X_MIN_P2 + TARGET_X_MAX_P2) / 2
        center_y = (TARGET_Y_MIN_P2 + TARGET_Y_MAX_P2) / 2
    
    # If prioritizing horizontal, weight X distance much more heavily
    if prioritize_horizontal:
        x_distance = abs(puck.x - center_x)
        y_distance = abs(puck.y - center_y)
        # Weight horizontal distance 5x more than vertical
        return x_distance * 5.0 + y_distance
    else:
        return math.hypot(puck.x - center_x, puck.y - center_y)

def generate_smart_action(puck_x, puck_y, player_idx, target_player_idx):
    """Generate action parameters based on puck and target positions"""
    player = players[player_idx]
    
    # Calculate where we need to move to reach the puck
    # Use actual max_y instead of range_limit
    y_in_range = max(player.min_y, min(player.max_y, puck_y))
    
    # Calculate linear_distance based on actual movement range
    movement_range = player.max_y - player.min_y
    linear_distance = (y_in_range - player.min_y) / movement_range
    linear_distance = max(0, min(1, linear_distance))
    
    # For positions near the extreme of the range, add extra variance
    # to explore different approach angles
    if linear_distance > 0.8:
        linear_distance += random.uniform(-0.15, 0.05)
    else:
        linear_distance += random.uniform(-0.1, 0.1)
    
    # Calculate angle to hit puck toward target
    if target_player_idx == 0:  # Target is Player 1's zone
        target_x = (TARGET_X_MIN + TARGET_X_MAX) / 2
        target_y = (TARGET_Y_MIN + TARGET_Y_MAX) / 2
    else:  # Target is Player 2's zone
        target_x = (TARGET_X_MIN_P2 + TARGET_X_MAX_P2) / 2
        target_y = (TARGET_Y_MIN_P2 + TARGET_Y_MAX_P2) / 2
    
    # Calculate player's position at target y using the player's mapping
    progress_for_angle_y = max(player.min_y, min(player.max_y, puck_y))
    estimated_player_x = player.get_x_for_y(progress_for_angle_y)
    
    # Calculate desired angle from puck to target (we want to strike puck toward target)
    angle_to_target = math.atan2(target_y - puck_y, target_x - puck_x)
    
    # Calculate rotation needed from default angle (-pi/2)
    rotation_needed = angle_to_target - (-math.pi / 2)
    
    # Normalize rotation to [-pi, pi]
    while rotation_needed > math.pi:
        rotation_needed -= 2 * math.pi
    while rotation_needed < -math.pi:
        rotation_needed += 2 * math.pi
    
    # Convert to normalized rotation_angle parameter [-1, 1]
    rotation_angle = rotation_needed / (2 * math.pi)
    
    # For extreme positions, add more rotational variance
    if linear_distance > 0.8:
        rotation_angle += random.uniform(-0.4, 0.4)
    else:
        rotation_angle += random.uniform(-0.15, 0.15)
    
    # Clamp values
    linear_distance = max(0, min(1, linear_distance))
    rotation_angle = max(-1, min(1, rotation_angle))
    
    # Use high speeds for reliable contact
    linear_speed = random.uniform(0.8, 1.0)
    rotation_speed = random.uniform(0.8, 1.0)
    
    return [linear_distance, linear_speed, rotation_angle, rotation_speed]

def simulate_action_scored(action, puck_x, puck_y, player_idx, target_player_idx, is_scoring=False):
    """
    Simulate action and return (success, score, final_puck_x, final_puck_y, puck_was_hit)
    Special behavior for player_idx == 2 (player3):
      - If puck_y > BOTTOM_GOAL_Y: puck is BELOW the goal line, reward upward movement.
      - If puck_y <= BOTTOM_GOAL_Y: puck is ABOVE the goal line, reward passing to Player 2 (red box).
    """
    test_player = copy.deepcopy(players[player_idx])
    test_puck = Puck(x=puck_x, y=puck_y)
    test_player.do_action(action)
    hit_after_rotation = False
    puck_was_hit = False

    # Run simulation
    for _ in range(100):
        test_player.update()
        test_puck.move()

        if test_puck.collide(test_player):
            puck_was_hit = True
            if test_player.has_rotated:
                hit_after_rotation = True

        # Check for goal (if in scoring mode)
        if is_scoring and hit_after_rotation and test_puck.y + test_puck.radius > BOTTOM_GOAL_Y + SCALE / 2:
            if WIDTH / 2 - GOAL_WIDTH / 2 < test_puck.x < WIDTH / 2 + GOAL_WIDTH / 2:
                return True, 10000.0, test_puck.x, test_puck.y, True

    # If puck wasn't hit at all, return very low score
    if not puck_was_hit:
        return False, 0.0, test_puck.x, test_puck.y, False

    # ---------- SCORING LOGIC ----------
    # Distance score (closer = better)
    final_distance = calculate_distance_to_target(test_puck, target_player_idx)
    score = 1000.0 / (final_distance + 1.0)

    # Determine whether puck ends in the target zone
    if target_player_idx == 0:  # Player 1
        in_zone = (
            TARGET_X_MIN <= test_puck.x <= TARGET_X_MAX and
            TARGET_Y_MIN <= test_puck.y <= TARGET_Y_MAX
        )
    else:  # Player 2
        in_zone = (
            TARGET_X_MIN_P2 <= test_puck.x <= TARGET_X_MAX_P2 and
            TARGET_Y_MIN_P2 <= test_puck.y <= TARGET_Y_MAX_P2
        )

    if in_zone:
        score += 500.0

    # Reward puck movement
    puck_movement = math.hypot(test_puck.x - puck_x, test_puck.y - puck_y)
    score += puck_movement * 2.0

    # Big bonus for significant movement
    if puck_movement > 50:
        score += 200

    # -------- ADD DIRECTIONAL REWARD HERE --------
    # Add directional reward - encourage movement toward target zone
    if target_player_idx == 0:  # Player 1
        target_center_y = (TARGET_Y_MIN + TARGET_Y_MAX) / 2
        
        # Check if puck needs to move up or down to reach target
        initial_y_distance = abs(puck_y - target_center_y)
        final_y_distance = abs(test_puck.y - target_center_y)
        y_improvement = initial_y_distance - final_y_distance
        
        # Reward moving closer to target Y
        if y_improvement > 0:
            score += y_improvement * 5.0  # Strong reward for correct direction
        
        # Extra bonus if moving into the target Y range
        if TARGET_Y_MIN <= test_puck.y <= TARGET_Y_MAX:
            score += 300.0

    elif target_player_idx == 1:  # Player 2
        target_center_y = (TARGET_Y_MIN_P2 + TARGET_Y_MAX_P2) / 2
        
        initial_y_distance = abs(puck_y - target_center_y)
        final_y_distance = abs(test_puck.y - target_center_y)
        y_improvement = initial_y_distance - final_y_distance
        
        if y_improvement > 0:
            score += y_improvement * 5.0
        
        if TARGET_Y_MIN_P2 <= test_puck.y <= TARGET_Y_MAX_P2:
            score += 300.0

    # -------- Player 3 special behavior ----------
    if player_idx == 2:
        if puck_y > BOTTOM_GOAL_Y:

            if (WIDTH / 2 - GOAL_WIDTH / 2) < test_puck.x < (WIDTH / 2 + GOAL_WIDTH / 2):
                print("Puck is WITHIN the goal line")
                # Puck is behind the goal AND between the posts
                # → reward moves toward x=0 or x=WIDTH (450)

                # Distance to left and right corners
                dist_left_before = abs(puck_x - 0)
                dist_right_before = abs(puck_x - WIDTH)

                dist_left_after = abs(test_puck.x - 0)
                dist_right_after = abs(test_puck.x - WIDTH)

                # Reward movement toward either corner (whichever increased)
                gain_left = dist_left_before - dist_left_after
                gain_right = dist_right_before - dist_right_after

                # Positive gain = movement toward a corner
                corner_gain = max(gain_left, gain_right, 0)

                score += corner_gain * 12.0  # strong reward for corner-clearing

                # Big bonus if puck ends near either corner
                if test_puck.x < 40 or test_puck.x > WIDTH - 40:
                    score += 1200.0

                # Small penalty for staying between goal posts
                if (WIDTH / 2 - GOAL_WIDTH / 2) < test_puck.x < (WIDTH / 2 + GOAL_WIDTH / 2):
                    score -= 200.0

            else:
                # Puck is below bottom goal line — reward upward movement toward center
                vertical_gain = puck_y - test_puck.y  # positive if puck moved up (y decreased)
                score += max(0.0, vertical_gain) * 6.0
                # bonus if puck crosses above goal line
                if test_puck.y <= BOTTOM_GOAL_Y - 2.0:
                    score += 800.0
        else:
            print("Puck is ABOVE the goal line - Target Player 2 (red box)")
            # Puck is above goal line — reward passing to Player 2's red box
            
            target_center_x = (TARGET_X_MIN + TARGET_X_MAX) / 2
            target_center_y = (TARGET_Y_MIN + TARGET_Y_MAX) / 2
            
            # DOMINANT FACTOR: Horizontal distance to center
            horizontal_distance_before = abs(puck_x - target_center_x)
            horizontal_distance_after = abs(test_puck.x - target_center_x)
            
            # Massive reward for reducing horizontal distance
            horizontal_gain = horizontal_distance_before - horizontal_distance_after
            score += horizontal_gain * 50.0  # VERY strong multiplier
            
            # Huge penalty for being far from horizontal center
            score -= horizontal_distance_after * 10.0
            
            # Big bonus if horizontally centered (within 20 pixels)
            if abs(test_puck.x - target_center_x) < 20:
                score += 3000.0
            
            # Bonus if puck ends inside target zone
            if TARGET_X_MIN <= test_puck.x <= TARGET_X_MAX and TARGET_Y_MIN <= test_puck.y <= TARGET_Y_MAX:
                score += 2000.0
            
            # Secondary: just need to be in Y range
            if TARGET_Y_MIN <= test_puck.y <= TARGET_Y_MAX:
                score += 500.0
            
            # Use horizontal-priority distance
            final_distance = calculate_distance_to_target(test_puck, 0, prioritize_horizontal=True)
            score += 1000.0 / (final_distance + 1.0)

    # final return (non-goal but scored by distance)
    return False, score, test_puck.x, test_puck.y, True

def find_best_action_smart(puck_x, puck_y, player_idx, target_player_idx, is_scoring=False, max_attempts=10000):
    """Find best action using smart generation + random search"""
    best_action = None
    best_score = -1
    best_puck_pos = (puck_x, puck_y)
    puck_hit_count = 0
    
    player = players[player_idx]
    
    # Check if puck is at extreme range using actual movement range
    movement_range = player.max_y - player.min_y
    y_in_range = max(player.min_y, min(player.max_y, puck_y))
    linear_distance_needed = (y_in_range - player.min_y) / movement_range
    is_extreme_range = linear_distance_needed > 0.8
    
    if is_extreme_range:
        print(f"⚠ Puck at extreme range (distance={linear_distance_needed:.2f})")
        print(f"  Player range: y={player.min_y:.1f} to {player.max_y:.1f}")
        print(f"  Puck y: {puck_y:.1f}")
        print(f"  Increasing search diversity...")
        # Increase random exploration for extreme cases
        smart_ratio = 0.4
        max_attempts = int(max_attempts * 2)  # More attempts for hard cases
    else:
        smart_ratio = 0.7
    
    print(f"Searching for best action (max {max_attempts} attempts)...")
    
    # Phase 1: Smart guided search
    smart_attempts = int(max_attempts * smart_ratio)
    for i in range(smart_attempts):
        if i % 1000 == 0:
            print(f"Smart search: {i}/{smart_attempts}, best score: {best_score:.2f}, hits: {puck_hit_count}")
        
        action = generate_smart_action(puck_x, puck_y, player_idx, target_player_idx)
        success, score, final_x, final_y, puck_hit = simulate_action_scored(
            action, puck_x, puck_y, player_idx, target_player_idx, is_scoring
        )
        
        if puck_hit:
            puck_hit_count += 1
        
        if success:
            print(f"✓ Found goal-scoring action! (attempt {i})")
            return action
        
        if score > best_score:
            best_score = score
            best_action = action
            best_puck_pos = (final_x, final_y)
            
            # If we found a really good action, we can stop early (unless at extreme range)
            if score > 800 and puck_hit and not is_extreme_range:
                print(f"✓ Found excellent action (score={score:.2f}) at attempt {i}")
                break
    
    # Phase 2: Random exploration
    random_attempts = int(max_attempts * (1.0 - smart_ratio))
    for i in range(random_attempts):
        if i % 500 == 0:
            print(f"Random search: {i}/{random_attempts}, best score: {best_score:.2f}, hits: {puck_hit_count}")
        
        # For extreme ranges, bias toward high linear distances
        if is_extreme_range:
            linear_dist = random.uniform(0.75, 1.0)  # Focus on extended positions
        else:
            linear_dist = random.uniform(0, 1)
        
        action = [
            linear_dist,
            random.uniform(0.6, 1),
            random.uniform(-1, 1),
            random.uniform(0.6, 1)
        ]
        success, score, final_x, final_y, puck_hit = simulate_action_scored(
            action, puck_x, puck_y, player_idx, target_player_idx, is_scoring
        )
        
        if puck_hit:
            puck_hit_count += 1
        
        if success:
            print(f"✓ Found goal-scoring action! (random attempt {i})")
            return action
        
        if score > best_score:
            best_score = score
            best_action = action
            best_puck_pos = (final_x, final_y)
    
    if best_action:
        print(f"\nBest action found:")
        print(f"  Score: {best_score:.2f}")
        print(f"  Final position: ({best_puck_pos[0]:.1f}, {best_puck_pos[1]:.1f})")
        print(f"  Total successful hits: {puck_hit_count}/{max_attempts}")
    else:
        print(f"⚠ No valid action found after {max_attempts} attempts")
        print(f"  This may indicate the puck is unreachable from this position")
    
    return best_action

def puck_in_target_zone(puck, player_idx):
    if player_idx == 0:  # Player 1
        return TARGET_X_MIN <= puck.x <= TARGET_X_MAX and TARGET_Y_MIN <= puck.y <= TARGET_Y_MAX
    else:  # Player 2
        return TARGET_X_MIN_P2 <= puck.x <= TARGET_X_MAX_P2 and TARGET_Y_MIN_P2 <= puck.y <= TARGET_Y_MAX_P2

def visualize_single_episode(action, puck_x, puck_y, player_idx):
    episode_timer = 0
    episode_duration = 120

    players[player_idx].do_action(action)
    puck = Puck(x=puck_x, y=puck_y)

    while True:
        screen.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        players[player_idx].update()
        puck.move()
        puck.collide(players[player_idx])
        draw_field(screen)
        for p in players:
            p.draw(screen)
        puck.draw(screen)
        pygame.display.flip()
        clock.tick(60)

        episode_timer += 1
        if episode_timer >= episode_duration:
            pygame.quit()
            return

def puck_in_hitting_range(puck_x, puck_y, player_idx):
    """Check if puck is in hitting range of a player"""
    player = players[player_idx]
    
    # Check if puck is within vertical range
    if not (player.min_y <= puck_y <= player.max_y):
        return False
    
    max_horizontal_reach = player.radius + STICK_LENGTH + PUCK_DIAMETER
    
    # For Player 3 (curved player), check entire horizontal movement range
    if player_idx == 2 and isinstance(player, CurvedPlayer):
        # Sample x-values along the vertical range to find min/max horizontal reach
        sample_points = 20
        y_step = (player.max_y - player.min_y) / sample_points
        x_positions = []
        
        for i in range(sample_points + 1):
            sample_y = player.min_y + i * y_step
            x_positions.append(player.get_x_for_y(sample_y))
        
        min_x_reach = min(x_positions) - max_horizontal_reach
        max_x_reach = max(x_positions) + max_horizontal_reach
        
        # Check if puck is within the horizontal range at this y-coordinate
        return min_x_reach <= puck_x <= max_x_reach
    
    else:
        # For regular players, calculate player position at puck's y using player's mapping
        player_x_at_puck = player.get_x_for_y(puck_y)
        
        # Check if puck is within horizontal reach
        horizontal_dist = abs(puck_x - player_x_at_puck)
        
        return horizontal_dist <= max_horizontal_reach

# --- Main logic ---
puck_x = args.puck_x if args.puck_x is not None else WIDTH // 2
puck_y = args.puck_y if args.puck_y is not None else HEIGHT // 2

player1_can_reach = puck_in_hitting_range(puck_x, puck_y, 0)
player2_can_reach = puck_in_hitting_range(puck_x, puck_y, 1)
player3_can_reach = puck_in_hitting_range(puck_x, puck_y, 2)

print(f"Puck position: x={puck_x}, y={puck_y}")
print(f"Player 1 can reach: {player1_can_reach}")
print(f"Player 2 can reach: {player2_can_reach}")
print(f"Player 3 can reach (curved): {player3_can_reach}")

puck_in_p1_zone = TARGET_X_MIN <= puck_x <= TARGET_X_MAX and TARGET_Y_MIN <= puck_y <= TARGET_Y_MAX
puck_in_p2_zone = TARGET_X_MIN_P2 <= puck_x <= TARGET_X_MAX_P2 and TARGET_Y_MIN_P2 <= puck_y <= TARGET_Y_MAX_P2

print(f"Puck in Player 1 zone: {puck_in_p1_zone}")
print(f"Puck in Player 2 zone: {puck_in_p2_zone}")

# choose which player to use:
# preference logic now checks the third player as well (choose nearest capable player)
capable_players = [i for i in range(len(players)) if puck_in_hitting_range(puck_x, puck_y, i)]
if capable_players:
    # choose player with smallest horizontal distance to puck
    distances = [(i, abs(puck_x - players[i].get_x_for_y(puck_y))) for i in capable_players]
    distances.sort(key=lambda t: t[1])
    chosen_idx = distances[0][0]
else:
    chosen_idx = None

if chosen_idx is None:
    print("Puck is not in hitting range of any player.")
    pygame.quit()
    sys.exit()

print(f"Using player index: {chosen_idx}")

if chosen_idx == 0:
    # existing logic for player1
    if TARGET_X_MIN <= puck_x <= TARGET_X_MAX and TARGET_Y_MIN <= puck_y <= TARGET_Y_MAX:
        print("Puck is in Player 1's scoring zone. Finding action to score goal...")
        scoring_action = find_best_action_smart(puck_x, puck_y, 0, 0, is_scoring=True, max_attempts=10000)
        if scoring_action:
            print(f"Action found: {scoring_action}")
            visualize_single_episode(scoring_action, puck_x, puck_y, 0)
        else:
            print("No action found.")
            pygame.quit()
    else:
        print("Puck not in Player 1's target zone. Finding action to move puck into scoring zone...")
        setup_action = find_best_action_smart(puck_x, puck_y, 0, 0, is_scoring=False, max_attempts=10000)
        if setup_action:
            print(f"Action found: {setup_action}")
            visualize_single_episode(setup_action, puck_x, puck_y, 0)
        else:
            print("No action found.")
        pygame.quit()

else:
    # If chosen player is not player1 (could be player2 or player3), try moving puck toward player1's scoring zone
    print(f"\n=== Using Player {chosen_idx + 1} ===")
    print("Finding action to move puck to Player 1's scoring zone...")
    setup_action = find_best_action_smart(puck_x, puck_y, chosen_idx, 0, is_scoring=False, max_attempts=10000)
    if setup_action:
        print(f"Action found: {setup_action}")
        visualize_single_episode(setup_action, puck_x, puck_y, chosen_idx)
    else:
        print("No action found.")
    pygame.quit()
