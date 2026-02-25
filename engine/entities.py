"""
Game entities: Player, CurvedPlayer, Puck, and the shared player list.

These classes are used by both the simulation (pure math) and visualization
(pygame drawing). The simulation never calls .draw(), so this module can be
imported in headless environments as long as pygame is installed.
"""

import math
import pygame

from .constants import (
    SCALE, PLAYER_DIAMETER, STICK_LENGTH, PUCK_DIAMETER,
    HALF_FIELD_LENGTH, BEHIND_GOAL_SPACE, PLAYER_MIN_DIST_FROM_GOAL,
    BOTTOM_GOAL_Y, GOAL_WIDTH, WIDTH, HEIGHT,
    BLUE, RED, BLACK,
    center_x, min_y_center, max_y_center,
    right_wing_x, min_y_right_wing, max_y_right_wing,
    right_d_x, min_y_right_d, max_y_right_d,
    left_d_x, min_y_left_d, max_y_left_d,
)


# ============================================================
#  Player
# ============================================================

class Player:
    """A linearly-moving player figure on a rod.

    Moves vertically within [min_y, max_y], then rotates its stick
    to strike the puck. Actions are normalized vectors [0, 1].
    """

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

    def get_x_for_y(self, y):
        """Map y position to x position (small horizontal drift across the range)."""
        movement_range = max(1e-6, self.max_y - self.min_y)
        progress = max(0.0, min(1.0, (y - self.min_y) / movement_range))
        return self.start_x + progress * SCALE * 1.0

    def do_action(self, action):
        """Apply a normalized action vector [linear_dist, linear_speed, rotation_angle, rotation_speed]."""
        linear_distance, linear_speed, rotation_angle, rotation_speed = action
        self.has_rotated = False
        self.has_hit_puck = False
        self.x = self.start_x
        self.y = self.min_y
        self.angle = -math.pi / 2
        if not self.is_goalie:
            if linear_speed > 0:
                movement_range = self.max_y - self.min_y
                self.movement_speed = linear_speed * self.max_linear_speed
                target_y = self.min_y + linear_distance * movement_range
                self.target_y = max(self.min_y + self.radius,
                                    min(self.max_y - self.radius, target_y))
                self.target_x = self.get_x_for_y(self.target_y)
            if rotation_speed > 0:
                self.rotation_speed = rotation_speed * self.max_rotation_speed
                self.target_angle = self.angle + rotation_angle * 2 * math.pi
            self.movement_in_progress = True
            self.position_reached = False
        return self.x, self.y, self.angle

    def update(self):
        """Advance one frame: slide to target y, then rotate the stick."""
        if not self.is_goalie and self.movement_in_progress:
            move_speed = getattr(self, 'movement_speed', 0)
            rot_speed  = getattr(self, 'rotation_speed', 0)
            if not self.position_reached:
                if abs(self.y - self.target_y) > move_speed:
                    self.y += (1 if self.target_y > self.y else -1) * move_speed
                    self.x = self.get_x_for_y(self.y)
                else:
                    self.y = self.target_y
                    self.x = self.get_x_for_y(self.y)
                    self.position_reached = True
            elif abs(self.angle - self.target_angle) > rot_speed:
                self.angle += (1 if self.target_angle > self.angle else -1) * rot_speed
                self.has_rotated = True
            else:
                self.angle = self.target_angle
                self.movement_in_progress = False
                self.position_reached = False
                if self.angle != -math.pi / 2:
                    self.has_rotated = True
        return self.x, self.y, self.angle

    def draw(self, screen):
        color = BLUE if self.team == 'blue' else RED
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), int(self.radius))
        dx = math.cos(self.angle)
        dy = math.sin(self.angle)
        start = (self.x + dx * self.radius * 0.5, self.y + dy * self.radius * 0.5)
        end   = (self.x + dx * (self.radius * 0.5 + STICK_LENGTH),
                 self.y + dy * (self.radius * 0.5 + STICK_LENGTH))
        pygame.draw.line(screen, BLACK, start, end, 3)


# ============================================================
#  CurvedPlayer  (Left Wing)
# ============================================================

class CurvedPlayer(Player):
    """Left Wing player with a two-segment path.

    Segment A (vertical sweep): y in [y_min_a, y_max_a] — small x drift.
    Segment B (behind-net sweep): y in [y_min_b, y_max_b] — wide horizontal sweep.

    Coordinates are provided in inches and multiplied by SCALE to pixels.
    """

    def __init__(self, y_min_a, y_max_a, x_min_a, x_max_a,
                       y_min_b, y_max_b, x_min_b, x_max_b,
                       team='blue', start_x=None):
        self.y_min_a = y_min_a * SCALE
        self.y_max_a = y_max_a * SCALE
        self.x_min_a = x_min_a * SCALE
        self.x_max_a = x_max_a * SCALE
        self.y_min_b = y_min_b * SCALE
        self.y_max_b = y_max_b * SCALE
        self.x_min_b = x_min_b * SCALE
        self.x_max_b = x_max_b * SCALE

        min_y = min(self.y_min_a, self.y_min_b)
        max_y = max(self.y_max_a, self.y_max_b)

        if start_x is None:
            start_x = self.x_min_a

        super().__init__(start_x, min_y, team, is_goalie=False, min_y=min_y, max_y=max_y)
        self.start_x = start_x
        self.x = self.get_x_for_y(self.y)
        self.y = min_y

    def get_x_for_y(self, y):
        """Piecewise x-mapping for the two-segment curved path."""
        # Segment A: vertical sweep on the right side of the net
        if y <= self.y_max_a:
            denom = max(1e-6, self.y_max_a - self.y_min_a)
            prog  = max(0.0, min(1.0, (y - self.y_min_a) / denom))
            return self.x_min_a + prog * (self.x_max_a - self.x_min_a)

        # Segment B: horizontal sweep behind the net
        if y >= self.y_min_b:
            turn_y = self.y_min_b + 0.3 * (self.y_max_b - self.y_min_b)
            if y <= turn_y:
                # Sweep from right to left
                prog = (y - self.y_min_b) / max(1e-6, turn_y - self.y_min_b)
                return self.x_max_b + prog * (self.x_min_b - self.x_max_b)
            else:
                # Vertical drop at constant x
                return self.x_min_b

        # Transition zone between A and B
        denom = max(1e-6, self.y_min_b - self.y_max_a)
        prog  = max(0.0, min(1.0, (y - self.y_max_a) / denom))
        return self.x_max_a + prog * (self.x_max_b - self.x_max_a)


# ============================================================
#  Puck
# ============================================================

class Puck:
    """The puck — moves with friction, bounces off walls and sticks."""

    def __init__(self, x=None, y=None):
        self.x = x if x is not None else WIDTH // 2
        self.y = y if y is not None else HEIGHT // 2
        self.radius = PUCK_DIAMETER / 2
        self.vx = 0
        self.vy = 0
        self.friction = 0.925
        self.was_hit = False

    def move(self):
        self.x  += self.vx
        self.y  += self.vy
        self.vx *= self.friction
        self.vy *= self.friction
        if abs(self.vx) < 0.1: self.vx = 0
        if abs(self.vy) < 0.1: self.vy = 0
        # Side walls
        if self.x - self.radius < 0 or self.x + self.radius > WIDTH:
            self.vx *= -1
            self.x = max(self.radius, min(WIDTH - self.radius, self.x))
        # Top / bottom
        if self.y - self.radius < 0:
            self.vy *= -1
            self.y = self.radius
        if self.y + self.radius > HEIGHT:
            self.vy *= -1
            self.y = HEIGHT - self.radius

    def collide(self, player):
        """Apply stick-puck collision physics. Returns True if a hit occurred."""
        dx = math.cos(player.angle)
        dy = math.sin(player.angle)
        sx = player.x + dx * player.radius * 0.5
        sy = player.y + dy * player.radius * 0.5
        ex = player.x + dx * (player.radius * 0.5 + STICK_LENGTH)
        ey = player.y + dy * (player.radius * 0.5 + STICK_LENGTH)

        lx, ly = ex - sx, ey - sy
        line_len_sq = lx ** 2 + ly ** 2
        t = max(0.0, min(1.0, ((self.x - sx) * lx + (self.y - sy) * ly) / line_len_sq))
        cx = sx + t * lx
        cy = sy + t * ly

        if math.hypot(self.x - cx, self.y - cy) < self.radius + 3:
            angle = math.atan2(self.y - cy, self.x - cx)
            self.vx = math.cos(angle) * 6
            self.vy = math.sin(angle) * 6
            self.x  = cx + math.cos(angle) * (self.radius + 3)
            self.y  = cy + math.sin(angle) * (self.radius + 3)
            self.was_hit = True
            player.has_hit_puck = True
            return True
        return False

    def draw(self, screen):
        pygame.draw.circle(screen, (50, 50, 50), (int(self.x), int(self.y)), int(self.radius))
        pygame.draw.circle(screen, BLACK,         (int(self.x), int(self.y)), int(self.radius), 2)


# ============================================================
#  Player instances  (ordered by PlayerID)
# ============================================================

center = Player(
    center_x, min_y_center, 'blue',
    min_y=min_y_center, max_y=max_y_center,
)

right_wing = Player(
    right_wing_x, min_y_right_wing, 'red',
    min_y=min_y_right_wing, max_y=max_y_right_wing,
)

left_wing = CurvedPlayer(
    y_min_a=21.5, y_max_a=28.0, x_min_a=16.0, x_max_a=16.5,
    y_min_b=28.0, y_max_b=30.0, x_min_b=5.72, x_max_b=16.5,
    team='red',
    start_x=16.0 * SCALE,
)

right_d = Player(
    right_d_x, min_y_right_d, 'blue',
    min_y=min_y_right_d, max_y=max_y_right_d,
)

left_d = Player(
    left_d_x, min_y_left_d, 'blue',
    min_y=min_y_left_d, max_y=max_y_left_d,
)

# Indexed by PlayerID (CENTER=0, RIGHT_WING=1, LEFT_WING=2, RIGHT_D=3, LEFT_D=4)
players = [center, right_wing, left_wing, right_d, left_d]
