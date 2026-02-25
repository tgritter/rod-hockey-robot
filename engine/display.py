"""
Pygame visualization — draw_field and episode playback.

All functions that touch the pygame screen live here. The screen and clock
are created in planner.py and passed in as parameters so this module does
not own global display state.
"""

import sys
import pygame

from .constants import (
    PlayerID,
    SCALE, WIDTH, HEIGHT, GOAL_WIDTH,
    TOP_GOAL_Y, BOTTOM_GOAL_Y, PUCK_DIAMETER,
    TARGET_X_MIN, TARGET_X_MAX, TARGET_Y_MIN, TARGET_Y_MAX,
    LEFT_WING_SEG_B_X_MID,
    min_y_center,
    WHITE, RED, BLACK,
)
from .entities import players, Puck


def draw_field(screen):
    """Draw the rink, goals, target zones, and player bounding boxes."""

    # Rink outline
    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, HEIGHT), 3)

    # Center line and face-off circle
    center_y = HEIGHT / 2
    pygame.draw.line(screen, BLACK, (0, center_y), (WIDTH, center_y), 2)
    pygame.draw.circle(screen, BLACK, (WIDTH / 2, center_y), SCALE * 2, 2)

    # Goal lines
    pygame.draw.line(screen, RED, (0, TOP_GOAL_Y), (WIDTH, TOP_GOAL_Y), 2)
    pygame.draw.line(screen, RED, (0, BOTTOM_GOAL_Y + SCALE / 2), (WIDTH, BOTTOM_GOAL_Y + SCALE / 2), 2)

    # Goal nets (4-slot grid)
    goal_x = WIDTH / 2 - GOAL_WIDTH / 2
    pygame.draw.rect(screen, BLACK, (goal_x, TOP_GOAL_Y,    GOAL_WIDTH, SCALE / 2), 2)
    pygame.draw.rect(screen, BLACK, (goal_x, BOTTOM_GOAL_Y, GOAL_WIDTH, SCALE / 2), 2)
    for i in range(1, 5):
        x = goal_x + i * (GOAL_WIDTH / 5)
        pygame.draw.line(screen, BLACK, (x, TOP_GOAL_Y),    (x, TOP_GOAL_Y    + SCALE / 2), 1)
        pygame.draw.line(screen, BLACK, (x, BOTTOM_GOAL_Y), (x, BOTTOM_GOAL_Y + SCALE / 2), 1)

    # Center red scoring box
    pygame.draw.rect(screen, (255, 0, 0),
                     (TARGET_X_MIN, TARGET_Y_MIN,
                      TARGET_X_MAX - TARGET_X_MIN, TARGET_Y_MAX - TARGET_Y_MIN), 3)

    # Yellow zone (between center y-start and red box top)
    y_top, y_bottom = min(TARGET_Y_MIN, min_y_center), max(TARGET_Y_MIN, min_y_center)
    if y_bottom > y_top:
        pygame.draw.rect(screen, (255, 255, 0),
                         (int(TARGET_X_MIN), int(y_top),
                          int(TARGET_X_MAX - TARGET_X_MIN), int(y_bottom - y_top)), 3)

    # --- Left Wing (curved player) bounding boxes ---
    lw     = players[PlayerID.LEFT_WING]
    margin = lw.radius + SCALE * 1.75 + SCALE    # radius + STICK_LENGTH + PUCK_DIAMETER
    xs = [lw.get_x_for_y(sy)
          for sy in range(int(lw.min_y), int(lw.max_y) + 1,
                          max(1, int((lw.max_y - lw.min_y) / 40)))]
    if xs:
        # Vertical sweep box
        pygame.draw.rect(screen, (0, 0, 255),
                         (int(lw.start_x - margin / 2), int(lw.min_y),
                          int(margin), int(lw.max_y - lw.min_y)), 3)

        # Behind-net horizontal sweep box
        horiz_x = int(max(0, min(xs) - margin))
        horiz_y = int(lw.max_y - margin / 2)
        horiz_w = int(min(WIDTH, max(xs) + margin) - horiz_x)
        horiz_h = int(margin)
        pygame.draw.rect(screen, (0, 0, 255), (horiz_x, horiz_y, horiz_w, horiz_h), 3)

        # Vertical divider between Left Wing zone 2 (right) and zone 3 (left)
        pygame.draw.line(screen, (0, 0, 255),
                         (int(LEFT_WING_SEG_B_X_MID), horiz_y),
                         (int(LEFT_WING_SEG_B_X_MID), horiz_y + horiz_h), 3)

    # --- Bounding boxes for Right Wing, Right D, Left D ---
    for p_idx in [PlayerID.RIGHT_WING, PlayerID.RIGHT_D, PlayerID.LEFT_D]:
        p      = players[p_idx]
        margin = p.radius + SCALE * 1.75 + SCALE
        box_x  = int(p.start_x - margin / 2)
        box_y  = int(p.min_y)
        box_w  = int(margin)
        box_h  = int(p.max_y - p.min_y)
        pygame.draw.rect(screen, (0, 0, 255), (box_x, box_y, box_w, box_h), 3)

        # Right Wing: two lines dividing range into thirds
        if p_idx == PlayerID.RIGHT_WING:
            third = (p.max_y - p.min_y) / 3
            for i in [1, 2]:
                line_y = int(p.min_y + third * i)
                pygame.draw.line(screen, (0, 0, 255), (box_x, line_y), (box_x + box_w, line_y), 3)

        # Right D: one line at midpoint
        elif p_idx == PlayerID.RIGHT_D:
            mid_y = int((p.min_y + p.max_y) / 2)
            pygame.draw.line(screen, (0, 0, 255), (box_x, mid_y), (box_x + box_w, mid_y), 3)

        # Left D: two lines dividing range into thirds
        elif p_idx == PlayerID.LEFT_D:
            third = (p.max_y - p.min_y) / 3
            for i in [1, 2]:
                line_y = int(p.min_y + third * i)
                pygame.draw.line(screen, (0, 0, 255), (box_x, line_y), (box_x + box_w, line_y), 3)


def visualize_single_episode(action, puck_x, puck_y, player_idx, screen, clock):
    """Play out one action in the pygame window and return the final puck position."""
    episode_timer    = 0
    episode_duration = 120  # frames (~2 seconds at 60 fps)

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
            return puck.x, puck.y
