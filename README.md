# Bubble Hockey Robot

An autonomous bubble hockey robot that uses computer vision and physics simulation to play the game.

## Overview

The robot uses a camera to detect the puck position, runs a physics-based simulation to find the optimal player action, and sends motor commands via the Viam robotics platform.

## Project Structure

```
├── main.py          # Robot entry point
├── simulate.py      # CLI simulator for testing
├── engine/          # Python game engine
│   ├── constants.py     # Numeric constants, zone boundaries, PlayerID enum
│   ├── entities.py      # Player, CurvedPlayer, Puck classes + player instances
│   ├── planner.py       # Action planning and physics simulation
│   └── display.py       # pygame rendering (field, episode playback)
└── robot/           # Viam hardware integration
    ├── const.py         # Robot credentials, camera bounds, motor constants
    ├── vision.py        # Camera-based puck detection
    └── execution.py     # Motor command execution
```

## How It Works

1. **Vision** — the camera detects the puck position and scales it to game coordinates
2. **Planning** — `plan_action` selects the best player for the puck's zone and searches for the optimal action using a physics simulation
3. **Execution** — motor commands are sent to the robot via Viam

### Action Planning (`engine/planner.py`)

| Function                           | Description                                                      |
| ---------------------------------- | ---------------------------------------------------------------- |
| `plan_action(puck_x, puck_y)`      | Top-level: picks player by zone, returns `(action, PlayerID)`    |
| `find_best_action_for_player(...)` | Two-phase search (guided + random) for a given player and target |
| `simulate_action_for_player(...)`  | Runs a single 100-frame physics simulation and returns a score   |

## Requirements

- Python 3.x
- pygame
- viam-sdk

## Setup

1. Activate the virtual environment:

```bash
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install pygame viam-sdk
```

3. Configure credentials, camera bounds, and motor constants in `robot/const.py`

## Usage

### Run the robot

Detects the puck, plans the best action, and executes it:

```bash
python main.py
```

### Simulate a shot (with visualization)

```bash
python simulate.py --puck_x 225 --puck_y 400
```

### Loop mode — feed each final puck position back as the next start

```bash
python simulate.py --puck_x 225 --puck_y 400 --loop
```

### Headless mode — no pygame window, just prints the action

```bash
python simulate.py --puck_x 225 --puck_y 400 --headless
```

### Headless loop

```bash
python simulate.py --puck_x 225 --puck_y 400 --loop --headless
```

## Built With

- [Viam](https://www.viam.com/) - Robotics platform for hardware control
- [Pygame](https://www.pygame.org/) - Physics simulation and visualization

## License

MIT
