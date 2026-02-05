# Rod Hockey Robot

An autonomous rod hockey (bubble hockey) robot that uses computer vision and physics simulation to play the game.

## Overview

This project uses a camera to detect the puck position, runs a physics-based simulation to calculate optimal player movements, and executes motor commands to control the rod hockey players.

## Components

- **vision.py** - Computer vision module that detects the puck position using a camera and scales coordinates for the simulation
- **calc.py** - Physics simulation that calculates the best action to take given the current puck position
- **action.py** - Motor control module that executes the calculated movements via Viam robotics platform

## How It Works

1. The camera detects the puck position on the table
2. Coordinates are scaled from camera space to game simulation space
3. A physics simulation evaluates possible player movements
4. The best action is calculated and sent to the motors
5. Motors execute the movement and return to original position

## Requirements

- Python 3.x
- pygame
- viam-sdk
- Camera with object detection capability
- Viam robot with motors for controlling rod hockey players

## Setup

1. Install dependencies:
```bash
pip install pygame viam-sdk
```

2. Configure your Viam robot credentials in `action.py` and `vision.py`

3. Calibrate camera coordinates in `vision.py` to match your table dimensions

## Usage

Run the simulation with puck coordinates:
```bash
python calc.py --puck_x <x_coordinate> --puck_y <y_coordinate>
```

Get puck coordinates from the camera:
```bash
python vision.py
```

## Project Structure

```
├── action.py       # Motor control and execution
├── calc.py         # Physics simulation and action calculation
├── vision.py       # Computer vision and puck detection
└── README.md       # This file
```

## Built With

- [Viam](https://www.viam.com/) - Robotics platform for hardware control
- [Pygame](https://www.pygame.org/) - Physics simulation and visualization

## License

MIT
