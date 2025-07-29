# SnakeAI - Genetic Algorithm with Heuristic Model

## Overview

This project implements an AI for the classic "Snake" game using a **heuristic model** guided by a **genetic algorithm**. Instead of utilizing neural networks, the snake's intelligence is driven by weighted behavioral traits that evolve through generations. The AI starts off as a random player and progressively becomes an expert by learning through simulated evolution.

### Key Features

- ** Heuristic-Based AI**: Behavioral genes represent decision-making tendencies (e.g., food attraction, danger avoidance), not network weights.
- ** Genetic Algorithm**: Implements elitism and tournament selection for evolution without the need for labeled datasets.
- ** Real-Time Visualization**: Interactive simulation built with Pygame for visual feedback.
- ** Performance Monitoring**: Fitness and scores logged per generation, with optional graph plotting.
- ** Speed Control**: Adjustable simulation speed for faster training or detailed inspection.

##  Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
pip install -r requirements.txt
```

##  Running the Simulation

### Visual Mode
```bash
python snakeGenAlg.py
```
Use keyboard controls to interact with the simulation:

| Key     | Action                                         |
|---------|------------------------------------------------|
| `ESC`   | Exit the application                           |
| `SPACE` | Toggle speed (slow/fast)                       |
| `B`     | Show best snake from the previous generation   |
| `R`     | Restart simulation from generation 1           |

### Headless Mode (Text Analysis)
Call the `run_text_simulation()` function in the script to run 100 generations without GUI and receive statistical output.

##  Project Architecture

### Module Structure
All code is located in a single file `snakeGenAlg.py`, divided into logical components:

- `DNA`: Defines the gene structure and genetic operations (crossover and mutation).
- `Snake`: Represents an individual snake with decision logic and fitness calculation.
- `GeneticAlgorithm`: Manages the population and the evolution loop.
- `GameVisualizer`: Responsible for GUI rendering using Pygame.

### AI Model Design (Heuristic Approach)

The AI's "brain" is a set of heuristic functions weighted by genes. These functions evaluate each possible move:

```
Input: Game State (snake position, food, walls)
             ↓
Heuristic Functions (scored and weighted by genes)
    ├── Food Attraction          → weighted by `food_attraction`
    ├── Wall Avoidance           → weighted by `wall_avoidance`
    ├── Self Avoidance           → weighted by `self_avoidance`
    ├── Exploration Incentive    → weighted by `exploration`
    └── Forward Movement Bias    → weighted by `forward_bias`
             ↓
Decision Logic (choose_direction)
             ↓
Output: Best move (Up / Down / Left / Right)
```

##  Fitness Function

The genetic algorithm uses a detailed fitness function:

- **Score**: +1000 per food eaten
- **Survival Bonus**: +2 per time step survived
- **Efficiency Bonus**: Reward for eating food quickly
- **Body Length**: Bonus for snake growth
- **Wandering Penalty**: Deduction if no food is eaten for a while

## Configuration

Modify these values at the top of `snakeGenAlg.py`:
```python
GRID_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20
WINDOW_WIDTH = GRID_WIDTH * GRID_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * GRID_SIZE + 100
FPS = 30
```

#### Genetic Algorithm Parameters:
```python
# For visual simulation
ga = GeneticAlgorithm(population_size=30)

# For headless mode
run_text_simulation(num_generations=100, population_size=50)
```

## Performance & Results

### Expected Results (Sample)
| Generation Range | Typical Max Score | Observed Behavior                              |
|------------------|-------------------|------------------------------------------------|
| 1-10             | 50-70             | Basic survival and food seeking                |
| 10-40            | 70-90             | Consistent food gathering and pathfinding      |
| 40+              | 85+               | Expert-level navigation and efficiency         |

### Learned Strategy Example
The final evolved gene profile typically favors:
- Strong food-seeking (`food_attraction: 1.80`)
- High self-awareness (`self_avoidance: 1.22`)
- Minimal randomness (`risk_taking: 0.18`)
- Corner proximity not punished (`corner_avoidance: 0.10`)

## Authors
- Andrea Vasiljević
- Dragan Vučićević  
- Dragan Mišić
