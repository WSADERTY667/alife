# ALife Creatures-like Prototype

2D artificial life simulation with:
- spiking neural network brain
- reward-modulated STDP learning
- genome bottleneck
- Darwin-Lamarckian inheritance
- hormone-based mood modulation
- social behavior / tribes

## Requirements

- Python 3.8+
- pygame
- numpy

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Controls

- **Space**: pause/resume simulation
- **F**: add food
- **A**: add agent
- **R**: reset world
- **+/-**: increase/decrease simulation speed
- **Click** on an agent to select it
- **T**: give reward to selected agent
- **P**: punish selected agent
- **Escape**: quit

## Project Structure

```
alife/          # Core simulation modules
├── __init__.py     # Package exports
├── config.py       # Configuration parameters
├── utils.py        # Utility functions
├── genome.py       # Genome and genetics
├── brain.py        # Spiking neural network
├── hormones.py     # Hormone system
├── agent.py        # Agent class
└── world.py        # World simulation

tests/          # Test suite
├── test_genome.py
└── test_world.py

main.py         # Entry point with pygame rendering
requirements.txt
README.md
```

## Tests

Run the test suite:

```bash
pytest
```

## Features

### Neural Network
- Input layer: sensors (hunger, food, agents, walls, pain)
- Hidden layer: spiking neurons with membrane potential
- Output layer: motor controls (movement, eating, attack)
- Learning: reward-modulated STDP (Spike-Timing-Dependent Plasticity)

### Genetics
- Heritable genome with mutation
- Crossover during reproduction
- Lamarckian inheritance of learned weights (optional)

### Hormones
- **Dopamine**: reward/motivation
- **Serotonin**: mood stability
- **Oxytocin**: social bonding
- **Cortisol**: stress response
- **Testosterone**: aggression/dominance

### Social Dynamics
- Kin recognition via genome similarity
- Tribe tags (visual + behavioral)
- Aggression modulated by hormones and kinship
