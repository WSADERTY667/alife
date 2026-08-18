# Utils module for ALife simulation

import math
from alife import config


def clamp(v, lo=0.0, hi=1.0):
    """Clamp value to [lo, hi] range."""
    if v < lo:
        return float(lo)
    if v > hi:
        return float(hi)
    return float(v)


def normalize_angle(a):
    """Normalize angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def wall_front_sensor(pos, angle):
    """
    Sensor that detects distance to wall in front of agent.
    Returns normalized value (1.0 = very close to wall, 0.0 = far).
    """
    dx = math.cos(angle)
    dy = math.sin(angle)
    dists = []

    if dx > 1e-6:
        dists.append((config.WORLD_W - pos[0]) / dx)
    if dx < -1e-6:
        dists.append((0.0 - pos[0]) / dx)

    if dy > 1e-6:
        dists.append((config.WORLD_H - pos[1]) / dy)
    if dy < -1e-6:
        dists.append((0.0 - pos[1]) / dy)

    d = min(dists) if dists else 1000.0
    return clamp(1.0 - d / 120.0, 0.0, 1.0)
