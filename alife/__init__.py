# ALife module - Artificial Life Simulation

from alife.config import *
from alife.utils import clamp, normalize_angle, wall_front_sensor
from alife.genome import Genome, genome_similarity, GENOME_KEYS, BOUNDS
from alife.brain import Brain
from alife.hormones import Hormones
from alife.agent import Agent
from alife.world import World

__all__ = [
    # Config
    'WORLD_W', 'WORLD_H', 'SCREEN_W', 'SCREEN_H', 'FPS',
    'N_HIDDEN', 'INPUT_SIZE', 'OUTPUT_SIZE', 'TOTAL_NEURONS', 'LEARNING', 'SYNAPTIC_SCALE',
    'AGENT_COUNT', 'MIN_AGENTS',
    'FOOD_MAX', 'FOOD_RESPAWN',
    'MAX_ENERGY', 'START_ENERGY', 'REPRO_ENERGY', 'REPRO_COST', 'REPRO_BASE',
    'REPRODUCE_COOLDOWN', 'MATURE_AGE', 'MAX_AGE',
    'SENSE_RANGE', 'SOCIAL_RANGE', 'MATE_RANGE', 'EAT_RANGE', 'ATTACK_RANGE',
    'TURN_RATE', 'MAX_SPEED', 'AGENT_RADIUS',
    'EAT_THRESHOLD', 'ATTACK_THRESHOLD',
    'ATTACK_DAMAGE', 'ATTACK_COST',
    'REFLEX_ASSIST', 'LAMARCKIAN',
    'TAG_COLORS',
    # Utils
    'clamp', 'normalize_angle', 'wall_front_sensor',
    # Genome
    'Genome', 'genome_similarity', 'GENOME_KEYS', 'BOUNDS',
    # Brain
    'Brain',
    # Hormones
    'Hormones',
    # Agent
    'Agent',
    # World
    'World',
]
