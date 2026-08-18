#!/usr/bin/env python3
# main.py - Main entry point for ALife simulation
# 2D artificial life game with SNN brain, genome, hormones, and social dynamics.

import pygame
import random
import math
import numpy as np

from alife import (
    WORLD_W, WORLD_H, PANEL_H, SCREEN_W, SCREEN_H, FPS,
    TAG_COLORS, AGENT_RADIUS, MAX_ENERGY,
)
from alife.world import World
from alife.genome import genome_similarity


def draw_text(screen, font, x, y, text, color=(220, 220, 220)):
    """Draw text on screen."""
    screen.blit(font.render(text, True, color), (x, y))


def draw_panel(screen, world, font, selected):
    """Draw info panel at bottom of screen."""
    panel_rect = pygame.Rect(0, WORLD_H, SCREEN_W, PANEL_H)
    pygame.draw.rect(screen, (16, 18, 24), panel_rect)
    pygame.draw.line(screen, (50, 55, 70), (0, WORLD_H), (SCREEN_W, WORLD_H), 2)

    avg_gen = 0.0
    if world.agents:
        avg_gen = sum(a.generation for a in world.agents) / len(world.agents)

    y = WORLD_H + 8
    draw_text(
        screen,
        font,
        10,
        y,
        f"Tick: {world.tick}  Agents: {len(world.agents)}  Food: {len(world.foods)}  Avg gen: {avg_gen:.1f}",
    )
    y += 18
    draw_text(
        screen,
        font,
        10,
        y,
        "Keys: Space pause, F food, A agent, R reset, +/- speed, Click select, T reward, P punish",
        (170, 180, 200),
    )
    y += 24

    if selected is None:
        draw_text(screen, font, 10, y, "No agent selected.", (150, 150, 160))
        return

    h = selected.hormones
    color = TAG_COLORS[selected.genome.tag % len(TAG_COLORS)]

    draw_text(
        screen,
        font,
        10,
        y,
        f"Agent {selected.id}  gen {selected.generation}  age {int(selected.age)}  energy {selected.energy:.0f}",
        color,
    )
    y += 18

    draw_text(
        screen,
        font,
        10,
        y,
        f"D {h.D:.2f}  S {h.S:.2f}  O {h.O:.2f}  C {h.C:.2f}  T {h.T:.2f}",
    )
    y += 18

    mood = "normal"
    if h.breakdown > 0.5:
        mood = "BREAKDOWN"
    elif h.depression > 0.5:
        mood = "depressed"

    draw_text(
        screen,
        font,
        10,
        y,
        f"mood: {mood}  allostatic: {h.allostatic:.2f}  depression: {h.depression:.2f}",
    )
    y += 18

    draw_text(
        screen,
        font,
        10,
        y,
        f"tribe tag: {selected.genome.tag}  aggression: {h.effects(selected.genome)['aggression']:.2f}  sociality: {h.effects(selected.genome)['sociality']:.2f}",
        (170, 190, 220),
    )


def draw(screen, world, font, selected):
    """Draw the entire game scene."""
    screen.fill((8, 10, 14))

    for f in world.foods:
        if not f["eaten"]:
            pygame.draw.circle(
                screen,
                (70, 220, 120),
                (int(f["pos"][0]), int(f["pos"][1])),
                3,
            )

    for a in world.agents:
        base_color = TAG_COLORS[a.genome.tag % len(TAG_COLORS)]

        if a.hormones.breakdown > 0.5:
            body_color = (255, 70, 70)
        elif a.hormones.depression > 0.5:
            body_color = (90, 95, 120)
        else:
            body_color = base_color

        r = int(AGENT_RADIUS + 4.0 * clamp(a.energy / MAX_ENERGY, 0.0, 1.0))
        pos = (int(a.pos[0]), int(a.pos[1]))

        pygame.draw.circle(screen, body_color, pos, r)
        pygame.draw.circle(screen, (210, 220, 255), pos, r, 1)

        end_x = int(a.pos[0] + math.cos(a.angle) * (r + 6))
        end_y = int(a.pos[1] + math.sin(a.angle) * (r + 6))
        pygame.draw.line(screen, (220, 220, 220), pos, (end_x, end_y), 1)

        if selected is not None and selected.id == a.id:
            pygame.draw.circle(screen, (255, 255, 120), pos, r + 4, 2)

    draw_panel(screen, world, font, selected)


def clamp(v, lo=0.0, hi=1.0):
    """Clamp value to [lo, hi] range."""
    if v < lo:
        return float(lo)
    if v > hi:
        return float(hi)
    return float(v)


def get_agent_by_id(world, agent_id):
    """Get agent by ID from world."""
    if agent_id is None:
        return None
    for a in world.agents:
        if a.id == agent_id:
            return a
    return None


def main():
    """Main game loop."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("ALife MVP: SNN + Genome + Hormones")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas,monospace", 14)

    world = World()
    selected_id = None

    paused = False
    sim_speed = 1
    running = True

    print("ALife MVP started.")
    print("Space: pause | F: food | A: agent | R: reset | +/-: speed")
    print("Click agent, then T = reward, P = punish.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_f:
                    for _ in range(10):
                        world.spawn_food()

                elif event.key == pygame.K_a:
                    world.spawn_random_agent()

                elif event.key == pygame.K_r:
                    world = World()
                    selected_id = None

                elif event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                    sim_speed = min(8, sim_speed + 1)

                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    sim_speed = max(1, sim_speed - 1)

                elif event.key in (pygame.K_t, pygame.K_p):
                    sel = get_agent_by_id(world, selected_id)
                    if sel is not None:
                        if event.key == pygame.K_t:
                            sel.pending_reward += 1.0
                        elif event.key == pygame.K_p:
                            sel.pending_punishment += 1.0
                            sel.last_pain = max(sel.last_pain, 0.5)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if my < WORLD_H:
                    mouse_pos = np.array([mx, my], dtype=np.float32)
                    best = None
                    best_d = 18.0

                    for a in world.agents:
                        d = float(np.linalg.norm(a.pos - mouse_pos))
                        if d < best_d:
                            best_d = d
                            best = a

                    selected_id = best.id if best is not None else None

        if not paused:
            for _ in range(sim_speed):
                world.update()

        selected = get_agent_by_id(world, selected_id)

        draw(screen, world, font, selected)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
