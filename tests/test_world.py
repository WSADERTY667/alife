from alife.world import World


def test_world_updates_without_crash():
    world = World()

    for _ in range(100):
        world.update()

    assert len(world.agents) > 0


def test_world_runs_1000_ticks():
    """Smoke test: verify simulation runs 1000 ticks without errors."""
    world = World()

    for _ in range(1000):
        world.update()

    # Verify world is still in valid state
    assert len(world.agents) >= 0  # Can be 0 if all died, but shouldn't crash
    assert world.tick == 1000