"""Pure episode bookkeeping for a physical automatic spring latch.

State columns: opened, hand touched door, released, supported, dwell, last step,
success, failed. No prescribed grasp history and no simulated locking.
"""

import math

import torch


def new_door_latch_state(num_envs, device):
    state = torch.zeros((num_envs, 8), device=device)
    state[:, 5] = -1
    return state


def update_door_latch_state(
    state,
    step,
    door_angle,
    door_speed,
    latch_height,
    latch_speed,
    door_hand_contact,
    supported,
    hands_clear,
    hold_steps=60,
):
    """Count consecutive stable, supported, hands-free steps after door contact."""
    if hold_steps < 1:
        raise ValueError("hold_steps must be positive")
    fresh = step != state[:, 5]
    reset = step < state[:, 5]
    state[reset] = 0
    state[reset, 5] = -1
    # A skipped evaluation cannot certify the intervening release interval.
    gap = step > state[:, 5] + 1
    state[gap, 4] = 0
    active = fresh & ~state[:, 7].bool()
    # Start closed without failing. A return to closed after 15 degrees fails.
    state[active, 0] = torch.maximum(state[active, 0], (door_angle[active] >= math.radians(15)).float())
    state[active, 1] = torch.maximum(state[active, 1], door_hand_contact[active].float())
    released = state[:, 1].bool() & hands_clear & ~door_hand_contact
    state[fresh, 2] = released[fresh].float()
    state[fresh, 3] = supported[fresh].float()
    shut = state[:, 0].bool() & (door_angle <= math.radians(5))
    state[fresh, 7] = torch.maximum(state[fresh, 7], shut[fresh].float())
    seated = (
        (door_angle >= math.radians(84))
        & (door_angle <= math.radians(94))
        & (door_speed.abs() < 0.08)
        & (latch_height <= 0.012)
        & (latch_speed.abs() < 0.03)
        & supported
        & released
        & ~state[:, 7].bool()
    )
    state[fresh, 4] = torch.where(seated[fresh], state[fresh, 4] + 1, 0)
    # Success is a current physical dwell, never a sticky angle-only latch.
    state[fresh, 6] = (state[fresh, 4] >= hold_steps).float()
    state[fresh, 5] = step[fresh].float()
    return state[:, 6].bool(), state[:, 7].bool()
