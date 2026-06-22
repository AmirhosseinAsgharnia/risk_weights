"""
Surrounding vehicle runtime state management.

Vehicle state dict:
  s     : absolute arc-length position along left road boundary (m)
  y_abs : absolute rightward distance from left boundary (m) — fixed at lane centre
  v_x   : longitudinal speed (m/s)
  lane  : lane index (0=rightmost)
  valid : 1 = exists, 0 = empty slot

Surrounding vehicles travel at constant speed with no lateral motion (a=0, delta=0).
"""

from environment.scenarios import SandwichScenario


def init_vehicles(scenario: SandwichScenario, s_ego_init: float = 0.0):
    """
    Build runtime vehicle list from scenario specs.
    Only valid (valid=1) vehicles get real s/y_abs values;
    invalid slots are kept for index-consistency but ignored in simulation.
    """
    vehicles = []
    for spec in scenario.vehicles:
        if spec['valid']:
            vehicles.append({
                's':     s_ego_init + spec['x_rel'],
                'y_abs': scenario.lane_center_abs(spec['lane']),
                'v_x':   spec['v_x'],
                'lane':  spec['lane'],
                'valid': 1,
            })
        else:
            vehicles.append({'s': 0.0, 'y_abs': 0.0, 'v_x': 0.0, 'lane': 0, 'valid': 0})
    return vehicles


def step_vehicles(vehicles, dt: float):
    """Advance all valid surrounding vehicles by one timestep (constant speed)."""
    return [
        {**v, 's': v['s'] + v['v_x'] * dt} if v['valid'] else v
        for v in vehicles
    ]
