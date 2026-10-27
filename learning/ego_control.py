"""
Closed-loop ego-controller interface -- lets any controller (PPO, MPC, a
test stub) drive EgoTrafficEnv through its ordinary step() without touching
the env, the shared physics (model.car), or the shared traffic stepping
(model.traffic_step).

A controller only ever produces a PHYSICAL command (accel [m/s^2], delta
[rad], SAE J670: +delta steers right). to_env_action() clips it to the env's
own actuator box and maps it onto the env's normalized [-1, 1]^2 action --
the env then applies its own steering-rate/jerk limits exactly as it does
for PPO, so no controller can bypass them.

Controllers read the live simulation through EgoView: the env's own road,
agents and ego Car objects (read-only by convention -- a controller must
never mutate them). That is the same information PPO's observation is built
from (see EgoTrafficEnv._get_obs), just unflattened.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from learning.env import EgoTrafficEnv, ACCEL_MIN, ACCEL_MAX, DELTA_MAX, DT


@dataclass(frozen=True)
class EgoView:
    """What a controller may look at on each step. `env` is exposed for its
    read-only helpers (road, agents, ego_car, t, _goal_s); never mutate it."""
    env: EgoTrafficEnv
    observation: np.ndarray

    @property
    def road(self):
        return self.env.road

    @property
    def agents(self):
        return self.env.agents

    @property
    def ego(self):
        return self.env.ego_car

    @property
    def t(self) -> float:
        return self.env.t


class EgoController(Protocol):
    """reset() is called once per episode, after env.reset(); act() once per
    env step. act() returns a PHYSICAL (accel, delta) command."""

    def reset(self, view: EgoView, scenario_info: dict) -> None:
        ...

    def act(self, view: EgoView) -> tuple[float, float]:
        ...


def to_env_action(accel: float, delta: float) -> np.ndarray:
    """Physical command -> EgoTrafficEnv's normalized action. Exact inverse of
    EgoTrafficEnv._unscale_action on the clipped box."""
    accel = float(np.clip(accel, ACCEL_MIN, ACCEL_MAX))
    delta = float(np.clip(delta, -DELTA_MAX, DELTA_MAX))
    a_norm = 2.0 * (accel - ACCEL_MIN) / (ACCEL_MAX - ACCEL_MIN) - 1.0
    return np.array([a_norm, delta / DELTA_MAX], dtype=np.float32)


class PPOController:
    """Adapter so a trained SB3 policy runs through the same closed-loop
    evaluator as every other controller. Unscales its normalized action
    with the env's own mapping, so its behaviour is identical to
    learning.eval_batch.run_episodes."""

    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def reset(self, view: EgoView, scenario_info: dict) -> None:
        pass

    def act(self, view: EgoView) -> tuple[float, float]:
        action, _ = self.model.predict(view.observation, deterministic=self.deterministic)
        return view.env._unscale_action(action)


class ConstantController:
    """Fixed command every step -- a test stub / sanity baseline."""

    def __init__(self, accel: float = 0.0, delta: float = 0.0):
        self.accel, self.delta = accel, delta

    def reset(self, view: EgoView, scenario_info: dict) -> None:
        pass

    def act(self, view: EgoView) -> tuple[float, float]:
        return self.accel, self.delta


__all__ = ["EgoView", "EgoController", "to_env_action", "PPOController", "ConstantController", "DT"]
