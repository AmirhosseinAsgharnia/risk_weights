"""
Stanley steering controller — lane-keeping for a bicycle-model vehicle.

For surrounding traffic only, not the ego vehicle (ego will be driven by an
MPC, designed later).
"""

import math


def stanley_steering(e_y: float, e_psi: float, v: float, k: float, eps: float = 1e-3) -> float:
    """
    Front-wheel steering command driving cross-track error e_y and heading
    error e_psi to zero:
        delta = -e_psi - atan(k * e_y / (v + eps))

    Sign convention (SAE J670, +right — CarState's convention, where
    positive delta yaws the car rightward — see CarDynamics):
      - e_y is the car's lateral offset from the target path, +right. A car
        to the right of the path (e_y > 0) must turn left to correct,
        hence the minus sign on the cross-track term.
      - e_psi, per CarDynamics' own state ODE (d(e_psi)/dt = r - kappa*s_dot
        = d(psi_vehicle)/dt - d(psi_path)/dt), accumulates as
        psi_vehicle - psi_path (actual heading minus the path's), not the
        reverse. A car yawed clockwise of the path (e_psi > 0) must turn
        left (negative delta) to correct, hence the minus sign here too --
        leaving this term unflipped is a positive-feedback loop that
        diverges (verified: e_psi and delta blow up within a few seconds).

    Inputs:
      e_y:   [m] lateral offset from the target path (e.g. the car's lane
             centreline), +right.
      e_psi: [rad] heading error, psi_vehicle - psi_path.
      v:     [m/s] longitudinal speed (>= 0 expected).
      k:     [-] cross-track gain — larger k corrects lateral error more
             aggressively, at the cost of a sharper correction at low v.
      eps:   [m/s] speed floor added to the denominator so the atan term
             stays finite as v -> 0.

    Returns:
      delta: [rad] commanded front-wheel steering angle.
    """
    return -e_psi - math.atan(k * e_y / (v + eps))
