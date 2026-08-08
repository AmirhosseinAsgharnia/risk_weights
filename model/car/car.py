import numpy as np
from dataclasses import dataclass

@dataclass
class CarState:

    e_y:    float | None = None
    e_psi:  float | None = None
    v_x:    float | None = None
    v_y:    float | None = None
    delta:  float | None = None
    x:      float | None = None
    y:      float | None = None
    s:      float | None = None
    lane:   int   | None = None

class Car:

    def __init__(self,
                 car_id     = None,
                 model      = None,
                 controller = None,
                 state      = None) -> None:


        self.car_id = car_id
        