import numpy as np
from model.road import RoadScenario
from model.car_init import Car, init_surr, resolve_params
from model.config import VehicleParameters
from model.IDM_MOBIL import IDMParams, MOBILParams
from model.traffic import build_snapshot


def test_each_traffic_car_gets_its_own_v0_not_the_shared_default():
    road = RoadScenario(lane_num=3)
    cars = init_surr(road, ego_speed=20.0, cars=[(0, 0.0, 5.0), (1, 0.0, -5.0)])
    shared_default = IDMParams(v0=30.0)  # deliberately different from either car's v0
    snapshot = build_snapshot(cars, road, shared_default, MOBILParams())

    v0s = {s.id: s.idm_params.v0 for s in snapshot}
    assert v0s[cars[0].id] == 25.0  # ego_speed + 5
    assert v0s[cars[1].id] == 15.0  # ego_speed - 5
    assert 30.0 not in v0s.values()


def test_a_car_can_override_its_own_full_idm_params():
    road = RoadScenario(lane_num=2)
    custom = IDMParams(v0=40.0, T=0.5, a_max=3.0)
    car = Car(current_lane=0, target_lane=0, state=np.array([0.0, road.d_c[0], 0.0, 20.0, 0.0, 0.0]),
              idm_params=custom)
    default = IDMParams(v0=30.0)
    snapshot = build_snapshot([car], road, default, MOBILParams())
    assert snapshot[0].idm_params.v0 == 40.0
    assert snapshot[0].idm_params.T == 0.5


def test_a_car_can_have_its_own_vehicle_params_without_affecting_others():
    road = RoadScenario(lane_num=2)
    heavy = VehicleParameters(m=3000.0)
    light_default = VehicleParameters(m=1500.0)

    car_heavy = Car(current_lane=0, target_lane=0, state=np.array([0.0, road.d_c[0], 0.0, 20.0, 0.0, 0.0]),
                     vehicle_params=heavy)
    car_default = Car(current_lane=1, target_lane=1, state=np.array([0.0, road.d_c[1], 0.0, 20.0, 0.0, 0.0]))

    assert resolve_params(car_heavy.vehicle_params, light_default).m == 3000.0
    assert resolve_params(car_default.vehicle_params, light_default).m == 1500.0


def test_actor_ids_are_unique_across_ego_and_traffic():
    road = RoadScenario(lane_num=3)
    cars = init_surr(road, ego_speed=20.0, cars=[(0, 0.0, 0.0), (1, 10.0, 0.0), (2, 20.0, 0.0)])
    ids = [c.id for c in cars]
    assert len(ids) == len(set(ids))
