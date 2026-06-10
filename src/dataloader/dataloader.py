"""
Waymo Motion dataset loader.

ego_vehicle:  (T, F)    — ego (SDC) trajectory, T timesteps, F features
sur_vehicle:  (T, F, N) — N surrounding vehicle trajectories, third dim indexes vehicles

Features (F=10): [center_x, center_y, center_z, heading, velocity_x, velocity_y,
                  length, width, height, valid]
"""

import glob
import os
import struct
from typing import Optional

import numpy as np
from waymo_open_dataset.protos import scenario_pb2

# Feature order written into the arrays
FEATURE_NAMES = [
    "center_x", "center_y", "center_z",
    "heading", "velocity_x", "velocity_y",
    "length", "width", "height",
    "valid",
]


def _iter_tfrecord(path: str):
    """Yield raw serialised proto bytes from a TFRecord file."""
    with open(path, "rb") as f:
        while True:
            len_buf = f.read(8)
            if not len_buf:
                break
            (length,) = struct.unpack("<Q", len_buf)
            f.read(4)           # masked CRC of length
            data = f.read(length)
            f.read(4)           # masked CRC of data
            yield data


def _state_to_row(state) -> np.ndarray:
    return np.array([
        state.center_x,
        state.center_y,
        state.center_z,
        state.heading,
        state.velocity_x,
        state.velocity_y,
        state.length,
        state.width,
        state.height,
        float(state.valid),
    ], dtype=np.float32)


def _track_to_array(track, num_timesteps: int) -> np.ndarray:
    """Return (T, F) array for one track; invalid steps kept with valid=0."""
    arr = np.zeros((num_timesteps, len(FEATURE_NAMES)), dtype=np.float32)
    for t, state in enumerate(track.states):
        arr[t] = _state_to_row(state)
    return arr


def _polyline_to_array(points) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)


def _parse_agents(scenario, T: int):
    """
    Returns all agent tracks as a single array plus type and ID arrays.
    Ego is NOT separated — use sdc_track_index to locate it.
    """
    N = len(scenario.tracks)
    agents     = np.zeros((T, len(FEATURE_NAMES), N), dtype=np.float32)
    agent_types = np.zeros(N, dtype=np.int32)
    agent_ids   = np.zeros(N, dtype=np.int32)
    for i, track in enumerate(scenario.tracks):
        agents[:, :, i] = _track_to_array(track, T)
        agent_types[i]  = track.object_type
        agent_ids[i]    = track.id
    return agents, agent_types, agent_ids


def _parse_map(scenario) -> dict:
    lanes, road_lines, road_edges = [], [], []
    stop_signs, crosswalks, speed_bumps, driveways = [], [], [], []

    for feat in scenario.map_features:
        fid = feat.id
        if feat.HasField("lane"):
            ln = feat.lane
            lanes.append({
                "id": fid,
                "type": int(ln.type),
                "interpolating": bool(ln.interpolating),
                "speed_limit_mph": float(ln.speed_limit_mph),
                "polyline": _polyline_to_array(ln.polyline),
                "entry_lanes": list(ln.entry_lanes),
                "exit_lanes": list(ln.exit_lanes),
            })
        elif feat.HasField("road_line"):
            road_lines.append({
                "id": fid,
                "type": int(feat.road_line.type),
                "polyline": _polyline_to_array(feat.road_line.polyline),
            })
        elif feat.HasField("road_edge"):
            road_edges.append({
                "id": fid,
                "type": int(feat.road_edge.type),
                "polyline": _polyline_to_array(feat.road_edge.polyline),
            })
        elif feat.HasField("stop_sign"):
            ss = feat.stop_sign
            stop_signs.append({
                "id": fid,
                "position": np.array([ss.position.x, ss.position.y, ss.position.z], dtype=np.float32),
                "lane_ids": list(ss.lane),
            })
        elif feat.HasField("crosswalk"):
            crosswalks.append({"id": fid, "polygon": _polyline_to_array(feat.crosswalk.polygon)})
        elif feat.HasField("speed_bump"):
            speed_bumps.append({"id": fid, "polygon": _polyline_to_array(feat.speed_bump.polygon)})
        elif feat.HasField("driveway"):
            driveways.append({"id": fid, "polygon": _polyline_to_array(feat.driveway.polygon)})

    return dict(lanes=lanes, road_lines=road_lines, road_edges=road_edges,
                stop_signs=stop_signs, crosswalks=crosswalks,
                speed_bumps=speed_bumps, driveways=driveways)


def _parse_traffic_lights(scenario) -> list:
    """Returns list of T entries, each a list of active signal states at that timestep."""
    result = []
    for dms in scenario.dynamic_map_states:
        states = []
        for ls in dms.lane_states:
            states.append({
                "lane_id": int(ls.lane),
                "state": int(ls.state),
                "stop_point": np.array([ls.stop_point.x, ls.stop_point.y, ls.stop_point.z], dtype=np.float32),
            })
        result.append(states)
    return result


def parse_scenario(scenario: scenario_pb2.Scenario) -> dict:
    """
    Parse all fields from a Scenario proto into a flat dict of numpy arrays
    and Python scalars/lists. No PyTorch dependency.

    Keys
    ----
    scenario_id        : str
    current_time_index : int          — splits history (<=) from future (>)
    timestamps         : (T,)         — seconds
    sdc_track_index    : int          — which agent column is the ego

    agents             : (T, F, N)    — all agents; ego at sdc_track_index
    agent_types        : (N,) int32   — 1=vehicle 2=pedestrian 3=cyclist 4=other
    agent_ids          : (N,) int32   — track IDs from the proto

    tracks_to_predict  : list[dict]   — {track_index, difficulty}
    objects_of_interest: list[int]

    lanes              : list[dict]   — {id, type, interpolating, speed_limit_mph,
                                         polyline (P,3), entry_lanes, exit_lanes}
    road_lines         : list[dict]   — {id, type, polyline (P,3)}
    road_edges         : list[dict]   — {id, type, polyline (P,3)}
    stop_signs         : list[dict]   — {id, position (3,), lane_ids}
    crosswalks         : list[dict]   — {id, polygon (P,3)}
    speed_bumps        : list[dict]   — {id, polygon (P,3)}
    driveways          : list[dict]   — {id, polygon (P,3)}

    traffic_lights     : list[list[dict]] — [timestep][signal]{lane_id, state, stop_point (3,)}
    """
    T = len(scenario.timestamps_seconds)
    agents, agent_types, agent_ids = _parse_agents(scenario, T)

    return {
        "scenario_id":         scenario.scenario_id,
        "current_time_index":  scenario.current_time_index,
        "timestamps":          np.array(list(scenario.timestamps_seconds), dtype=np.float32),
        "sdc_track_index":     scenario.sdc_track_index,
        "agents":              agents,
        "agent_types":         agent_types,
        "agent_ids":           agent_ids,
        "tracks_to_predict":   [{"track_index": r.track_index, "difficulty": r.difficulty}
                                 for r in scenario.tracks_to_predict],
        "objects_of_interest": list(scenario.objects_of_interest),
        **_parse_map(scenario),
        "traffic_lights":      _parse_traffic_lights(scenario),
    }


def load_scenario(scenario: scenario_pb2.Scenario):
    """
    Parse one Scenario proto into ego and surrounding vehicle arrays.

    Returns
    -------
    ego_vehicle : np.ndarray, shape (T, F)
    sur_vehicle : np.ndarray, shape (T, F, N)  — third dim indexes vehicles
    """
    T = len(scenario.timestamps_seconds)

    ego_track = scenario.tracks[scenario.sdc_track_index]
    ego_vehicle = _track_to_array(ego_track, T)            # (T, F)

    sur_tracks = [
        t for i, t in enumerate(scenario.tracks)
        if i != scenario.sdc_track_index
    ]

    if sur_tracks:
        sur_arrays = [_track_to_array(t, T) for t in sur_tracks]  # list of (T, F)
        sur_vehicle = np.stack(sur_arrays, axis=2)                 # (T, F, N)
    else:
        sur_vehicle = np.empty((T, len(FEATURE_NAMES), 0), dtype=np.float32)

    return ego_vehicle, sur_vehicle


def load_train(
    data_dir: str = "data/raw/waymo_motion/scenario/train",
    max_files: Optional[int] = None,
    max_scenarios: Optional[int] = None,
):
    """
    Load all training scenarios and return stacked arrays.

    Parameters
    ----------
    data_dir       : path to the train TFRecord directory
    max_files      : optional limit on how many shard files to read
    max_scenarios  : optional early-stop on total scenarios parsed

    Returns
    -------
    ego_vehicles : np.ndarray, shape (S, T, F)
    sur_vehicles : list[np.ndarray]  — length S, each element (T, F, N_s)
                   N_s may differ per scenario
    """
    pattern = os.path.join(data_dir, "*.tfrecord*")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No TFRecord files found in {data_dir!r}")
    if max_files is not None:
        files = files[:max_files]

    ego_list = []
    sur_list = []
    count = 0

    for fpath in files:
        for raw in _iter_tfrecord(fpath):
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(raw)
            ego, sur = load_scenario(scenario)
            ego_list.append(ego)
            sur_list.append(sur)
            count += 1
            if max_scenarios is not None and count >= max_scenarios:
                return np.stack(ego_list, axis=0), sur_list

    return np.stack(ego_list, axis=0), sur_list
