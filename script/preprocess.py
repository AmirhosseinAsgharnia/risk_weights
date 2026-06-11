"""
Reads one shard and saves each scenario as a .pt file under data/processd/train/.

Each .pt file is a dict. Numpy arrays are converted to torch tensors.
Variable-length map structures (lanes, road_lines, etc.) are lists of dicts
containing tensors — torch.save handles them via pickle.
"""

import glob
import os

import numpy as np
import torch
from waymo_open_dataset.protos import scenario_pb2

from src.dataloader.dataloader import _iter_tfrecord, parse_scenario

DATA_DIR = "data/raw/waymo_motion/scenario/train"
OUT_DIR = "data/processd/train"

os.makedirs(OUT_DIR, exist_ok=True)


def _to_tensor(obj):
    """Recursively convert numpy arrays to torch tensors in nested structures."""
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, dict):
        return {k: _to_tensor(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_tensor(v) for v in obj]
    return obj


files = sorted(glob.glob(os.path.join(DATA_DIR, "*.tfrecord*")))
if not files:
    raise FileNotFoundError(f"No TFRecord files found in {DATA_DIR!r}")

shard_path = files[0]
print(f"Shard: {shard_path}")

count = 0
for raw in _iter_tfrecord(shard_path):
    scenario = scenario_pb2.Scenario()  # type: ignore[attr-defined]
    scenario.ParseFromString(raw)

    parsed = parse_scenario(scenario)
    scenario_id: str = parsed["scenario_id"]  # type: ignore[assignment]
    data = _to_tensor(parsed)

    out_path = os.path.join(OUT_DIR, f"{scenario_id}.pt")
    torch.save(data, out_path)
    count += 1

    if count % 100 == 0:
        print(f"  {count} scenarios saved...")

print(f"Done. {count} scenarios saved to {OUT_DIR}/")
