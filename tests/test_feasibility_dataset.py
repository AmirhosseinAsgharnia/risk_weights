"""
Tests for durable dataset storage (learning.feasibility_dataset) -- JSONL
append/read, atomic per-scenario result files, side isolation, and
tolerance of a truncated final line (the one thing a crash mid-append can
leave behind, given single-writer appends -- see that module's docstring).

Run with: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_feasibility_dataset.py -v
"""

from pathlib import Path

import pytest

from learning.feasibility_dataset import (
    append_scenario_record, append_rollout_records, read_jsonl,
    scenarios_path, rollouts_path, scenario_result_exists, write_scenario_result, read_scenario_result,
)


def test_append_and_read_scenario_records(tmp_path):
    append_scenario_record("cutin", 1, {"scenario_id": "a", "k": 3, "n": 5}, root=tmp_path)
    append_scenario_record("cutin", 1, {"scenario_id": "b", "k": 1, "n": 5}, root=tmp_path)
    records = read_jsonl(scenarios_path("cutin", 1, root=tmp_path))
    assert [r["scenario_id"] for r in records] == ["a", "b"]


def test_append_rollout_records_one_line_per_record(tmp_path):
    rows = [{"scenario_id": "a", "episode": i, "y": i % 2} for i in range(5)]
    append_rollout_records("cutin", 1, rows, root=tmp_path)
    records = read_jsonl(rollouts_path("cutin", 1, root=tmp_path))
    assert len(records) == 5
    assert [r["episode"] for r in records] == list(range(5))


def test_read_jsonl_missing_file_returns_empty_list(tmp_path):
    assert read_jsonl(tmp_path / "does_not_exist.jsonl") == []


def test_blocker_side_and_family_are_isolated(tmp_path):
    append_scenario_record("cutin", 1, {"scenario_id": "a"}, root=tmp_path)
    assert read_jsonl(scenarios_path("cutin", -1, root=tmp_path)) == []
    assert read_jsonl(scenarios_path("sandwich", 1, root=tmp_path)) == []
    assert len(read_jsonl(scenarios_path("cutin", 1, root=tmp_path))) == 1


def test_read_jsonl_tolerates_truncated_final_line(tmp_path):
    path = scenarios_path("cutin", 1, root=tmp_path)
    append_scenario_record("cutin", 1, {"scenario_id": "a"}, root=tmp_path)
    append_scenario_record("cutin", 1, {"scenario_id": "b"}, root=tmp_path)
    with open(path, "a") as f:
        f.write('{"scenario_id": "c", "truncated": tr')   # no closing, no newline
    records = read_jsonl(path)
    assert [r["scenario_id"] for r in records] == ["a", "b"]


def test_read_jsonl_raises_on_malformed_non_final_line(tmp_path):
    path = scenarios_path("cutin", 1, root=tmp_path)
    path.parent.mkdir(parents=True)
    with open(path, "w") as f:
        f.write("not valid json\n")
        f.write('{"scenario_id": "a"}\n')
    with pytest.raises(ValueError):
        read_jsonl(path)


def test_scenario_result_round_trip(tmp_path):
    assert scenario_result_exists("cutin", 1, "abc", root=tmp_path) is False
    write_scenario_result("cutin", 1, "abc", {"scenario_id": "abc", "k": 4, "n": 10}, root=tmp_path)
    assert scenario_result_exists("cutin", 1, "abc", root=tmp_path) is True
    assert read_scenario_result("cutin", 1, "abc", root=tmp_path) == {"scenario_id": "abc", "k": 4, "n": 10}


def test_scenario_result_isolated_per_scenario_id(tmp_path):
    write_scenario_result("cutin", 1, "abc", {"scenario_id": "abc"}, root=tmp_path)
    assert scenario_result_exists("cutin", 1, "xyz", root=tmp_path) is False
    assert read_scenario_result("cutin", 1, "xyz", root=tmp_path) is None


def test_scenario_result_overwrite_is_clean(tmp_path):
    write_scenario_result("cutin", 1, "abc", {"scenario_id": "abc", "k": 1}, root=tmp_path)
    write_scenario_result("cutin", 1, "abc", {"scenario_id": "abc", "k": 2}, root=tmp_path)
    assert read_scenario_result("cutin", 1, "abc", root=tmp_path) == {"scenario_id": "abc", "k": 2}
