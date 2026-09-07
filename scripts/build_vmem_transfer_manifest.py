#!/usr/bin/env python
"""Write the fixed VMem external-transfer generation suite, without loading models."""

import argparse
import json
from pathlib import Path


SCENES = ("oxford", "jesus", "living_room", "open_door", "changi")
TRAJECTORIES = (("pan_45", 0.1), ("pan_90", 0.1), ("out_and_back", 0.02))


def build_rows():
    rows = []
    for scene_index, scene in enumerate(SCENES):
        for trajectory_index, (trajectory, step_size) in enumerate(TRAJECTORIES):
            case_id = f"transfer_v1_{scene}_{trajectory}"
            common = {
                "image": f"test_samples/{scene}.jpg", "trajectory": trajectory,
                "step_size": step_size, "duration_seconds": 60, "fps": 13,
                "frames_per_action": 4, "seed": 501 + 10 * scene_index + trajectory_index,
                "inference_steps": 50, "surfel_niter": 400,
                "memory_scope": "surfel_indexed_view_memory",
                "_case_id": case_id,
            }
            rows.append({**common, "run_id": f"{case_id}_unbounded", "memory_policy": "unbounded"})
            rows.append({**common, "run_id": f"{case_id}_geocov32",
                         "memory_policy": "slam_covisibility", "memory_budget": 32})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("manifests/vmem_transfer_v1.jsonl"))
    args = parser.parse_args()
    rows = build_rows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite an already frozen protocol; choose a new output path.
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write("# VMem external transfer v1: 15 matched cases, unbounded vs adapted GeoCov-32.\n")
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} runs to {args.output}. Rows 0-1 are the Oxford pilot.")


if __name__ == "__main__":
    main()
