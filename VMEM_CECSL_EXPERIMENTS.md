# VMem CECSL Transfer and Scaling Runbook

Status: implementation prepared, not executed. The Mac is for code only. The
user pushes/pulls and runs every command below on CECSL. No remote jobs were
launched or terminated by this update. Do not start the full suite before the
pilot is validated.

## 1. Check the Code First

After pulling the final code, from the VMem repository in the existing `vmem`
environment, please run and return the result:

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
```

This is a CPU unit-test request, not generation. New tests cover backing-storage
deduplication, timing boundaries with a fake CUDA interface, frozen settings,
commanded paths, budget/protected-frame checks and failure classification. They
do not substitute for a real model-loading and instrumented generation check.
No new tests or dry runs have been run on the Mac.

The analysis commands also need `ffprobe`, OmegaConf and matplotlib in this
environment. Do not install/change the environment between paired arms without
recording and rechecking the pair.

## 2. Freeze the Unchanged Transfer Manifest

Only after code checks are satisfactory:

```bash
python scripts/audit_vmem_runs.py freeze manifests/vmem_transfer_v1.jsonl \
  --output outputs/locks/transfer_v1_resources.json
```

This does not load models. It checks paired settings, constructs endpoint poses,
checks that paths return, and records source/config/image/manifest hashes plus
the planned paths. It refuses to overwrite an existing lock. Keep the lock,
inputs and source config with the results. All commands assume the repo root as
working directory. A changed generation source/config/input requires an explicit
new lock and rerunning affected matched pairs, not silently replacing one arm.

The named core source files are hashed, not the entire dependency tree. After
loading, each run additionally hashes the VMem and CUT3R checkpoint files; the
inventory compares those across arms. VAE/CLIP checkpoint files and all package
versions are not byte-locked by this tool. Preserve the existing checkpoint cache,
environment export and GPU/driver details with the experiment record.

## 3. Oxford Matched Pilot

Read a fresh `nvidia-smi` and choose the device yourself. These commands use
`GPU=1` as an example, not an instruction to use a busy device. One process runs
at a time. GPU selection neither reserves VRAM nor protects other jobs against
contention; no OOM immunity is promised. Avoid concurrent workloads on the
selected GPU when interpreting latency. Record any unavoidable contention.

```bash
nvidia-smi
GPU=1
for job in 0 1; do
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_vmem_demo_manifest.py \
    manifests/vmem_transfer_v1.jsonl --job-index "$job" \
    --experiment-lock outputs/locks/transfer_v1_resources.json \
    --output-root outputs/vmem_transfer_v1_resources || break
done
```

These are the final-suite Oxford `pan_45` cases, 60 seconds per arm, not
throwaway samples. Expected per video: 195 actions, 781 decoded frames at 13 fps,
576x576, approximately 60.08 seconds. GeoCov counts both anchor and latest frame
toward B=32. Do not change denoising, reconstruction or trajectories in one arm.

## 4. Validate, Then Expand

```bash
python scripts/audit_vmem_runs.py inventory \
  --lock outputs/locks/transfer_v1_resources.json \
  --output-root outputs/vmem_transfer_v1_resources \
  --output outputs/vmem_transfer_v1_inventory.json
python scripts/plot_vmem_resources.py \
  --inventory outputs/vmem_transfer_v1_inventory.json \
  --output outputs/vmem_transfer_v1_analysis
```

After the pilot, expect **1/15 validated pairs**, with other cases explicitly
missing. Inspect `attempts` as well as `pairs`; the tool's exit code alone does
not mean the suite is complete. It checks lock/spec/metadata consistency,
effective configuration, VMem/CUT3R hashes, action/pose/frame counts, per-step
resource completeness, eligibility/descriptor logging, retrieval eligibility,
and decoded video count/resolution/fps/duration via `ffprobe -count_frames`.

The inventory preserves missing, failed, unfinished and invalid attempts. It
uses the most recently completed validated attempt per arm, never a quality
score, and rejects pairs with different checkpoint hashes or recorded Torch/CUDA
versions. Retain all retries and explain why they occurred. A model/import
failure before a run directory/spec exists remains a launcher failure in the
terminal log, not a validated run. Keep terminal logs too.

Watch both videos and check common operational failures. Poor but completed
generations remain in the record; do not pick only favorable scenes or policies.
An OOM is a resource failure, not a video-quality win. Review the inventory and
pilot curves before requesting the rest:

```bash
for job in {2..29}; do
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_vmem_demo_manifest.py \
    manifests/vmem_transfer_v1.jsonl --job-index "$job" \
    --experiment-lock outputs/locks/transfer_v1_resources.json \
    --output-root outputs/vmem_transfer_v1_resources || break
done
```

Then rerun inventory/plot commands. There is no automatic resume or skip of
completed runs; `--all` would rerun the pilot. Use explicit missing row indices
for recovery. Do not update generation code mid-suite and mix the resulting arms.

## 5. Separate Exploration Scaling Pilot

The original 15 cases remain unchanged. New versioned manifests are independent:

| Manifest | Cases | Duration | Paths and intent |
|---|---|---|---|
| `manifests/vmem_scaling_v1_pilot.jsonl` | Oxford, two paired paths, four videos | 30 seconds, 98 actions, 393 frames | Small predefined operational pilot |
| `manifests/vmem_scaling_v1_extended.jsonl` | Same two paired paths, four videos | 180 seconds, 585 actions, 2341 frames | Optional long extension after pilot review |

`fixed_region_v1` repeats eight forward/eight backward actions. Step size 0.02
gives a maximum radius of 0.16 in Navigator units. `expanding_excursions_v1`
uses 8 forward/8 backward, then 16/16, then 24/24, and so on. Its analytic maximum
radius is 0.48 in the 30-second prefix and 1.28 at 180 seconds. Return cycles are
completed repeatedly, although the duration cutoff may end mid-excursion. Both
arms use seed 701 and identical commands; each long path begins with its pilot
prefix. Same seed is not a promise of bitwise numerical reproducibility.

These are **commanded excursions**, not proof of new textures or reconstructed
coverage. Inspect actual videos separately. A one-dimensional excursion is a
controlled scaling probe, not a general open-world exploration benchmark.

Before any scaling generation, freeze both horizons and inspect path plots:

```bash
python scripts/audit_vmem_runs.py freeze manifests/vmem_scaling_v1_pilot.jsonl \
  --output outputs/locks/scaling_v1_pilot.json
python scripts/audit_vmem_runs.py freeze manifests/vmem_scaling_v1_extended.jsonl \
  --output outputs/locks/scaling_v1_extended.json
python scripts/plot_vmem_resources.py --lock outputs/locks/scaling_v1_pilot.json \
  --output outputs/scaling_v1_pilot_paths
python scripts/plot_vmem_resources.py --lock outputs/locks/scaling_v1_extended.json \
  --output outputs/scaling_v1_extended_paths
```

Then, when ready for this separate pilot:

```bash
for job in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_vmem_demo_manifest.py \
    manifests/vmem_scaling_v1_pilot.jsonl --job-index "$job" \
    --experiment-lock outputs/locks/scaling_v1_pilot.json \
    --output-root outputs/vmem_scaling_v1_pilot || break
done
python scripts/audit_vmem_runs.py inventory \
  --lock outputs/locks/scaling_v1_pilot.json \
  --output-root outputs/vmem_scaling_v1_pilot \
  --output outputs/vmem_scaling_v1_pilot_inventory.json
python scripts/plot_vmem_resources.py \
  --inventory outputs/vmem_scaling_v1_pilot_inventory.json \
  --output outputs/vmem_scaling_v1_pilot_analysis --checkpoints 10 20 30
```

The 180-second extension is not requested for launch yet. After shared plumbing
and path review, use the extended manifest/lock and a distinct output root.
Proceed or revise based on operational validity, not which policy wins.

## 6. Reading the Artifacts

Each successful run preserves the full MP4, spec, effective config, commanded
path, actual actions, retrieval/memory logs and raw `resource_trace.jsonl`.
`run_status.json` distinguishes normal completion from caught failure; abrupt
termination can leave `running`, which inventory calls `unfinished`. Earlier
completed resource/action JSONL records survive a later ordinary failure.

Analysis produces memory-versus-duration and retrieval-latency-versus-duration
plots for validated pairs, per-step CSVs including all four stage timings and
component byte estimates, and checkpoint CSVs. Checkpoints use the first update
at or after the requested duration and record its actual time, without invented
interpolation. Warm-up points are separate. See the
[ownership/measurement contract](VMEM_MEMORY_OWNERSHIP.md) before interpreting
the curves. No plot is manufactured for missing data. Optional
`--include-incomplete` draws provenance-verified partial runs as incomplete,
never as completed quality comparisons.

The user's later quality evaluation should preserve the matched original
resolution/fps and report revisit behavior separately from outward exploration.
The official VBench-Long `long_custom_input` path documents six supported
dimensions and scene/clip preprocessing. This makes it a candidate, not a
validated protocol for these particular 13-fps rollouts. First inspect split
counts, temporal coverage and matched preprocessing on the pilot. Do not call
this the full standard prompt-based benchmark. [Official VBench-Long README](https://github.com/Vchitect/VBench/blob/master/vbench2_beta_long/README.md)

Camera movement legitimately changes appearance, while static/collapsed videos
can score well on some consistency measures. Interpret consistency alongside
motion and visual quality. There is no exact-index GT for newly imagined views;
initial-view comparisons are commanded-revisit diagnostics, not GT novel-view
LPIPS or reconstruction fidelity. A CUT3R camera/geometry evaluation needs its
own sanity validation before being reported.

## Deliverable Status

| Deliverable | Current state |
|---|---|
| Ownership audit and precise bound | Written; static inspection only |
| Matched-run inventory, including failures | Validator implemented; CECSL run data pending |
| Memory and retrieval-latency curves | Plotter implemented; measured curves pending |
| Separate fixed-region/excursion protocol | Versioned manifests and path checks prepared; execution pending |
| Paired quality/revisit/exploration results | Pending generation and separate user evaluation |
| Supported claims and limitations | In ownership audit and MemCam handoff; no improvement claim yet |
