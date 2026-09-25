# One-Command CECSL Results

For the new, separately gated H100/Slurm path see [VMEM_NEWTON.md](VMEM_NEWTON.md).
Never use this page's detached default or physical GPU selection inside Slurm.
The adapter changes orchestration only; CECSL generation checkpoints retain their
source identity. A previously frozen *workflow report* spec may require a new
`--output` after the orchestration changes; keep its generation root/lock intact.

## Run

After pushing these new files and pulling on CECSL, from `~/vmem`:

```bash
bash scripts/run_vmem_results.sh --gpu 1
```

This detaches a logged controller. No tmux session or separate conda activation
is required. It uses `$HOME/miniconda3/envs/vmem/bin/python` for generation and
the sibling `vbench` environment for quality. Override the first with
`VMEM_PYTHON=/path/to/vmem/bin/python`, or the second with `--vbench-python`.
It never installs packages or changes either environment.

The default is the requested **Oxford pan_45, 60-second unbounded/GeoCov-32
pair**, not a new short diagnostic and not all 30 videos. It uses the existing
`outputs/locks/transfer_v3_controlled.json` and generation directory
`outputs/vmem_transfer_v3_controlled`. It resumes the most recently committed
compatible checkpoint into a new attempt, then runs the other arm. The source
attempt is preserved. A validated completed video is reused, not regenerated.
If an existing attempt has no checkpoint, fails provenance, is already running,
or claims completion but fails validation, it stops for review rather than
silently starting over. An interrupted export can resume from the final action
checkpoint. A partially written checkpoint is never selected.

The 2026-09-23 log showed 95 actions completed, but the last atomic checkpoint
may be earlier. Recovery uses the checkpoint, not that action-log count.
Generation source files have not changed in this automation update, preserving
v3 recovery compatibility. Model/runtime identity is still checked by recovery.

## What Runs

1. Verify the original source/config/image lock, run all CPU tests in `vmem`,
   check report dependencies, and run the VBench CPU import/video-writer check.
   This does not preload/download every scoring weight; later metric loading
   can still fail and is reported honestly.
2. Wait until the selected GPU has no compute processes, <=5% utilization, and
   at least 85,000 MiB free. Only that GPU is exposed to generation/evaluation.
   No existing job is killed. This admission check is not a VRAM reservation or
   an OOM guarantee, especially for unbounded memory. No automatic settings
   downgrade, reconstruction window, precision change or shortened video occurs.
3. Resume/finish unbounded and generate GeoCov-32 sequentially. Preserve every
   attempt. Validate decoded video length/format, provenance, bank/payload
   limits and the pre-eviction pixel/retrieval pairing check.
4. Evaluate aesthetic quality, imaging quality, subject consistency, background
   consistency, motion smoothness and dynamic degree with the existing
   VBench-Long adapter and environment. Each dimension gets its own preserved
   attempt. A metric failure does not discard other dimensions.
5. Score saved PNGs at commanded returns using RGB MSE and anchor availability/
   selection. No GT LPIPS/FVD or unvalidated CUT3R accuracy is claimed.
6. Refresh the HTML report after each stage/metric, then package it as a ZIP.
   It includes both videos, all six score rows and signed deltas, resource
   curves/tables, per-update component/stage CSVs, revisit scores, raw score
   records, provenance, logs, and failed/incomplete attempts. Missing data are
   shown as missing, not zeros; lower GeoCov scores remain visible.

The same command can be run again after a failure: it resumes generation and
reuses only verified completed metric pairs with matching inputs, evaluator
sources and scoring environment. Failed metric attempts remain; only their
dimensions are retried in fresh directories. There is no automatic retry loop.

## Progress, Cancel, Download

```bash
bash scripts/run_vmem_results.sh status
bash scripts/run_vmem_results.sh stop
```

`stop` signals the controller, which terminates only its active child process
group. It does not touch unrelated GPU jobs. PNGs/committed checkpoints remain.
The status file distinguishes running, waiting, complete, incomplete, failed
and cancelled. Inspect the printed controller/stage logs for details.

Report: `outputs/vmem_results_oxford60/report/index.html`.
Final bundle: `outputs/vmem_results_oxford60/results.zip`.
On the Mac, after the bundle is ready:

```bash
scp ab575577@10.171.42.25:~/vmem/outputs/vmem_results_oxford60/results.zip ~/Downloads/vmem_oxford60_results.zip
```

Unzip and open `report/index.html`; videos/plots are local assets, with no web
server or CDN required. A failed/cancelled workflow also produces a partial
bundle, so inspect its status rather than treating file existence as success.
`bash scripts/run_vmem_results.sh report` rebuilds the report while no controller
owns it. Rebuilding verifies the recorded metric artifacts; it does not launch
generation or scoring.

For the unchanged full 15-case v3 suite later, use the same command with
`--all-cases`. It reuses valid Oxford outputs, evaluates each completed pair
before moving to the next, and writes `outputs/vmem_results_transfer60`.
Pass `--all-cases` to its status/stop commands too. Full runs are much slower;
this update does not make full-history reconstruction constant time.

## Research Scope and Checklist Handoff

This implements the VMem generation-to-results path. It is **end-to-end external
controller transfer on our audited fork**, not transplantation of VMem's policy
into MemCam/WorldMem at matched budget. It does not implement Memory Forcing,
SpMem, paper edits, dev/test splits or seed replications from the pasted checklist.
Those remain separate tasks owned by their repositories. No new parameters are
tuned here. Existing v2 negative quality results and prior failures stay intact.

The Oxford pair is one scene/path/seed, so there is no seed-variance CI or
significance test. Clips and revisit frames are not independent trials. Even
the full 15 cases span only five images. Resource plots show actual session
boundaries and exclude per-session warm-up from latency summaries. A resumed
process's elapsed time is not the full original rollout time; no uninterrupted
speedup ratio is inferred. CUDA values cover this process's PyTorch allocator,
not other jobs. Total RAM/VRAM, surfels, metadata and disk output remain outside
the resident frame-count cap. Continued exploration and camera-accuracy checks
remain separate from these constrained revisits.

Implementation was statically reviewed on the Mac; tests and the full
orchestration have not been executed there. The single command runs the CPU
suite on CECSL before expensive work. Actual quality results are pending.
