# Restarting the Oxford Pair

2026-09-08. The first unbounded attempt stopped after its last recorded action
189 (190 completed actions, 761 frames). No matching process remained. Kernel
logs were inaccessible, so the termination cause is unknown; this is not a
confirmed OOM. Keep that directory and its old experiment lock unchanged.
It predates recovery support and cannot be resumed from its JSONL traces.

## What Changed

- Every completed action now saves new frames as atomic PNG files, including
  the initial image. `--save-frames` is retained for CLI compatibility; saving
  is now unconditional in the demo-action runner.
- `recovery/latest.pt` is written at initialization, every five actions, on a
  deliberate pause, and at the final action. Atomic replacement keeps the
  previous checkpoint until the new file is committed. A failure during a
  checkpoint can require replaying five actions, plus any in-progress action.
- The state includes latents/embeddings, commanded camera state, depth/focal
  history, surfels/index, frame bank/statistics, retrieval threshold, traces,
  Navigator state and Python/NumPy/Torch CPU/selected-CUDA RNG states. Model
  weights are reloaded and their recorded hashes compared, not duplicated in
  each checkpoint. Images are stored as PNGs and hash-checked when resuming.
- A resume creates a new run directory and copies only the checkpoint's image
  and trace prefixes. Later partial work and the original failure record remain
  in the parent. It does not overwrite or delete the old attempt.
- The manifest launcher saves child stdout/stderr and PID/exit status under
  `launcher_logs`. A negative return code identifies a terminating signal, not
  its cause. A killed launcher may still leave an unfinished launcher status.
- Default checkpoint interval is five actions and is covered by new v2 locks.
  The 15-case manifest, GeoCov scores and inference settings are unchanged.

Checkpoint I/O can increase host-memory peaks and disk usage; no OOM prevention
is claimed. Budget space for PNGs, two checkpoint-sized files during replacement,
the final video and any resumed attempt's copies. Only the latest committed
checkpoint is retained within each attempt. These are trusted local pickle
files: do not pass downloaded/untrusted checkpoints to `--resume-from`.

## 1. User-Run Checks on CECSL

After pushing/pulling the changed code, in the `vmem` environment:

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
```

The earlier 38-test pass predates this recovery update. New tests have not been
run on the Mac. Do not freeze or launch the long retry if the checks fail.

Next, check a fresh `nvidia-smi` and disk/host memory (`df -h outputs`, `free -h`).
GPU 0 was free in the last supplied snapshot, but that is not a reservation.
Use the currently available device, and run one process at a time.

Run this short pause/resume plumbing test first; it is not a benchmark sample:

```bash
GPU=0
CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_vmem_demo_actions.py \
  --image test_samples/oxford.jpg --run-id recovery_smoke \
  --trajectory pan_45 --num-actions 3 --seed 501 \
  --memory-policy slam_covisibility --memory-budget 32 \
  --inference-steps 50 --surfel-niter 400 --stop-after-actions 2 \
  --output-root outputs/vmem_recovery_smoke
```

It should report `paused`, two completed actions, nine durable PNGs and a
`recovery/latest.pt`. Set `PAUSED` to the exact reported directory, then run:

```bash
CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_vmem_demo_actions.py \
  --image test_samples/oxford.jpg --run-id recovery_smoke \
  --trajectory pan_45 --num-actions 3 --seed 501 \
  --memory-policy slam_covisibility --memory-budget 32 \
  --inference-steps 50 --surfel-niter 400 --resume-from "$PAUSED" \
  --output-root outputs/vmem_recovery_smoke_resumed
```

Expect only action 3/3 to run, then a 13-frame MP4 and `complete`. Retain both
directories. Inspect the checkpoint's frame count, resumed action count and
output integrity before the long retry. CPU RNG/state tests do not establish
bitwise CUDA/video parity; that remains a separate validation question.

## 2. Freeze the Retry

After the checks succeed, use a new lock and output root. Do not overwrite the
old `transfer_v1_resources.json` lock, or mix old and new source hashes.

```bash
python scripts/audit_vmem_runs.py freeze manifests/vmem_transfer_v1.jsonl \
  --output outputs/locks/transfer_v1_recovery.json
```

Stop if freezing fails. A later generation-code change requires a new lock and
matched retry; exact-identity recovery will intentionally refuse changed code.

## 3. Restart Without Depending on Tmux

After choosing an available GPU, from the repository root with `vmem` active:

```bash
mkdir -p outputs/vmem_transfer_v1_recovery
nohup env CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/run_vmem_demo_manifest.py \
  manifests/vmem_transfer_v1.jsonl --job-indices 0 1 \
  --experiment-lock outputs/locks/transfer_v1_recovery.json \
  --output-root outputs/vmem_transfer_v1_recovery \
  > outputs/vmem_transfer_v1_recovery/controller.log 2>&1 < /dev/null &
```

`nohup` removes the dependency on keeping a tmux pane/terminal alive. It cannot
protect against OOM, reboot, SIGKILL or every site logout policy. The launcher
runs unbounded then GeoCov-32 sequentially and stops on a nonzero child exit;
it does not automatically retry failed runs or skip completed ones.

```bash
tail -n 30 outputs/vmem_transfer_v1_recovery/controller.log
```

This prints each active child PID and its exact log path. Follow that file with
`tail -f`. Status and PID alone are not evidence of continued progress; check
the action log and process together. Do not launch a duplicate on a shared GPU.

## 4. Future Recovery

Only after verifying that the old process has stopped, set `PARENT` to the
interrupted recovery-enabled directory and select its single manifest row:

```bash
CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_vmem_demo_manifest.py \
  manifests/vmem_transfer_v1.jsonl --job-index 0 --resume-from "$PARENT" \
  --experiment-lock outputs/locks/transfer_v1_recovery.json \
  --output-root outputs/vmem_transfer_v1_recovery
```

Use row 1 for GeoCov. Add the same `nohup` redirection pattern when leaving it
unattended. The last committed checkpoint, not the newest action JSON record,
determines the restart point. A checkpoint at the final action can retry export
without regenerating frames. There is no automatic resume of the original
pre-recovery run.

Even without a usable checkpoint, durable PNGs can produce a *partial diagnostic*
video. From the selected run directory, this writes a new, explicitly partial
file and refuses to overwrite an existing one:

```bash
ffmpeg -n -framerate 13 -start_number 0 -i generated_frames/%04d.png \
  -c:v libx264 -pix_fmt yuv420p partial_diagnostic.mp4
```

Do not count it as a completed 60-second result.

## 5. Validation and Resource Interpretation

```bash
python scripts/audit_vmem_runs.py inventory \
  --lock outputs/locks/transfer_v1_recovery.json \
  --output-root outputs/vmem_transfer_v1_recovery \
  --output outputs/vmem_transfer_v1_recovery_inventory.json
python scripts/plot_vmem_resources.py \
  --inventory outputs/vmem_transfer_v1_recovery_inventory.json \
  --output outputs/vmem_transfer_v1_recovery_analysis
```

Resumed videos can be structurally validated, but are labelled in the inventory.
Default resource plots exclude any pair with a resumed arm. Restoring NumPy/PIL
state can change backing-storage layout, and CUDA peaks/allocator caches restart.
Inherited and new trace segments therefore must not be interpreted as one
uninterrupted physical-memory experiment. Each segment has its own warm-up
classification; model-load/total-wall/CUDA-peak metadata cover the new process,
not summed lifetime across all attempts. PNG and checkpoint times/bytes are in
`recovery_trace.jsonl`, outside the four generation-stage timers. Their host
allocator effects may still appear in later RSS measurements.

The unknown termination cause, real-GPU resume validation, checkpoint overhead,
and quality outcomes remain unresolved. No scoring, retrieval, denoising or
reconstruction-window change was made to obtain a faster retry or policy win.
