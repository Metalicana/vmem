# VMem Budgeted Memory Runs

Current transfer/scaling work uses CECSL only. Follow the
[instrumented CECSL runbook](VMEM_CECSL_EXPERIMENTS.md) for source locks, matched
pilots and validation. The Mac is code-only; new instrumentation has not been
executed. General inference and legacy Newton examples below remain available,
but are not the current locked experiment protocol.

VMem now supports the same fixed-frame-budget policy family used in the MemCam
and WorldMem experiments:

```text
unbounded
fifo
rarity_irreplaceability
slam_covisibility
```

The budget unit is a generated/conditioning frame. The default runner scope is
`surfel_indexed_view_memory`: retained frame ids define the memory bank, VMem's
surfel-vote retrieval selects context frames from that bank, and evicted frame
ids are pruned from `surfel_to_timestep` so the surfel index does not act as
hidden unbounded memory. For an ablation matching the earlier lightweight hook,
set `MEMORY_SCOPE=view_context`.

## Normal Single-Image Inference

For a quick VMem run without any dataset, use one input image and a scripted
camera path:

```bash
conda activate vmem

python scripts/run_vmem_image_trajectory.py \
  --image test_samples/living_room.jpg \
  --trajectory arc \
  --num-frames 17 \
  --output-root outputs/manual_vmem
```

That writes:

```text
outputs/manual_vmem/<run_name>/generated.mp4
outputs/manual_vmem/<run_name>/metadata.json
outputs/manual_vmem/<run_name>/trajectory.json
outputs/manual_vmem/<run_name>/retrieval_trace.json
```

Cheap plumbing smoke test:

```bash
python scripts/run_vmem_image_trajectory.py \
  --image test_samples/living_room.jpg \
  --trajectory arc \
  --num-frames 9 \
  --inference-steps 2 \
  --surfel-niter 5 \
  --output-root outputs/manual_vmem
```

Available trajectories:

```text
forward
arc
out_and_back
square_loop
```

For longer runs, prefer chaining the same small actions used by the Gradio demo:

```bash
python scripts/run_vmem_demo_actions.py \
  --image test_samples/open_door.jpg \
  --duration-seconds 5 \
  --pattern forward \
  --output-root outputs/demo_actions
```

The Gradio demo uses `4` generated frames per action at `13` fps, so a long run
is many small actions:

```text
30 seconds  -> about 98 actions
180 seconds -> about 585 actions
```

Named trajectory presets are available for deterministic long-horizon runs:

```text
pattern       repeat --pattern exactly
forward       forward-only exploration
out_and_back  repeated forward/backward revisit path
square_walk   forward sides plus 90 degree right turns
pan_180       left-right 180 degree sweep
spin_360      repeated 360 degree rotation
random_walk   seeded random walk over demo actions
```

Examples:

```bash
# Gentle forward exploration.
python scripts/run_vmem_demo_actions.py \
  --image test_samples/open_door.jpg \
  --duration-seconds 30 \
  --pattern forward \
  --output-root outputs/demo_actions

# Small loop-like motion using demo-size actions.
python scripts/run_vmem_demo_actions.py \
  --image test_samples/open_door.jpg \
  --num-actions 40 \
  --pattern forward,forward,left5,forward,forward,right5 \
  --output-root outputs/demo_actions
```

First unbounded 60-second sanity pass:

```bash
python scripts/run_vmem_demo_actions.py \
  --image test_samples/open_door.jpg \
  --run-id open_door_square_walk_60s \
  --trajectory square_walk \
  --duration-seconds 60 \
  --memory-policy unbounded \
  --output-root outputs/demo_actions_60s
```

The 60-second setting produces about `195` demo actions and `781` frames. To
launch the five-run unbounded smoke manifest locally or on CECSL:

```bash
python scripts/run_vmem_demo_manifest.py \
  manifests/vmem_60s_unbounded_smoke.jsonl \
  --all \
  --output-root outputs/demo_actions_60s
```

To test the manifest without loading VMem:

```bash
python scripts/run_vmem_demo_manifest.py \
  manifests/vmem_60s_unbounded_smoke.jsonl \
  --all \
  --dry-run
```

## Local or CECSL

### External Transfer Generation

The current external-backbone test compares **unbounded VMem versus VMem +
GeoCov-32**. MemCam and WorldMem carry the primary evaluation; the user will
evaluate these generated videos with VBench later. See the
[audit and MemCam handoff](VMEM_TRANSFER_AUDIT.md) for the fixed protocol,
implementation differences and limitations, and the
[CECSL runbook](VMEM_CECSL_EXPERIMENTS.md) for the full command sequence.

`manifests/vmem_transfer_v1.jsonl` contains 15 paired cases (30 videos): five
repository images and three constrained paths, each 60 seconds with matched
seeds and generation settings. After tests pass, freeze it before the Oxford pair:

```bash
python scripts/audit_vmem_runs.py freeze manifests/vmem_transfer_v1.jsonl \
  --output outputs/locks/transfer_v1_resources.json
for job in 0 1; do
  CUDA_VISIBLE_DEVICES=1 python scripts/run_vmem_demo_manifest.py \
    manifests/vmem_transfer_v1.jsonl --job-index "$job" \
    --experiment-lock outputs/locks/transfer_v1_resources.json \
    --output-root outputs/vmem_transfer_v1_resources || break
done
```

Rows 2-29 are the remaining cases. Each run saves its planned actions,
source/config/input fingerprints, effective config, video and traces. GPU 1 is
an example selection, not a reservation; use a current GPU status reading. The
runbook includes completion/provenance validation and measured resource plots;
run these before requesting rows 2-29. B=32 limits eligibility, not total RAM/VRAM.

### GeoCov Pilot

This earlier RI/GeoCov/revisit pilot is optional diagnostic infrastructure;
use the external-transfer protocol above for the supporting VMem experiment.

`manifests/vmem_geocov_pilot.jsonl` pairs unbounded VMem with
`slam_covisibility` (adapted GeoCov, B=32) and `rarity_irreplaceability` (B=32).
Rows 0/1/2 use Oxford, 3/4/5 Jesus, and 6/7/8 living room. Each group shares
its seed and the 60-second `pan_45` trajectory. This pans from 0 to +45 to
-45 and back, a 90-degree total sweep. It generates 781 frames, with ten
returns to the initial commanded pose, at frames 72, 144, ..., 720.

Start with Oxford unbounded and GeoCov; this loop runs jobs sequentially.
Select a GPU using a fresh `nvidia-smi` reading. The example selects GPU 1;
device selection is not a VRAM reservation or allocation limit.

```bash
for job in 0 1; do
  CUDA_VISIBLE_DEVICES=1 python scripts/run_vmem_demo_manifest.py \
    manifests/vmem_geocov_pilot.jsonl --job-index "$job" \
    --output-root outputs/geocov_pilot
done
```

Use indices `1 2` for GeoCov versus RI instead, or `--all` for all nine
runs sequentially. Add `--dry-run` to validate without model loading.
The pilot saves PNG frames so image errors do not depend on MP4 compression.

On Newton, the same manifest can use the existing array wrapper:

```bash
MANIFEST=manifests/vmem_geocov_pilot.jsonl \
OUTPUT_ROOT="$HOME/vmem_results/geocov_pilot" \
sbatch --array=0-1%1 slurm/newton_vmem_demo_manifest.sbatch
```

After generation, evaluate on CPU (no generation GPU needed):

```bash
python scripts/evaluate_vmem_revisits.py manifests/vmem_geocov_pilot.jsonl \
  --output-root outputs/geocov_pilot --csv-out outputs/geocov_pilot/revisits.csv
```

This writes per-run means and final-return RGB MSE, plus every return in
`revisits.returns.csv`. RGB MSE is mean squared pixel error over RGB in [0, 1].
Add `--lpips` for full-resolution AlexNet LPIPS (requires `lpips` and its
weights); CPU is the default. `--traces-only` skips image loading and metrics.
Retention and selection columns track frame 0 in the retrieval event for
each return, making retained-but-unused evidence visible. A missing trace
stays unknown; a path without an actual return is not scored as a revisit.
Old runs without seed metadata are marked `seed_verified=False`.

Interpret this as a small transfer pilot measuring commanded-pose revisit
consistency. It is not GT novel-view fidelity, FVD, or recovered 3D accuracy.
`local_loop` is excluded: forward, turn, backward, counter-turn does not
close in translation. Three scenes and one seed each do not establish a
general quality improvement; inspect all paired outcomes before expanding.

In VMem, `slam_covisibility` uses the same utility formula as the paper's
GeoCov but uses VMem image-encoder embeddings (latent fallback), not DINO.
RI uses those embeddings too. Both pin frame 0, count it toward the budget,
and protect the newest endpoint. Retrieval still uses VMem's surfel/pose
selection. Default pruning also changes the surfel index and reconstruction
history; this is an end-to-end memory intervention, not a retrieval-only
ablation. Neither strict constant total RAM/VRAM nor constant wall-clock
retrieval time follows from this implementation: output frames and latent
history are still accumulated, and the surfel count is variable.

### Context-as-Memory Runs

Run from the VMem repo after activating the VMem environment:

```bash
conda activate vmem

DATA_ROOT=/path/to/Context-as-Memory-Dataset/Context-as-Memory-Dataset \
SCENE=<scene_id> \
MEMORY_POLICY=fifo \
MEMORY_BUDGET=32 \
bash scripts/run_vmem_memory_policy_smoke.sh
```

Defaults:

```text
DURATION_SECONDS=180
FPS=13
NUM_FRAMES=DURATION_SECONDS * FPS + 1
CHUNK_SIZE=4
CAMERA_CONVENTION=unreal
OUTPUT_ROOT=/data/ab575577/vmem/outputs/memory_policy
MEMORY_SCOPE=surfel_indexed_view_memory
```

Cheap smoke test:

```bash
DATA_ROOT=/path/to/Context-as-Memory-Dataset/Context-as-Memory-Dataset \
SCENE=<scene_id> \
DURATION_SECONDS=2 \
INFERENCE_STEPS=2 \
SURFEL_NITER=5 \
SURFEL_RECONSTRUCTION_WINDOW=8 \
MEMORY_POLICY=rarity_irreplaceability \
MEMORY_BUDGET=4 \
bash scripts/run_vmem_memory_policy_smoke.sh
```

Full 180-second grid for one or more scenes:

```bash
DATA_ROOT=/path/to/Context-as-Memory-Dataset/Context-as-Memory-Dataset \
SCENES=scene_a,scene_b \
DURATIONS=180 \
POLICIES=unbounded,fifo,rarity_irreplaceability,slam_covisibility \
BUDGETS=16,32,64,128 \
bash scripts/run_vmem_memory_policy_grid.sh
```

Each run writes:

```text
generated.mp4
metadata.json
retrieval_trace.json
memory_trace.json
```

Sanity-check a run:

```bash
python scripts/summarize_vmem_context_run.py /path/to/run_dir --fail-on-violation
```

## Newton

Set the same environment variables, plus repo/env locations as needed:

```bash
VMEM_REPO_ROOT=$HOME/vmem \
CONDA_ENV=vmem \
DATA_ROOT=/path/on/newton/Context-as-Memory-Dataset/Context-as-Memory-Dataset \
SCENE=<scene_id> \
MEMORY_POLICY=slam_covisibility \
MEMORY_BUDGET=64 \
sbatch slurm/newton_vmem_memory_policy_smoke.sbatch
```

Array grid:

```bash
VMEM_REPO_ROOT=$HOME/vmem \
CONDA_ENV=vmem \
DATA_ROOT=/path/on/newton/Context-as-Memory-Dataset/Context-as-Memory-Dataset \
SCENES=scene_a,scene_b \
DURATIONS=180 \
POLICIES=unbounded,fifo,rarity_irreplaceability,slam_covisibility \
BUDGETS=16,32,64,128 \
OUTPUT_ROOT=$HOME/vmem_results/memory_policy \
sbatch slurm/newton_vmem_memory_policy_grid.sbatch
```

`slurm/newton_vmem_memory_policy_grid.sbatch` is declared as `--array=0-63`.
If the scene x duration x policy x budget grid has more than 64 jobs, increase
that array range before submitting.

Unbounded 60-second demo-action manifest:

```bash
VMEM_REPO_ROOT=$HOME/vmem \
CONDA_ENV=vmem \
OUTPUT_ROOT=$HOME/vmem_results/demo_actions_60s \
sbatch slurm/newton_vmem_demo_manifest.sbatch
```

`slurm/newton_vmem_demo_manifest.sbatch` is declared as `--array=0-4`, matching
the five rows in `manifests/vmem_60s_unbounded_smoke.jsonl`.
