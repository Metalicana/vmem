# VMem Budgeted Memory Runs

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

## Local or CECSL

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
