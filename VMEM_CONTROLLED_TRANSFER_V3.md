# Controlled VMem Transfer V3

## Decision and Evidence

The user-supplied CECSL combined-control 12-action pair passes the short
pre-eviction comparison: all 33 saved images through frame 32 match exactly;
recorded settings, environment, contexts and geometry counts have no
pre-eviction discrepancies. First eviction is step 7, after frame 32.
Reconstruction end-state RNG first differs at step 8; conditioning and output
first differ at step 9. Initial and sampler diffusion noise still match across
all 12 actions. This is consistent with changed reconstruction inputs after
eviction, not evidence of a remaining pre-eviction policy defect.

Both videos have 49 frames. Final GeoCov RGB, latent, embedding, intrinsic and
depth resident counts are each 32 versus 49 for unbounded, with no pending
evictions. Retrieval is legal; appearance uses CLIP with zero latent fallback.
Raw geometry tensors are not hashed by this diagnostic, and no new quality
measurement is implied. Preserve the earlier negative VBench results.

## Frozen Settings

Manifest: [vmem_transfer_v3.jsonl](manifests/vmem_transfer_v3.jsonl).
The five images, three paths per image, seeds, actions, resolution, checkpoint
weights, 50 denoising steps and 400 reconstruction iterations are unchanged
from v2. Each arm generates 195 actions, 781 frames at 13 fps (60.08 seconds).
Both arms use resident frame storage. The first pair is Oxford `pan_45`.

Both arms now use:

- `rng_mode=isolated`, with `vmem_phase_action_rng_v1` phase/step seeds.
- `clip_attention=math` and `cut3r_attention=math`, scoped to those encoders.
- `checkpoint_every=5` and normal resource traces, without debug fingerprints.

`generation_rng.py` shares the seed rule with the successful diagnostic. A
reconstruction's changing random draw count cannot advance the next diffusion
stream. Recovery derives seeds from the restored global step, not a reset
counter. Locks, effective config, run metadata and checkpoint identity record
the controls and RNG schema. `generation_environment.json` records the runtime.
Legacy defaults remain unchanged. The v4 lock schema is distinct from the v3
experiment name; source hashes must be frozen on CECSL after pulling the code.

This changes numerical execution in both arms, not GeoCov's scoring, retrieval
algorithm, denoising precision or checkpoints. It is still an end-to-end
controller test: bounded reconstruction uses retained plus newly generated
frames. Do not compare a new GeoCov video against the old v2 baseline.

## CECSL Pilot

No tests, dry runs or generation were executed on the Mac. After pushing/pulling,
run the CPU suite on CECSL first. New coverage includes production/debug sampling
parity, isolated-RNG recovery, manifest equivalence and lock/config auditing.
The new non-debug CUDA path and its CUDA resume are not yet independently tested.

```bash
conda activate vmem
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
```

Only after tests pass, freeze once and check the GPU:

```bash
python scripts/audit_vmem_runs.py freeze manifests/vmem_transfer_v3.jsonl \
  --output outputs/locks/transfer_v3_controlled.json
nvidia-smi
```

The lock refuses overwrites. Do not delete or replace it to bypass a mismatch.
If GPU 1 is available, launch the two arms sequentially:

```bash
GPU=1
ROOT=outputs/vmem_transfer_v3_controlled
mkdir -p "$ROOT"
nohup env CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/run_vmem_demo_manifest.py \
  manifests/vmem_transfer_v3.jsonl --job-indices 0 1 \
  --experiment-lock outputs/locks/transfer_v3_controlled.json \
  --output-root "$ROOT" \
  > "$ROOT/controller.log" 2>&1 < /dev/null &
tail -n 20 "$ROOT/controller.log"
```

The controller prints a per-child log path. GPU selection is not a reservation;
do not launch duplicate controllers. Preserve any failed attempts. Checkpoints
use the existing rolling recovery format with stricter identity checks. To
resume, use the same launcher, manifest, lock and output root, select only the
interrupted row with `--job-index 0` or `--job-index 1`, and add
`--resume-from EXACT_INTERRUPTED_RUN_DIRECTORY`. Resume creates a new attempt
and does not overwrite the parent. Old v2/debug checkpoints are not compatible.

## After Completion

```bash
python scripts/audit_vmem_runs.py inventory \
  --lock outputs/locks/transfer_v3_controlled.json \
  --output-root outputs/vmem_transfer_v3_controlled \
  --output outputs/vmem_transfer_v3_controlled_inventory.json
python scripts/plot_vmem_resources.py \
  --inventory outputs/vmem_transfer_v3_controlled_inventory.json \
  --output outputs/vmem_transfer_v3_controlled_analysis
```

Set `A` and `B` to the exact completed unbounded and GeoCov run directories:

```bash
CUDA_VISIBLE_DEVICES="" python scripts/audit_vmem_pairing.py \
  --unbounded "$A" --bounded "$B" --compare-pixels \
  > outputs/vmem_transfer_v3_controlled/pairing_pixels.json
```

Check the pre-eviction prefix again on the normal path, not whole-video pixel
equality. There is intentionally no `generation_debug.jsonl` in a normal run.
Then evaluate both videos using the same established VBench-Long configuration,
in a new output directory. Do not launch the remaining 14 cases automatically
from a short diagnostic pass or select cases according to metric wins.

Report resource measurements from this implementation separately from v1/v2.
Resumed attempts are not uninterrupted latency comparisons. B=32 bounds
post-update resident frame payload counts, not transient reconstruction size,
total RAM/VRAM, surfel count, small histories or disk output. These paths mainly
test revisits; sustained exploration remains a separate protocol. Quality
preservation, improvement and full-suite transfer results are still pending.
