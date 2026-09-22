# Fixed-Input Reconstruction Probe

## Why This Check

The CLIP-only math control passes the supplied nine-frame generation check:
recorded generation events match across unbounded repeats and across policies,
and all nine unbounded/GeoCov decoded frames match. Reconstruction still
produces different geometry. Its first confidence-derived alignment scores
already vary, before surfel merging. See [the measured results](VMEM_CLIP_PROBE.md#completed-generation-check-2026-09-21).

Before changing another backend or running a long rollout, isolate the remaining
variation using exactly the same saved first five RGB images. This is not a
quality experiment or an eviction-policy comparison.

## Implementation and Limits

[probe_vmem_reconstruction.py](scripts/probe_vmem_reconstruction.py) loads only
the CUT3R checkpoint, verified against the reference run's recorded SHA-256.
It imports the usual generation runtime definitions to preserve import side
effects, but does not construct VMem, CLIP or VAE models. No video generation,
surfel merging, pruning, visualization, or changes to production source occur.

It checks the completed reference, first-action reconstruction frame indices,
saved RGB PNG hashes and dimensions, and replays the actual Navigator action
with generation stubbed to collect poses. It checks the recorded endpoint,
applies the pipeline's float32 pose storage and coordinate transform, and uses
the saved reconstruction learning rate/iteration count. The first reconstruction
has no prior depth maps. Original ambient no-grad/CUDA autocast behavior remains;
the reconstruction functions keep their own gradient/autocast scopes unchanged.

The script wraps existing function boundaries in its own process and restores
them on success or failure. Each repetition saves compressed, non-pickle NPZ
arrays plus shape/dtype/hash records for three stages:

1. `preprocessed`: the actual views supplied to CUT3R, including masks and shapes.
2. `predictions`: raw predicted fields, before collation and global alignment.
3. `aligned`: returned point clouds, depths, confidences and camera parameters.

Two repetitions per process use the same phase-derived Python/NumPy/Torch RNG
seed, with the original states restored afterward. This controls alignment's
random initialization; it is **not** the unavailable original post-diffusion RNG
state. The old debug trace contains RNG hashes, not recoverable states. The
standalone model-load/allocator history also differs from full VMem. Stage
capture forces CPU copies and disk I/O, so timings are not benchmark timings.
Allow disk space for the compressed dense arrays; these are not only tiny logs.

Reports include pre-load and pre-reconstruction environments, recorded source
hashes, checkpoint/loaded-parameter/buffer fingerprints, and whether loaded
weights changed during the probe. Source coverage includes vendored CUT3R Python
files, not compiled extensions or every installed dependency. A completed probe
is not necessarily repeatable: inspect comparisons. Failures leave diagnostics
and are not accepted by `compare`; existing outputs cannot be overwritten.

## CECSL Commands

No tests or experiments were run on the Mac. After pushing/pulling, first run
the new CPU tests in the generation environment:

```bash
conda activate vmem
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -p 'test_reconstruction_probe.py' -v
nvidia-smi
```

If tests pass and GPU 1 is free, run two fresh processes sequentially. Each
loads CUT3R once and reconstructs the same five frames twice. This does not
reserve VRAM or guarantee OOM immunity on a shared GPU.

```bash
GPU=1
REF=outputs/vmem_debug_clip_math_v1/debug_math_unbounded_a_pan_45_A2_unbounded_20260921_150029
OUT=outputs/vmem_reconstruction_probe_v1
for REP in a b; do
  CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/probe_vmem_reconstruction.py run \
    --reference-run "$REF" --output "$OUT/$REP" || break
done

python scripts/probe_vmem_reconstruction.py compare \
  --left "$OUT/a" --right "$OUT/b" | tee "$OUT/comparison.json"
```

Retain both run reports, `within_process.json`, RNG traces, NPZ stage artifacts
and any failures. Comparison verifies artifact hashes and individual array
fingerprints before trusting equality. Expected NaN ray-map placeholders are
handled explicitly; numeric errors use each field's native units.

## Interpretation

Require matching inputs, settings, loaded weights, recorded sources/environments
and RNG states (`matched_control: true`) before attributing differences to
execution repeatability. Inspect `environment_matches_reference` separately.
Differences there limit how closely the probe represents the original run.

- Different `preprocessed` values: investigate the input path first.
- Matching preprocessing but different `predictions`: investigate CUT3R forward
  execution. This does not by itself identify a particular kernel defect.
- Matching predictions but different `aligned` outputs: investigate collation,
  alignment and postprocessing, with recorded pre-alignment RNG states checked.
- All three stages match: the standalone probe has not reproduced the issue.
  Full-process history and surfel construction/merging remain possible locations;
  do not claim the original geometry discrepancy has been fixed.

After this investigation, the separately planned budget-crossing control still
needs isolated diffusion RNG. No 60-second rerun, new benchmark protocol or
quality-improvement claim follows automatically from a stable short probe.
