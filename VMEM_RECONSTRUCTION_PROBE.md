# Fixed-Input Reconstruction Probe

## Completed Math Probe, 2026-09-23

The user supplied both within-process reports and the cross-process comparison
for `outputs/vmem_reconstruction_math_v1/a` and `b`. With two reconstructions
per process, all 40 preprocessing arrays, all 30 prediction arrays and all 19
aligned-output arrays match exactly in every reported comparison.
`matched_control`, RNG equality, attention-record equality and unchanged
weights all pass. Ambient environments match the saved reference as well.

This is a passed fixed-five-frame math-profile reconstruction check. It does
not identify one faulty native kernel, prove determinism for all input lengths,
or establish a quality improvement. The submitted log does not include the
updated CPU test output; only the earlier 16-test pass is on record here.

The next step is the [combined-attention generation check](VMEM_GENERATION_DEBUG.md#combined-attention-generation-check).
The generator now has an opt-in `--cut3r-attention math` path with the same
inference-only scope. Its default remains native, and nonnative CLI use is
restricted to short fresh debug runs. Do not repeat the standalone commands
below just because they remain in the runbook; they document completed work.

## Completed Native Probe, 2026-09-23

The user ran all 16 original probe tests successfully on CECSL, then ran two
fresh probe processes under `outputs/vmem_reconstruction_post_eviction_v1/a`
and `b`, each reconstructing the same five saved frames twice. The reference
was `outputs/vmem_debug_post_eviction_v1/` followed by
`post_eviction_unbounded_pan_45_A12_unbounded_20260922_205407`.

The supplied comparison reports `matched_control: true`: input records,
settings, loaded weights, recorded sources/environments and RNG states match.
Both ambient environments match the reference run and loaded weights remain
unchanged. Within each process and across both processes:

| Stage | Outcome |
|---|---|
| Preprocessed views | All 40 arrays match |
| Raw CUT3R predictions | 24 of 30 arrays differ |
| Aligned reconstruction | 16 of 19 arrays differ |

The first differing prediction field in traversal order is view 1's
`camera_pose` (seven float32 values). Across processes its maximum absolute
difference is about 3.49e-6 for repeat 0 and 1.41e-5 for repeat 1. The first
reported aligned difference is focal length, with maxima about 0.0895 and
0.1210 in that field's units. These are not maxima over all prediction arrays,
video errors, quality metrics, or proof of the first internal layer to diverge.

This reproduces variability in raw CUT3R inference on fixed inputs, before
alignment or surfel construction. No GeoCov scoring/eviction runs in the probe.
It does not identify a specific kernel, establish that alignment is itself
repeatable with fixed predictions, or explain the direction of the old VBench
gap. The generation-level frame-21 divergence remains a separate observation.

## Math-Attention Control

The probe now accepts `--attention native|math`, with `native` unchanged by
default. The math control reuses the scoped attention helper used by the CLIP
probe, but applies it to the CUT3R inference call only: it disables the MHA
fast path and allows only math SDPA. CUT3R self/cross-attention directly calls
SDPA in `extern/CUT3R/src/dust3r/blocks.py`. This control does not establish that
either fast path is defective. It does not change dtype, autocast, TF32,
weights, poses, alignment settings, GeoCov, or the production generation path.

Each repetition records attention flags before/during/after inference. Backend
state is restored before prediction capture and alignment, even on inference
failure. New reports are rejected if these records are missing, disagree with
the requested profile, or show unrestored flags. Comparison includes the
profile/flags; old native reports remain readable. Source hashes now include
the shared attention helper. `environment_matches_reference` describes ambient
state, not equality of the deliberately overridden inference dispatch.

The following commands document the completed math probe. The previous
16-test pass predates its new control tests; no new tests or experiments have
been run on the Mac. The next full-generation check is linked above.

```bash
conda activate vmem
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -p 'test_reconstruction_probe.py' -v
nvidia-smi
```

The user ran two fresh math-profile processes. The native results above were
preserved; no existing output was overwritten.

```bash
GPU=1
REF=outputs/vmem_debug_post_eviction_v1/post_eviction_unbounded_pan_45_A12_unbounded_20260922_205407
OUT=outputs/vmem_reconstruction_math_v1
for REP in a b; do
  CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/probe_vmem_reconstruction.py run \
    --reference-run "$REF" --attention math --output "$OUT/$REP" || break
done
python scripts/probe_vmem_reconstruction.py compare \
  --left "$OUT/a" --right "$OUT/b" | tee "$OUT/comparison.json"
```

Inspect both within-process reports and the cross-process comparison. A stable
math probe would support this execution profile on these inputs, not universal
determinism or improved video quality. If predictions match but alignment still
varies, distinguish those outcomes. If predictions still vary, attention alone
has not resolved the issue; do not change pruning thresholds to hide it.
Directly comparing native and math probes is an intentional settings mismatch,
not a same-settings repeatability test. The existing native reports also have
older probe source hashes. Full-generation validation and versioned production
integration remain necessary before claiming a corrected long-video comparison.

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

These are the original native-probe commands; use the math-control commands
above for the next check. No tests or experiments were run on the Mac. The
user's 16-test CECSL pass applies to the original implementation:

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

The completed budget-crossing control verifies isolated diffusion noise but
still diverges in context/output before eviction; see
[the recorded result](VMEM_GENERATION_DEBUG.md#budget-crossing-result-2026-09-23).
No 60-second rerun, new benchmark protocol or quality-improvement claim follows
automatically from a stable short probe.
