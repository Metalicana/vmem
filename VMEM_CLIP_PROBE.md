# Initial CLIP Encoding Probe

## Observed Result

The user's CECSL comparisons of the two-action `observe` runs locate the first
recorded difference at `initial_encoding.embeddings` (step -1), both for
unbounded A versus unbounded B and for unbounded A versus GeoCov-32. Both
comparisons report seven different events, no setting/source/checkpoint-hash
mismatches among the recorded fields, no missing listed provenance, and equal
recorded environments. The initial input, initial VAE latent, poses/intrinsics,
phase CPU/CUDA RNG-state fingerprints, initial diffusion noise and all recorded
sampler-noise fingerprints agree. The later differences are conditioning `c`,
diffusion latents and decoded samples at both generation steps.

This localizes the earliest recorded discrepancy to the image-encoding path,
before reconstruction, retrieval from generated history, or eviction. It also
occurs without changing policy. It does not prove a specific kernel defect,
establish the full extent of nondeterminism, or explain the 60-second quality
gap. Matching hashes cover recorded values only; these runs did not hash all
loaded CLIP tensors or save numerical embeddings. The separate shared-RNG issue
after the budget binds remains real, but cannot explain these early differences.

`CLIPConditioner` and its underlying model are already in eval mode, and the
underlying model's parameters have gradients disabled. The path is Kornia
resize/normalization followed by OpenCLIP image encoding. Do not silently swap
the descriptor, round/cache embeddings, or modify only the GeoCov arm.

### Completed Encoder Probes, 2026-09-21

The user supplied all four CECSL probe reports (`native_a`, `native_b`, `math_a`,
`math_b`), with three forwards per process. All report matching reference
environments. Across each same-profile pair, the input, loaded parameters and
buffers, recorded sources and environment match. Preprocessing is bitwise
identical within and across the processes.

| Profile | Within-process embeddings | Across-process embeddings |
|---|---|---|
| Native | Differ on repeated forwards in both processes | Differ at all three repetitions; relative L2 0.00354-0.00591 |
| Math | Bitwise equal in both processes | Bitwise equal at all three repetitions; zero measured error |

This supports a repeatability issue associated with native attention dispatch
in the tested CLIP path. It does not identify one faulty kernel: the math
profile changes both MHA fast-path eligibility and SDPA selection. It also does
not prove the cause or direction of the 60-second GeoCov quality difference.
`matches_reference_embedding: false` is not a probe failure: that reference was
itself produced by the variable native path. The relevant comparison is
repeatability with matching inputs and weights, not agreement with that one
reference embedding.

The subsequent full-generator triplet using the same math profile symmetrically
has now completed, with the results below. Do not rerun the completed controls
or launch the long suite from this evidence alone.

### Completed Generation Check, 2026-09-21

The user supplied completion logs and audit output for these CECSL runs under
`outputs/vmem_debug_clip_math_v1`:

- A: `debug_math_unbounded_a_pan_45_A2_unbounded_20260921_150029`
- B: `debug_math_unbounded_b_pan_45_A2_unbounded_20260921_150130`
- C: `debug_math_geocov32_pan_45_A2_slam_covisibility_B32_20260921_150232`

Both A/B and A/C comparisons report `clip_attention: [math, math]`, equal
recorded environments, no settings/provenance mismatches or missing listed
provenance, and zero different recorded generation events. Initial CLIP/VAE
encoding, conditioning, initial/sampler noise, final diffusion latents and
decoded samples match across these two-action runs. The A/C pixel audit
additionally verifies all saved PNG hashes and finds all nine decoded frames
identical. Actions, eligible frames, selected contexts and reconstruction frame
indices match. No eviction occurs: the bank never reaches B=32.

This passes the short generation-output check, not full memory-state parity.
The A/C resource audit reports 679 versus 678 surfels after the first update
(step 0), also seen at retrieval on step 1. The preceding completion logs show
different first-action merge counts even between A and B: A `[218,144,255,283]`,
B `[215,164,266,276]`; C `[218,143,249,282]`. Therefore geometry variation also
occurs without changing policy.

`all_recorded_events_equal` is not a geometry-equivalence assertion: existing
debug events hash reconstruction phase RNG states, not CUT3R predictions,
aligned depths/points/confidences, or the full surfel/reference bank. A one-item
count difference does not bound differences in surfel values or references.
In these nine frames it has not changed the selected contexts or decoded
video; its effect at longer duration is unknown.

Static follow-up found that the already-supplied first-action `init edge (0*,1*)`
scores differ: A `25.695222854614258`, B `25.696765899658203`, C
`25.692729949951172`. These are computed from means of predicted confidences
in [edge_conf](extern/CUT3R/cloud_opt/commons.py), invoked before the optimization
loop by [minimum_spanning_tree](extern/CUT3R/cloud_opt/dust3r_opt/init_im_poses.py).
The confidence arrays come from `pred1.conf_self` and `pred2.conf` in the
[aligner constructor](extern/CUT3R/cloud_opt/dust3r_opt/base_opt.py).
This places an observed discrepancy before iterative alignment and surfel
merging. It does not distinguish prediction differences from preprocessing or
confidence-score computation, and it does not identify a faulty CUDA kernel.

The next targeted diagnostic holds the first five saved RGB inputs and
commanded poses fixed and compares reconstruction preprocessing, raw CUT3R
predictions and aligned outputs before investigating merge thresholds. A
[reconstruction-only probe](VMEM_RECONSTRUCTION_PROBE.md) is now implemented,
with CPU tests and GPU results pending user execution on CECSL. No geometry
algorithm was changed and no GPU run was performed here. The separate policy-dependent diffusion RNG issue
after eviction also remains pending an isolated-phase, budget-crossing control.
These results do not reverse the old VBench scores or establish a quality gain.

## Probe Scope

[probe_vmem_clip.py](scripts/probe_vmem_clip.py) uses the existing image loader
and conditioner, importing the usual runtime modules to preserve import-time
backend settings, but instantiates **only CLIP**. It does not load video-generator,
VAE or CUT3R weights or generate a video. It verifies input/config hashes and
requires the reloaded input tensor to exactly match the reference diagnostic.
Initial encoding uses FP32 without adding autocast/inference-mode overrides,
matching the initial-encoding call site. All submodules must be in eval mode.

The probe hashes loaded parameters and buffers (including nonpersistent
normalization buffers) on CPU before the usual FP32 device transfer, fingerprints
installed CLIP/visual/resize source files, records environment flags, and saves
preprocessed inputs and embeddings for three forward passes. NPZ artifacts are
hash-verified before read-only comparisons; no pickle loading is used. Error
values are in preprocessing/embedding units, not video quality metrics.

Run each profile in two fresh processes:

- `native`: existing attention dispatch, no backend change.
- `math`: disable the MHA native fast path and select math SDPA, restoring both
  settings afterward. No change to precision, TF32 or convolution flags. This
  is an attention-dispatch diagnostic, not a guaranteed deterministic mode or
  evidence that native kernels are faulty. PyTorch documents the
  [SDPA backend control](https://docs.pytorch.org/docs/2.7/generated/torch.nn.attention.sdpa_kernel.html)
  and [MHA fast-path control](https://docs.pytorch.org/docs/2.7/backends.html#torch.backends.mha).

The original standalone probe did not change generation. Its reduced model-load/
allocator history differs from full VMem; stable probe results alone cannot
certify full-pipeline reproducibility. The follow-up integration below changes
generation source hashes, but neither the frozen manifests nor old artifacts.

## Completed Probe Commands

Historical commands, retained for reproducibility. GPU reports have now been
supplied by the user; nothing was run on the Mac. Updated CPU tests still need
user execution after pulling the integration changes:

```bash
conda activate vmem
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -p 'test_clip_probe.py' -v
```

Then choose a currently free GPU with `nvidia-smi`. These four processes run
sequentially and load only the CLIP weights, but do not reserve GPU memory:

```bash
GPU=1
REF=outputs/vmem_debug_observe_v1/debug_unbounded_a_pan_45_A2_unbounded_20260921_140657
OUT=outputs/vmem_clip_probe_v1
for PROFILE in native math; do
  for REP in a b; do
    CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/probe_vmem_clip.py run \
      --reference-run "$REF" --attention "$PROFILE" \
      --output "$OUT/${PROFILE}_${REP}" || break 2
  done
done

python scripts/probe_vmem_clip.py compare --left "$OUT/native_a" --right "$OUT/native_b"
python scripts/probe_vmem_clip.py compare --left "$OUT/math_a" --right "$OUT/math_b"
```

Output directories cannot be reused or overwritten. Retain failures. Inspect
`environment_matches_reference` in each run report and input/weight/source/
environment equality in comparisons before interpreting output differences.
Stable preprocessing with variable embeddings narrows the discrepancy to the
visual forward path; varying preprocessing must be investigated first. Weight
differences invalidate a numerical-kernel interpretation. Native/math outputs
need not match each other; the test is within-profile repeatability, including
across fresh processes, without selecting settings by video-quality wins.

## Full-Pipeline Gate

The commands below document the completed two-action control, not a request to
repeat it. See the measured generation/geometry distinction above.

`--clip-attention math` applies that exact profile through `CLIPConditioner`
to both initial-image encoding and every generated-frame batch. The profile
helper is shared with the probe in `clip_attention.py`. Backend state is
restored on normal return or error, so it does not wrap VMem diffusion or
CUT3R reconstruction. It does not change checkpoints, GeoCov scoring, the CLIP
descriptor type, RNG handling, TF32 flags or the caller's autocast context.
In particular, the initial encoding remains FP32 and generated-frame encoding
keeps the existing generation autocast. The full-generator check now provides
short-run conditioning/output agreement, beyond the initial-image-only probe;
it does not verify every archived embedding or longer-run behavior.

The default remains `native`. The action runner currently accepts `math` only
for fresh, unlocked `--generation-debug` runs with checkpoints disabled and at
most 12 four-frame actions. Resume and frozen-manifest runs are not being
migrated yet. The runner records the profile in its arguments, metadata and
effective YAML; provenance includes the shared helper. Both comparison tools
flag mismatched profiles, treating historical absent flags as native.

After pushing/pulling, first run the CPU suite on CECSL, not on the Mac:

```bash
conda activate vmem
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
nvidia-smi
```

If tests pass and GPU 1 is available, run this **three-process, two-action**
control. It uses `observe`, not `isolated`, so the only numerical intervention
relative to the earlier triplet is CLIP attention dispatch. Each video has nine
frames: no GeoCov eviction is possible. This bash loop runs sequentially and
does not reserve GPU memory.

```bash
GPU=1
ROOT=outputs/vmem_debug_clip_math_v1
for CASE in unbounded_a unbounded_b geocov32; do
  POLICY=(--memory-policy unbounded)
  if [ "$CASE" = geocov32 ]; then
    POLICY=(--memory-policy slam_covisibility --memory-budget 32)
  fi
  CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/run_vmem_demo_actions.py \
    --image test_samples/oxford.jpg --run-id "debug_math_${CASE}" \
    --output-root "$ROOT" --trajectory pan_45 --num-actions 2 \
    --fps 13 --frames-per-action 4 --step-size 0.1 --seed 501 \
    --frame-storage resident --memory-scope surfel_indexed_view_memory \
    --inference-steps 50 --surfel-niter 400 --checkpoint-every 0 \
    --generation-debug observe --clip-attention math "${POLICY[@]}" || break
done
```

Set `A`, `B`, `C` to the exact printed directories for unbounded A, unbounded B
and GeoCov-32, respectively, without mixing previous attempts:

```bash
python scripts/audit_vmem_generation_debug.py --left "$A" --right "$B"
python scripts/audit_vmem_generation_debug.py --left "$A" --right "$C"
python scripts/audit_vmem_pairing.py --unbounded "$A" --bounded "$C" --compare-pixels
```

Require matching settings/provenance/environment before interpreting the
fingerprints. Check initial encoding, conditioning, noise, decoded values,
geometry/retrieval traces and all nine saved frames. If they differ, investigate
the earliest remaining mismatch instead of launching 60-second runs. Even a
passing short gate does not prove full CUDA determinism or long-run quality.
The separate post-eviction RNG issue still needs an isolated-phase control.
The user-run short GPU validation above is available. No new quality evaluation,
post-eviction validation or full geometry parity result is available.
