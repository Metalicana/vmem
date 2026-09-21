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

No generation source or frozen protocol is changed by adding this standalone
probe. Its reduced model-load/allocator history differs from full VMem; stable
probe results alone cannot certify full-pipeline reproducibility. A stable math
probe would motivate a symmetric, versioned full-pipeline control, not an
immediate rerun of the benchmark or a quality-improvement claim.

## CECSL Commands

Nothing below has been run on the Mac. New probe tests and GPU behavior are
pending user execution. First run the CPU tests in the generation environment:

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
