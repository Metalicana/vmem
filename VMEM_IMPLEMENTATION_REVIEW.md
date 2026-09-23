# VMem Quality-Drop Implementation Review

2026-09-21. Static review of HEAD `8ab465b` and the downloaded resident-v2
Oxford records in `~/Downloads/vmem_oxford_v2_review/outputs`. Generation was
recorded at `e92dc0479dba896cf12548605fcab38d79fe4c46`; the inspected generation,
storage and CUT3R files had no diff against that commit at the initial audit.
No tests, inference, neural metrics, video decoding or new experiment was run
on the Mac. Existing JSON records were read with `jq`. The user subsequently
ran all 21 pairing/pixel-audit tests on CECSL; all passed. The pixel check
confirmed that generated frame 1 differs: MAE 0.169186, RMSE 0.447214, maximum
26 channel levels out of 255; changed-pixel fraction 0.337556. This is a small
average difference, not just PNG metadata, with an unresolved cause. See the
[short generation diagnostic](VMEM_GENERATION_DEBUG.md) for newly added opt-in
noise/conditioning fingerprints and phase/action RNG isolation. The subsequent
user-run full CECSL suite reported 122 tests: 119 passed, including all new
diagnostic tests, and three VBench writer tests skipped. User-supplied logs now
show all three two-action `observe` GPU runs reaching export. Even the two
unbounded repeats differ in first-action merge counts. User-run fingerprint
comparisons now locate the earliest mismatch in initial CLIP embeddings in
both pairs. Recorded input/VAE latents, RNG states and diffusion/sampler noise
match, as do listed provenance/environment fields. The subsequent user-run
[encoder-only probes](VMEM_CLIP_PROBE.md) show stable preprocessing and matching
loaded weights, variable native embeddings, and bitwise-repeatable math-profile
embeddings within/across fresh processes. The subsequent CLIP-only math
full-generator control now passes recorded generation-event equality in both
unbounded-repeat and unbounded/GeoCov comparisons. All nine A/C decoded frames
are identical with saved-file hashes verified. Geometry still differs after
the first update (679 versus 678 surfels), without changing selected contexts.
Existing phase-event fingerprints do not cover reconstruction outputs.
See the [completed check](VMEM_CLIP_PROBE.md#completed-generation-check-2026-09-21)
for earlier differing confidence-derived edge scores. The subsequent
[fixed-input reconstruction probe](VMEM_RECONSTRUCTION_PROBE.md) now reproduces
raw CUT3R prediction differences within/across processes, with matching inputs,
weights, recorded environments and RNG. The user passed its 16 original tests.
A subsequent math-attention probe matches all preprocessing, prediction and
aligned-output arrays within/across two processes. An opt-in inference-only
generation hook is implemented; full-generation validation with both math
controls is pending. This does not identify a particular kernel defect or
explain the long-run quality direction. The earlier
[2026-09-23 budget-crossing check](VMEM_GENERATION_DEBUG.md#budget-crossing-result-2026-09-23)
verifies matched diffusion noise through all 12 actions under isolated RNG,
but still finds output divergence at frame 21 before eviction after frame 32.
Final resident payload counts are all 32. Default RNG
and native attention behavior are unchanged, but
source hashes have changed. No old experiment lock or result was overwritten.

## Findings

### P1: Reconstruction advances the diffusion noise stream

Diffusion obtains its initial noise from the default CPU generator:
[utils/util.py:710](utils/util.py#L710), `torch.randn(shape).to(device)`.
CUT3R constructs random depth maps and poses using that same default generator
for each reconstruction: [optimizer.py:29](extern/CUT3R/cloud_opt/dust3r_opt/optimizer.py#L29)
and [base_opt.py:164](extern/CUT3R/cloud_opt/dust3r_opt/base_opt.py#L164).
Even random initial values overwritten by alignment still consume RNG state.

The runner seeds once before loading models, at
[run_vmem_demo_actions.py:484](scripts/run_vmem_demo_actions.py#L484).
After eviction, the two arms reconstruct different numbers of frames at
[pipeline.py:1896](modeling/pipeline.py#L1896). Consequently a common starting
seed does not couple subsequent diffusion noise across policies. The Oxford
trace first has 36 reconstruction inputs versus 37 at zero-based step 8;
the subsequent diffusion call at step 9 generates frames 37-40.

This is a concrete experimental-control issue, not proof of a directional
quality penalty. Over independent seeds the noise distributions need not be
biased, but one nominally paired seed is not matched per-action noise. A
corrective experiment needs separate phase/action RNG streams and noise-state
fingerprints, including the sampler's CUDA `randn_like` draws at
[sampling.py:395](modeling/sampling.py#L395). Do not only reseed after
reconstruction or change one arm. New RNG behavior requires a new protocol
version/lock; preserve the original videos and scores.

### P1: The measured pair diverges before the first eviction

The saved records show:

| Event | Unbounded | GeoCov-32 |
|---|---|---|
| Initial frame 0 recorded PNG hash | Same | Same |
| First generated frame 1 recorded PNG hash | `c74bea2b...` | `47488228...` |
| Surfel count after action 1 (five frames stored) | 682 | 659 |
| Selected contexts at step 5 / action 6 | `[20,16,14,12]` | `[17,15,13,11]` |
| Eligible bank at that step | All 21 frames | All 21 frames |
| First GeoCov eviction | Not applicable | Step 7 / after action 8, frame 1 evicted |

The recorded camera commands/poses match exactly. The input/config hashes,
listed source hashes, both recorded checkpoint hashes, seed 501, and reported
Torch 2.7.0+cu128/CUDA 12.8 match. Both runs are uninterrupted resident-mode
runs. Before the first eviction the bank and reconstruction input IDs are the
same, and the GeoCov scoring path does not itself use RNG or modify images.

The subsequent user-run audit locates the first recorded PNG hash difference
at frame 1, the first generated frame. It compares 33 pre-eviction records
(frames 0-32 inclusive); it does not rehash PNG files or decode pixels. PNG
encoding differences can change a hash without changing RGB. The user's
subsequent decoded-pixel check ruled out that explanation for frame 1 and
measured the small differences reported above; all 33 compared PNG pairs were
verified against their saved hashes.

The step-8 RNG mismatch above cannot explain the earlier step-0 geometry
difference. Neither eviction nor the first reconstruction can explain the
confirmed first-frame pixel difference: generation produces the
PIL images before reconstruction, although durable output is written afterward.
Possible causes include numerical nondeterminism, unrecorded RNG consumption
or initialization, and unrecorded dependency differences; none is established
by these traces. Initial VAE encoding uses the posterior mean, not a random
posterior sample. The first diffusion call's RNG state and conditioning
fingerprints were not recorded for the old 60-second pair, so its traces cannot
establish their equality retrospectively. The new short diagnostics do record
them and show matching noise, with the first difference in CLIP embeddings even
between two unbounded repeats. These short-run results cannot by themselves
explain the long-run quality gap or prove eviction harmless.

The current inventory validates format, provenance fields, legal bank access
and physical payload release, not equality of the pre-eviction prefix. Its
`validated` status is not a successful no-op-policy equivalence test. Current
source provenance also excludes much of CUT3R, the VAE/CLIP implementation,
dependency versions, and VAE/CLIP weight hashes; matching the listed hashes
does not close all reproducibility questions.

### P2: The appearance term differs from the MemCam method

[pipeline.py:277](modeling/pipeline.py#L277) supplies VMem CLIP ViT-H/14 image
embeddings. The saved descriptor records show CLIP throughout and zero latent
fallback use. MemCam's paper/implementation uses DINOv2-Base descriptors.
VMem's utility equation, weights 0.65/0.35, threshold 0.65, three-observer rule,
pose convention and low-score eviction direction agree with the sibling
implementation. This is not a reversed-score or missing-geometry bug.

However, equal coefficients with different descriptors do not imply equal
affinities or eviction decisions. VMem's visual similarity uses float32 while
the current MemCam helper explicitly computes in float64, an additional
potential tie/threshold sensitivity. Neither change is shown to cause the
observed quality gap. The CLIP adaptation was documented, not a faithful
descriptor-level port. A DINO version would need to be separately identified
and evaluated, not silently swapped into the existing result.

### P2: Reconstruction continuity is a distinct intervention

Bounded reconstruction passes only retained plus new frames; unbounded passes
all history. CUT3R reprocesses that sequence from scratch. Prior depth is only
supplied if *every* reconstruction input has a stored depth, at
[pipeline.py:1530](modeling/pipeline.py#L1530). Fresh target frames do not, so
normal append-and-reconstruct calls do not anchor retained depths through that
branch. Historical surfels stay in place while new surfels are appended/merged.
The render focal is still averaged over the entire `surfel_Ks` history,
including stale entries for evicted views, at
[pipeline.py:1127](modeling/pipeline.py#L1127).

These are real behavioral differences and plausible calibration/geometry
interactions, not a demonstrated faulty local-to-global index mapping. The
prior-depth behavior also exists in the unbounded path. We should measure
focal/scale stability and test identical image inputs before changing geometry.
The existing `view_context` scope can isolate eligibility with full geometry,
but is an attribution experiment, not resident-memory bounding. Do not relabel
such an ablation as the primary resource result.

## Checks With No Fault Found

- All 195 retrieval steps in both runs select only eligible frame IDs. Each
  step after initialization supplies four context slots. The only recorded
  whole-retrieval fallback is step 1 in both arms; the trace does not expose
  every appended pose-fallback candidate inside an otherwise successful render.
- Frame 0 remains eligible throughout; GeoCov protected-frame checks pass.
  Residency does not guarantee the retriever actually chooses an anchor.
- Every bounded resource snapshot reports surfel references within the bank.
  Final live RGB, latent, embedding, intrinsic and depth counts are all 32,
  with 781 total output frames. No silent tombstone retrieval was found.
- Retained arrays are copied without changing values/dtypes, evicted slots use
  stable global IDs, and durable output precedes release. The code does not
  restore evicted images for conditioning. Existing tests cover ownership and
  synthetic bank parity, not full neural-output parity.
- Reconstruction inputs are sorted global IDs and output rows are mapped back
  through `input_time_indices`. The final four inputs are the four new target
  frames on this runner path. No shifted reconstruction index was found.
- Batch scores are not recomputed after each eviction. Sibling implementations
  share this behavior, so it is not a VMem-only port defect; changing it would
  change the controller rather than fix an indexing error.

## Next Diagnostic, No New Generation

Added [audit_vmem_pairing.py](scripts/audit_vmem_pairing.py), a read-only
comparison of specs, actions, retrieval, reconstruction, and saved PNG hashes
when available. The default path is standard-library only. Optional
`--compare-pixels` uses Pillow/NumPy on CPU to rehash and decode the saved PNGs
through the first evicting action, including frame 0. It reports each frame's
maximum/mean absolute channel error, channel RMSE and changed-pixel fraction.
Channel errors are in 0-255 units; there is no resizing, color conversion or
MP4 decoding. Missing/altered files and unexpected formats/sizes fail the
check rather than silently producing a verified match. Post-eviction PNGs are
not decoded by this option. These are reproducibility checks between two
generated outputs, not ground-truth quality scores. The diagnostic supplements,
not replaces, inventory validation, and never loads recovery checkpoints or
rewrites run metadata.

After pushing/pulling, ask the user to run on CECSL:

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -p 'test_vmem_pairing.py' -v
ROOT=outputs/vmem_transfer_v2_resident
python scripts/audit_vmem_pairing.py \
  --unbounded "$ROOT/transfer_v2_oxford_pan_45_unbounded_pan_45_A195_unbounded_20260916_140544" \
  --bounded "$ROOT/transfer_v2_oxford_pan_45_geocov32_pan_45_A195_slam_covisibility_B32_20260917_104717" \
  --compare-pixels > outputs/vmem_transfer_v2_prefix_pixels.json
jq '.saved_frame_pixels | {frames_compared, file_hashes_verified, prefix_pixels_equal, first_difference}' \
  outputs/vmem_transfer_v2_prefix_pixels.json
```

Interpret a matching RGB prefix as evidence of encoded-file differences only;
it does not explain the separately observed geometry divergence. A tiny RGB
error is different from a visibly different first sample; neither alone proves
its mechanism. The file records all per-frame errors for later inspection.

Opt-in observation and isolated phase/action RNG are now implemented but need
CPU/GPU validation; see [the short-run protocol](VMEM_GENERATION_DEBUG.md).
The first GPU diagnostic should be a
short no-eviction control (budget above the entire short rollout), with repeated
same-policy control if needed to measure numerical variability. Compare input,
latent/noise, output and geometry fingerprints per action. Only after that gate
should a short B=32 test cross the eviction boundary and check storage/retrieval
parity. Agree on these user-run checks before launching them; no 60-second
regeneration or remaining-suite launch is authorized by this review.

## Conclusion

The five completed quality/consistency dimensions are lower for GeoCov in the
user's supplied full metric table, and those observations must stay in the
record. They do not yet isolate the controller's effect. There is a confirmed
shared-RNG control problem, observed pre-eviction divergence of unresolved
origin, and known descriptor/reconstruction adaptations. No evidence currently
proves that correcting these will reverse the result. The initial review left
generation unchanged. Follow-up opt-in diagnostic hooks leave default RNG,
retrieval, scoring rules and manifests unchanged; isolated debug runs use a
different, explicitly named RNG protocol, not the frozen benchmark protocol.
