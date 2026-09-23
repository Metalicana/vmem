# Short VMem Reproducibility Diagnostic

The user verified all 21 pairing-audit tests on CECSL and compared actual saved
PNG pixels through frame 32. Frame 1 first differs: mean absolute channel error
0.169186, RMSE 0.447214, maximum 26 on a 0-255 scale; 33.7556% of pixels have at
least one changed channel. This is a small average difference with larger local
outliers, not proof of a different noise sample, a specific nondeterministic
kernel, or a GeoCov defect. Eviction only happens after frame 32. The later
negative VBench results remain recorded.

**Earlier result (2026-09-21):** the user completed the standalone CLIP probes and the
two-action `math`/`observe` full-generator triplet. Both unbounded-repeat and
unbounded/GeoCov comparisons have identical recorded generation events; the
unbounded/GeoCov audit also verifies identical decoded pixels in all nine
frames. Geometry still differs after the first update (679 versus 678 surfels),
despite matching selected contexts. See the [measured results](VMEM_CLIP_PROBE.md#completed-generation-check-2026-09-21).
This is a passed short generation-output check, not a passed geometry or
post-eviction check. Completed triplet commands below are historical.

## Budget-Crossing Result, 2026-09-23

The user supplied completed CECSL comparisons for the 12-action runs under
`outputs/vmem_debug_post_eviction_v1`:

- Unbounded: `post_eviction_unbounded_pan_45_A12_unbounded_20260922_205407`.
- GeoCov-32: `post_eviction_geocov32_pan_45_A12_slam_covisibility_B32_20260922_205958`.

Both used math CLIP and isolated phase/action RNG. The comparer reports no
settings/provenance mismatches or missing provenance, and equal recorded
environments. Its completeness checks passed. No initial-noise or sampler-noise
event differs across all 12 actions. Reconstruction end-state RNG first differs
at step 8, while subsequent diffusion noise remains equal: isolation works in
this GPU check despite differing reconstruction draw counts.

The pre-eviction output check does not pass. The first differing recorded
generation event is conditioning at zero-based step 5 (action 6), in `c`, `uc`
and `c2w`; diffusion latents and decoded samples then differ at that same step.
The first differing saved image is frame 21, with channel MAE 7.908666, RMSE
18.162395 and maximum 177 in 0-255 units; 98.6684% of pixels change. Frames
0-20 match. This is substantial paired-output divergence, not a quality score
or evidence of which video is better. First eviction is later, at step 7 after
frame 32 is generated, removing frame 3. The audit also flags pre-eviction
context and geometry differences. The supplied excerpt does not contain their
exact first-step records.

Both arms have no illegal retrieval selections, retain the initial frame in
the eligible bank, and use four context slots after the first step. Both use
the retrieval fallback at step 1. GeoCov descriptors are CLIP throughout, with
zero latent fallback. Descriptor counts of 33/36 describe the prospective
scoring bank before eviction, not a violation of the post-update B=32 cap.
Final GeoCov metadata reports 49 output/durable frames, exactly 32 resident
entries in each of RGB, latent, embedding, intrinsic and depth storage, owned
array storage, and no pending evictions. The payload logical-byte sum is
59,739,264, not total process RAM/VRAM. This log check does not independently
decode the MP4 or validate every resource snapshot.

Thus the noise-isolation and final-payload checks pass; pre-eviction
repeatability remains unresolved. Actual eviction cannot explain divergence
that precedes it. Static review finds scoring reads features without mutating
them and pruning/release is conditional on actual eviction. Earlier same-policy
repeat geometry differences make reconstruction variability a plausible source,
not a proven cause of the frame-21 change or the old VBench deficit.

Do not change GeoCov coefficients or call the quality issue fixed. The existing
[fixed-input reconstruction probe](VMEM_RECONSTRUCTION_PROBE.md) can now reuse
the saved unbounded frames above, without regenerating a video. It separates
preprocessing, raw CUT3R predictions and aligned outputs; a passing standalone
probe would not rule out full-process or surfel-merging effects. The subsequent
user-run native probe now reproduces variation in raw CUT3R predictions, with
matching preprocessing/weights/RNG, within and across processes. All 16 original
probe tests passed. See its report and probe-only math control in the linked
document. No 60-second corrected rerun has been supplied. Production RNG/attention
settings, recovery compatibility and a new frozen protocol still need migration
before a resumable long rerun. Old videos and quality results remain intact.

## Implementation

The runner accepts optional `--generation-debug observe|isolated`:

- `observe` records values without reseeding or restoring RNG. Run this first
  to inspect the existing stochastic behavior.
- `isolated` seeds Python, NumPy, Torch CPU and the selected Torch CUDA device
  independently for model loading, initialization, diffusion and reconstruction.
  Seeds derive from SHA-256 of a version tag, base seed, phase and global step,
  excluding policy, budget, output directory and job order. Phase exit restores
  outer RNG states, including on errors. Reconstruction draw counts therefore
  cannot advance the next diffusion step's noise stream. Both initial CPU
  diffusion noise and the sampler's CUDA noise are inside the phase scope.

Neither mode changes scoring, retrieval, reconstruction input selection, steps,
weights or precision settings. Isolated mode deliberately changes random draws;
it is not a byte-compatible rerun of the old protocol. RNG isolation does not
make all CUDA operations deterministic. See the official
[PyTorch 2.7 reproducibility notes](https://docs.pytorch.org/docs/2.7/notes/randomness.html)
and [RNG fork behavior](https://docs.pytorch.org/docs/2.7/random.html).

`generation_debug.jsonl` records phase-start/end CPU/CUDA RNG fingerprints,
initial input/latent/CLIP fingerprints, diffusion conditioning, actual initial
noise, every sampler noise tensor, final latents and decoded samples. Hashing
preserves values, dtype and shape, but CUDA-to-CPU copies force synchronization:
debug resource timings are NOT benchmark timings. Environment JSON records
package versions, Torch build, selected GPU, backend precision/determinism
flags and relevant environment variables. Existing run specs retain input,
config, source and recorded VMem/CUT3R checkpoint hashes. This is not an
exhaustive hash of dependencies or VAE/CLIP weights.

The reconstruction phase events record RNG states only, not reconstructed
geometry. Consequently `all_recorded_events_equal: true` can coexist with
different surfel counts, positions and references. Use resource/retrieval
traces alongside the generation comparison; geometry count equality alone
would not prove full geometry equality either.

Default runs do not construct a diagnostic or change RNG behavior. Debug runs
must be fresh, unlocked, 1-12 actions, four frames per action, with
`--checkpoint-every 0`. Resume is deliberately unsupported for these short
diagnostics. New source hashes will not match the old experiment lock: do not
overwrite it to bypass the check. No transfer manifest or completed artifact
has been modified.

## CECSL Validation

The user subsequently supplied completion/export logs for all three `observe`
runs on GPU 1, each completing both actions. Their directories under
`outputs/vmem_debug_observe_v1` are:

- A: `debug_unbounded_a_pan_45_A2_unbounded_20260921_140657`
- B: `debug_unbounded_b_pan_45_A2_unbounded_20260921_140807`
- C: `debug_geocov32_pan_45_A2_slam_covisibility_B32_20260921_140909`

First-action merge-count sequences already differ between the unbounded
repeats: A `[207,150,260,277]`, B `[189,160,233,271]`; C reports
`[208,153,259,274]`. This is observed variation in reconstruction/merge output
without a policy change, not proof of its origin or that GeoCov has no effect
after eviction. Subsequent user-run fingerprint comparisons find the same
earliest difference in A/B and A/C: `initial_encoding.embeddings`, step -1.
The recorded input, VAE latent, RNG states and all diffusion/sampler noise match;
conditioning and generated values subsequently differ. Listed provenance and
environments match. This identifies the initial image-encoder path, not a
policy-specific eviction defect. At that stage the nine-frame pixel comparison
and isolated GPU mode were unreported; the subsequent math-profile and
budget-crossing results are recorded above. The [encoder-only probe](VMEM_CLIP_PROBE.md)
is complete; its stable math profile motivated the full-pipeline checks.

The user ran the full suite in the CECSL `vmem` environment on 2026-09-21:
122 tests reported, 119 passed, three VBench video-writer encoding tests skipped
for missing optional dependencies in that environment. All generation-debug
tests passed, including CPU sampler observation parity and RNG isolation.
The printed 0.6/0.5 quality table and PID 123 were mocked test output, not new
evaluation results or a launched generation job. No tests, dry runs or GPU jobs
were executed on the Mac. The command used was:

```bash
conda activate vmem
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
```

New CPU tests cover RNG restoration, different reconstruction draw counts,
observation parity, real sampler callback parity, fingerprints, short-run
guards and diagnostic completeness. They do not certify neural GPU parity.
That earlier CPU gate passed. The following commands document the completed
native **three-run, two-action** diagnostic. Two identical unbounded runs
measure repeat variation; the third uses GeoCov-32. All end at nine frames, so
none can evict a frame. Use the new gate above for the next run.

```bash
GPU=1
ROOT=outputs/vmem_debug_observe_v1
for CASE in unbounded_a unbounded_b geocov32; do
  POLICY=(--memory-policy unbounded)
  if [ "$CASE" = geocov32 ]; then
    POLICY=(--memory-policy slam_covisibility --memory-budget 32)
  fi
  CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/run_vmem_demo_actions.py \
    --image test_samples/oxford.jpg --run-id "debug_${CASE}" \
    --output-root "$ROOT" --trajectory pan_45 --num-actions 2 \
    --fps 13 --frames-per-action 4 --step-size 0.1 --seed 501 \
    --frame-storage resident --memory-scope surfel_indexed_view_memory \
    --inference-steps 50 --surfel-niter 400 --checkpoint-every 0 \
    --generation-debug observe "${POLICY[@]}" || break
done
```

This bash loop runs one process at a time; it does not reserve GPU memory or
promise an OOM-free shared GPU. Preserve each printed run directory. Set `A`,
`B`, `C` to the exact paths of the three attempts, respectively:

```bash
python scripts/audit_vmem_generation_debug.py --left "$A" --right "$B"
python scripts/audit_vmem_generation_debug.py --left "$A" --right "$C"
python scripts/audit_vmem_pairing.py --unbounded "$A" --bounded "$C" --compare-pixels
```

The debug comparer accepts same-policy repeats and requires completed status
and full phase/noise records; missing data is not equality. Preserve raw traces
and check reported settings/provenance/environment differences.

## Interpretation

- Different initial encoding with the same saved input points to encoding or
  upstream runtime differences, not eviction.
- Different initial/sampler noise demonstrates different stochastic inputs.
  Phase RNG records help locate when streams separated.
- Matching conditioning/noise but different latents narrows investigation to
  denoising/model/runtime behavior. Matching latents but different decoded
  samples narrows it to decoding. Neither identifies a specific kernel cause.
- If the unbounded repeat also differs, pre-eviction hash mismatch alone is
  not evidence of a GeoCov-specific bug. Compare magnitude and growth, not
  just equality. A matching short prefix does not validate 60-second behavior.

The math/observe generation-output gate has passed for the supplied nine-frame
comparison. Investigate the remaining geometry variation with fixed inputs;
do not require regenerating the same videos just to inspect reconstruction.
A predefined short B=32 control crossing the eviction boundary now verifies
matched diffusion noise after reconstruction sizes diverge, but still has
pre-eviction output divergence (see the 2026-09-23 result above). No
remaining-suite launch or scoring-rule tuning follows from this result.
Promoting the RNG change to the benchmark requires a new version,
lock, both arms and recovery validation; this debug implementation does not
perform that migration.
