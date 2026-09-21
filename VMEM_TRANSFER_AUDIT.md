# VMem External Transfer: Audit and MemCam Handoff

## 2026-09-21 Implementation Review

A subsequent complete user-supplied VBench-Long table has five lower
quality/consistency scores for GeoCov-32 and tied dynamic degree. However,
inspection of the downloaded generation records shows geometry divergence
after the first action and different retrieval at step 5, before the first
eviction at step 7. Static code review also confirms that reconstruction
advances the CPU RNG used by subsequent diffusion calls, with policy-dependent
draw counts after the budget binds. Neither finding establishes the cause or
direction of the metric differences. See [the implementation review and
read-only pairing diagnostic](VMEM_IMPLEMENTATION_REVIEW.md). The controller's
arithmetic and low-score eviction direction match the sibling implementations;
no invalid frame access or broken resident release was found. Production
generation behavior, policies and frozen manifests were unchanged by that initial review. Diagnose
these controls before running the remaining suite or making a causal claim.

Follow-up CECSL audit: all 21 pairing/pixel tests passed. The hash-verified
pixel comparison confirms a small real difference at frame 1 (MAE 0.169186,
RMSE 0.447214 in 0-255 channel units), before the first eviction after frame 32.
The shared post-eviction RNG issue cannot explain this earlier divergence.
Opt-in [short generation diagnostics](VMEM_GENERATION_DEBUG.md) now record
conditioning/noise and offer isolated phase/action RNG. The subsequent full
CECSL suite reported 119 passed and three skipped out of 122 tests, including
passing generation-debug CPU tests. The user subsequently supplied completion
logs for all three two-action `observe` GPU runs. The two unbounded repeats
already differ in first-action merge counts. Subsequent fingerprint comparisons
identify initial CLIP embeddings as the first mismatch in both the unbounded
repeat and the unbounded/GeoCov pair, with matching recorded input/VAE latent,
RNG states and diffusion noise. The user's subsequent [encoder-only probes](VMEM_CLIP_PROBE.md)
show matching inputs/loaded weights and stable preprocessing, but variable
native embeddings. The math profile is bitwise repeatable within/across the
two tested processes. New opt-in CLIP-only integration awaits a full-pipeline
short-run check in both arms. The particular kernel cause and long-run quality
impact remain unresolved; isolated-mode GPU validation also remains pending.
Default behavior and scoring are unchanged, but
source hashes have changed. Preserve the old lock/results. No controller
improvement is implied by the diagnostic or by passing CPU tests.

## 2026-09-21 Partial Quality Evidence

The user supplied three completed VBench-Long metric pairs for the validated
resident-v2 Oxford `pan_45` pilot. GeoCov-32 scores below unbounded on aesthetic
quality (0.576284 vs 0.584465), imaging quality (0.716447 vs 0.728686), and subject
consistency (0.947994 vs 0.958416). This one scene/path/seed does not establish a
general quality improvement or preservation claim. Background consistency
failed during DreamSim/PEFT adapter loading; motion smoothness and dynamic
degree were not reached. Completed scores and the failure remain in
`outputs/vmem_transfer_v2_resident_quality_compat2`. See
[the partial results and environment fix](VMEM_QUALITY.md).
The full metric table and resident-v2 resource interpretation are still pending;
do not reinterpret the missing metrics as zeros or tune GeoCov to rescue this
pilot. No generation or evaluation was run on the Mac.

## 2026-09-17 Resident Memory Follow-Up

The user supplied a validated Oxford v1 pair: both 781-frame, 576x576, 13-fps
videos passed the inventory. Resource plots/CSV show lower RAM and recorded
peak CUDA allocation for GeoCov, but higher persistent depth backing storage.
This is an eligibility-only pilot, not a physical frame-payload bound. No
paired quality metrics are available.

Resident payload eviction, disk-streamed export, retained-only image recovery
and stricter instrumentation are now implemented for transfer v2. The user
reports the updated CECSL CPU tests passed and supplied a validated 49-frame
pause/resume smoke: resume after 10 actions, 32 resident entries per payload
component and owned array storage. Both matched 60-second resident Oxford arms
then completed successfully. The supplied inventory reports **1/15 validated
pairs**, and the resource plotter reports one measured case. Actual plots/CSVs
and videos await inspection here; measured savings and quality improvement are
not yet established. Other GPU occupancy was reported at launch: do not use
these timings as a controlled idle-GPU speedup comparison. See
[VMEM_RESIDENT_MEMORY.md](VMEM_RESIDENT_MEMORY.md) for the exact attempts and
artifact locations; inspect them before launching the remaining cases.
Do not mix v1 and v2 resource measurements. GeoCov scoring and the paired
generation settings are unchanged; total process memory is still not bounded.

## 2026-09-08 Recovery Update

The user-reported Oxford unbounded attempt stopped after action index 189
(190 completed actions, 761 frames). No VMem process or CUDA compute job remained;
`running` was stale. Kernel-log access was denied, so OOM is not established.
The original runner saved no incremental image/state checkpoint and cannot
resume that attempt. Keep the old output directory and lock as failure evidence.

Recovery support is now implemented: per-action
atomic PNGs, rolling state/RNG checkpoints every five actions, new-directory
resume, process-session labelling and persistent launcher logs. See
[restart and validation instructions](VMEM_RESTART.md). The 38-test CPU pass
below predates this change. On 2026-09-08 the user reported the updated tests
pass, without a new count/transcript or source hash. The resumed smoke run's
supplied metadata/status subsequently showed 13 frames, two restored actions
and `complete` (PID 3628384). Recovery completed as expected by those records;
decoded video integrity and uninterrupted/resumed numerical parity remain
unverified. No tests or generation were run on the Mac.

The unchanged 15-case manifest must be executed under a **new source/config
lock and output root** after checks. GeoCov scoring and inference settings are
unchanged. Recovery is not an OOM fix or a physical-memory bound; it adds I/O,
host-memory overhead and disk use. Resumed arms are labelled and excluded from
default uninterrupted resource plots. The audit below describes the original
transfer design; its save-at-end limitation is superseded only for new runs.

Audit date: 2026-09-06; validation update: 2026-09-07.
Repository: `/Users/metalicana/projects_summer_2026/vmem`.
Audited base HEAD: `8bc405e`, plus the subsequent generation/experiment changes
described below. No new GPU generation or VBench evaluation was run here.

Current execution boundary: the Mac is code-only. The user runs checks and all
experiments on CECSL after pushing/pulling the code. On 2026-09-07 the user
reported **38 CPU tests passing in 1.732 seconds** on CECSL, including the new
resource/protocol tests. Real CUDA profiling, model generation, full run
inventory validation and plotting remain pending. The pasted test output does
not identify the tested commit or source hashes.

## Decision

MemCam and WorldMem remain the main experimental test-beds. VMem is a supporting
external-backbone transfer experiment: does the existing GeoCov controller also
help a generator that already uses geometric memory retrieval? The central pair
is **VMem unbounded versus VMem + adapted GeoCov, B=32**. RI, FIFO, MCE, K-center,
budget sweeps, and a new metric study are not necessary for this transfer test.

External here means another generation/memory architecture. It does not establish
that these demo images are unseen during training or constitute a held-out dataset.
There is no VMem quality improvement measured in this workspace yet.

The resource objective is preserved or improved quality/consistency at lower
**measured** resource cost, not a physical-memory bound inferred from B=32.
See the [memory ownership audit](VMEM_MEMORY_OWNERSHIP.md) and
[CECSL execution runbook](VMEM_CECSL_EXPERIMENTS.md) for the implementation and
the requested user-run checks. The original transfer manifest is unchanged.

## Design Argument for the Paper

Distinguish **query-time context selection** from **persistent archive retention**.
An archive can grow while a sophisticated retriever chooses only a few references
for each generation step. GeoCov controls what remains eligible in that archive.
The intended finding is that archive curation can complement an already careful
geometric retriever. It is not necessary to portray VMem as a naive baseline.

The VMem project describes querying surfel-indexed views with the target camera,
then writing newly generated views back to memory. This already includes
geometry-aware retrieval and surfel merging. [Official project](https://v-mem.github.io/)

The paper uses a camera-conditioned SEVA-derived generator, with an efficient
LoRA variant using four reference and four target views. It includes real-data
and cycle-trajectory evaluations, as well as qualitative in-the-wild images.
Our repeated 60-second demo-action suite is a new transfer protocol, not a
reproduction of its quantitative benchmark. [VMem paper, Sections 3-4](https://arxiv.org/html/2506.18903v3)

For the proposed two-family introduction, these properties are not mutually
exclusive: an unbounded archive may also have heuristic or geometric selection.
Avoid treating VMem's selection mechanism as evidence that its archive is bounded.

## Audit Findings

### 1. The baseline is the audited fork, not verified pristine upstream

The current `get_context_info` still performs surfel rendering, candidate
allocation, camera-pose ranking and diversity suppression. However, comparison
against the repository's original `b641beb` implementation shows shared changes:
empty-render fallback, candidate deduplication, appending pose-ranked eligible
frames beyond the surfel candidates, context-slot filling, and changed threshold
initialization. The current behavior also is not literally a plain top-K vote sort.

Both transfer arms use this same implementation. Label the baseline as the
unbounded VMem implementation in our audited fork. Before claiming equivalence
to untouched released VMem, perform a separate upstream parity check. This audit
compared local history; it did not certify the latest upstream code.

Evidence: [pipeline.py](modeling/pipeline.py), `get_context_info` around line 1080;
`git show b641beb:modeling/pipeline.py`, original lines 630-754.

### 2. Default budgeting changes both retention and scene reconstruction

`surfel_indexed_view_memory` restricts frame eligibility, removes evicted frame
references from the surfel index, and removes surfels with no surviving references.
For subsequent reconstruction, bounded policies use retained plus new frames;
unbounded uses the full history. The checkpoint and retrieval algorithm are shared,
but the memory state and reconstruction inputs can evolve differently.

This supports an **end-to-end memory-controller** comparison. It cannot isolate
retrieval selection as the cause of an improvement. Optional `view_context`
experiments retain full-history geometry and only restrict frame eligibility;
they are an attribution ablation, not a fully budgeted memory demonstration.

Evidence: [pipeline.py](modeling/pipeline.py), `_prune_surfels_to_memory` line 418,
`_update_memory_budget` line 444, reconstruction selection around line 1846.

### 3. GeoCov is an adaptation, with no new tuned coefficients

The implementation flag is `slam_covisibility`. It uses pose similarity with
weight 0.65 and visual cosine similarity with weight 0.35, threshold 0.65,
three substitute observers, and the same retention utility as the MemCam paper:

```text
u_i = 1 - min(c_i / 3, 1) + 0.5 / (c_i + 1) + 0.25 * (1 - max_affinity_i)
```

VMem supplies its CLIP image-encoder embeddings, with a latent fallback, instead
of MemCam's DINO descriptors. This controller's geometry term is a pose proxy;
it does not compute surfel co-visibility counts. Surfels are used by VMem retrieval.
The initial frame is pinned and the newest endpoint protected; both count toward
B. Utilities are computed once on the prospective bank before batch eviction,
not recomputed after every removed frame.

Evidence: [memory_policies.py](modeling/memory_policies.py),
`compute_slam_covisibility_scores` line 294; [pipeline.py](modeling/pipeline.py),
`_visual_feature_dict`, `_compute_memory_scores`, `_update_memory_budget`.

### 4. B is eligible frames, not total process memory or a fixed surfel count

The runner still accumulates output images, latents, embeddings and pose/depth
history. Surfel count varies with geometry. Do not claim constant total RAM,
VRAM, strict surfel count, or constant measured retrieval latency from this hook.
Current trace fields can verify the eligible bank and surfel/reference counts.
Recorded CUDA peaks cover PyTorch's allocator in this process, not all GPU use.

Evidence: [pipeline.py](modeling/pipeline.py), persistent lists in `__init__`,
frame appends around line 1831, and `_record_retrieval_trace`.

The expanded [ownership table](VMEM_MEMORY_OWNERSHIP.md#ownership-table) separates
generation dependencies from output/diagnostic histories. Notably, evicted
`surfel_Ks` entries still participate in full-history focal averaging. Global
indexing, Navigator aliases and view-backed allocations make payload release a
larger storage change; it is deliberately deferred. This update measures the
existing implementation rather than silently changing its storage semantics.

### 5. Existing evidence is a diagnostic example, not an improvement result

The locally available `open_door_square_walk_60s` unbounded run has 781 frames
at 13 fps. Its commanded pose returns near identity at frames 304 and 608.
In the corresponding retrieval traces, frame 0 remains eligible but is not
selected at either return. This establishes retained-but-unused anchor evidence
in this one run. It does not establish causal poisoning, explain all its visible
failure, or show that GeoCov would improve it. The old metadata does not log seed.

Sources: [actions](open_door_square_walk_60s_square_walk_A195_unbounded_20260713_143244/actions.json),
[retrieval trace](open_door_square_walk_60s_square_walk_A195_unbounded_20260713_143244/retrieval_trace.json),
[metadata](open_door_square_walk_60s_square_walk_A195_unbounded_20260713_143244/metadata.json).

### 6. Generation logging and batch isolation have been improved

The runner saves seed, named source/input/config hashes and git commit, planned
actions in `run_spec.json`, and effective `generation_config.yaml` before model
loading. The new experiment lock rejects mismatched source/config/input/settings
before loading. VMem/CUT3R checkpoint files are hashed after loading and compared
across completed arms. These checks do not cover every dependency or VAE/CLIP
weight file; preserve the cache and environment.

The new `resource_trace.jsonl` records each generation update, before/after
pruning counts and component byte estimates, current process RSS, process-local
PyTorch CUDA allocations/reservations and cumulative peaks. CUDA-synchronized
wall times separate retrieval, generation, reconstruction and memory update.
The first two steps are labelled warm-up; accounting overhead is separate.
Read the [measurement contract](VMEM_MEMORY_OWNERSHIP.md#measurement-contract)
for exact boundaries, observer overhead, alias attribution and exclusions.

`actions.partial.jsonl` and resource JSONL are appended during generation.
`run_status.json` marks success or an uncaught failure; abrupt termination can
leave an unfinished status. These JSONL artifacts alone preserve earlier completed
steps, but cannot recover a video. The recovery update above additionally saves
PNGs and state checkpoints for new runs. The full MP4 is still exported at the end.

Navigator previously deleted the repository-wide `visualization` directory on
initialization. It now creates its pipeline's configured directory without
deleting other jobs' files. Interactive app cleanup remains owned by the app.

## Frozen Generation Protocol

Manifest: [vmem_transfer_v1.jsonl](manifests/vmem_transfer_v1.jsonl).
Builder: [build_vmem_transfer_manifest.py](scripts/build_vmem_transfer_manifest.py).
The builder refuses to overwrite an existing protocol file.

| Setting | Value |
|---|---|
| Scene inputs | Oxford, Jesus, living room, open door, Changi, from `test_samples` |
| Camera cases per scene | `pan_45`, `pan_90`, shallow `out_and_back` |
| Interpretation | `pan_45` spans -45 to +45 degrees; `pan_90` spans -90 to +90 |
| Translation case | 20 forward, 20 backward; step size 0.02, maximum displacement 0.4 in Navigator units |
| Duration | 195 actions, 781 output frames, 13 fps, about 60.08 seconds |
| Generator | Existing checkpoint/config, 576x576, four context/four target slots |
| Inference | 50 denoising steps, 400 reconstruction iterations, no window override |
| Memory arms | Unbounded; adapted GeoCov-32, `surfel_indexed_view_memory` |
| Seeds | One fixed seed per case, identical across its two arms |
| Size | 15 matched cases across five images; 30 videos, not 30 independent scenes |
| Main artifacts | MP4, run spec, effective config, actions, retrieval/memory traces, metadata |

The instrumented execution additionally requires an experiment lock and records
`commanded_path.json`, `resource_trace.jsonl`, partial actions and run status.

No new dataset download, training, feature-model tuning, or Gradio UI automation
is involved. The runner calls the same Navigator actions as the demo. It excludes
`local_loop`, whose forward-turn-backward path does not close in translation.

1. Run rows 0 and 1: Oxford `pan_45`, unbounded and GeoCov-32. These are part of
   the final suite, not throwaway samples. Check both for completed generation,
   correct length/camera commands, and actual budget enforcement.
2. Run remaining rows 2-29 if the common generation setup is operational. Keep
   failures and poor generations in the experiment record. Do not select only
   scenes/seeds where GeoCov wins or change only one arm's inference settings.
3. The user will run VBench later. Keep both arms' original full videos and
   matching frame rate/resolution. No LPIPS/FVD/revisit score is the primary
   acceptance criterion in this external experiment.
4. A second paired seed across the same cases can strengthen the result later.
   It is not a prerequisite for this first transfer pass. Freeze the generation
   settings before seeing VBench results; do not use metric wins to tune the suite.

If both pilot arms collapse, fix shared implementation problems or revise the
protocol explicitly and rerun both. An unbounded OOM is a runtime outcome, not
a quality win. The old RI/GeoCov pilot and revisit scripts remain available as
optional diagnostics but are not the primary transfer protocol.

## Separate Scaling Protocol

The original pans and shallow revisits cannot establish continued-exploration
scaling. Two separate versioned manifests add Oxford fixed-region traversals and
expanding outward excursions followed by returns:

- [30-second pilot](manifests/vmem_scaling_v1_pilot.jsonl): two matched path pairs,
  four videos, 98 actions each.
- [180-second extension](manifests/vmem_scaling_v1_extended.jsonl): same two path
  pairs, 585 actions each; not requested for launch before the pilot is reviewed.

Both use step size 0.02 and seed 701, with the same inference and controller
settings as transfer_v1. Fixed traversals repeat 8 forward/8 backward actions;
expanding excursions use 8/8, 16/16, 24/24, etc. Duration prefixes are fixed,
not selected from quality results. The freeze tool records analytic endpoint
poses/returns; the inventory checks executed poses against them. A path plotter
is available for pre-generation inspection. The user-reported CPU tests include
path closure/extent and agreement with Navigator's actual movement methods with
generation stubbed out. The freeze command and generated paths still need the
pilot checks described in the runbook.

Storage can be sampled at 10/20/30 seconds in the pilot and
10/20/30/60/120/180 seconds in the long extension. Actual sample time is retained.
Commanded extent is not measured generated geometric coverage, and this small
one-dimensional probe is not a general exploration benchmark.

## Commands and Evaluation

Use the [CECSL runbook](VMEM_CECSL_EXPERIMENTS.md), starting with the requested CPU
unit tests, then freezing a source/config lock, then Oxford rows 0/1. It contains
the commands for inventory validation, measured resource plots, the remaining
transfer cases, and the separate scaling pilot. No commands were executed on
CECSL or Newton here. The legacy Newton wrapper is not the current locked-run
protocol; all experiments are now designated for CECSL.

One process at a time and explicit GPU selection are not VRAM reservations.
Choose the device from a fresh status reading and record competing workload
when interpreting latency. Do not mix old uninstrumented videos into the new
resource comparison. Do not interpret the mere existence of a directory or
`metadata.json` as a validated paired run.

VBench-Long has a documented custom-video path supporting six dimensions with
scene/clip preprocessing. Applicability to these videos remains provisional
until the user checks preprocessing/coverage on the pilot; this is not the full
standard prompt-based evaluation. [Official VBench-Long documentation](https://github.com/Vchitect/VBench/blob/master/vbench2_beta_long/README.md)
Keep revisit consistency distinct from outward-generation quality. No new-view
exact-index GT, validated CUT3R evaluation, or paired quality result is available.

## Validation and Outstanding Work

- Historical validation before the resource update: 26 CPU tests and all 30
  transfer manifest dry-run rows passed; the Slurm wrapper passed shell syntax.
  This does not validate the new instrumentation, locks, scaling or inventory.
- On 2026-09-07 the user supplied CECSL output for
  `CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v`:
  38 tests passed in 1.732 seconds, no failures or skips reported. This includes
  backing-storage accounting, stubbed timing, paths, locks and budget checks.
  Model loading, real CUDA synchronization/peaks and generation under the new
  lock still require user-run pilot validation. No tests were rerun on the Mac.
- Inventory and plotting tools are implemented, not measured results. Before
  publishing, collect validated pairs with decoded video integrity, matching
  provenance/checkpoints and traced bank limits. Preserve failed/unfinished
  attempts and explain retries. Quality outcomes remain pending.
- Unmodified-upstream equivalence, physical memory bounds, and a retrieval-only
  causal explanation remain unestablished. No generator/retriever algorithm was
  replaced as part of this generation-protocol update.

## Suggested MemCam Paper Wording

Before results: "We additionally test whether the memory controller transfers
to a VMem-based generator with surfel-indexed retrieval, using matched 60-second
rollouts from single images."

If supported by subsequent measurements: "The controller also improves [measured
dimensions] on VMem, suggesting that archive curation can complement an existing
geometry-aware retrieval mechanism." Report the actual paired outcomes and the
VMem embedding adaptation. Do not pre-fill an improvement claim, call these
results GT reconstruction fidelity, or use them to claim all heuristic memory
mechanisms fail. Keep the main controlled evidence in MemCam and WorldMem.

This transfer experiment cannot on its own establish that GeoCov's scoring rule
is superior to other handcrafted eviction rules. That claim belongs to the
matched-budget spatial-policy comparison in MemCam. Report any actual resource
savings with their measurement scope, never as a bound on total memory.
