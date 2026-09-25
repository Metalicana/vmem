# VMem Memory Ownership and Measurement Audit

## 2026-09-25 Newton Port, Not Yet Executed

[The Newton runbook](VMEM_NEWTON.md) adds dedicated environment setup, an
allocated CUDA/native-extension smoke, and a separate 18-action Oxford pair
through eviction and one commanded return. The foreground Slurm adapter keeps
the inherited GPU mask and records allocation identity; timeout warning exits
preserve the last committed checkpoint. Both arms keep v3 generation controls.
Long Oxford/full-suite jobs require explicit approval and a revalidated short
profile. No Newton environment, test, job or quality result is claimed yet.
No generation/retrieval/policy/recovery core source was edited for this port.

Reports now include measured action times, allocator allocated/reserved peaks
and maximum sampled post-update RSS (not a continuous host-memory peak). A
short-pilot constant-rate projection does not establish that unbounded 60-second
generation fits an 80GB H100 or four hours. The same resource-bound limitations
remain; the old CECSL pilot is preserved separately.

## 2026-09-25 Resume-to-Results Automation

The user chose to resume the 60-second Oxford v3 pair. New
[one-command orchestration](VMEM_RESULTS.md) reuses the original lock and last
committed checkpoint, runs inventory/prefix validation, six VBench-Long
dimensions, commanded-return RGB MSE, and produces a portable HTML/video/CSV
report. It waits for an idle selected GPU and preserves incomplete attempts.
No generation, retention, reconstruction or checkpoint implementation changed
in this update. No experiments/tests were run on the Mac.

The previous unbounded process was reported at action index 94 and cancellation
was requested; completion/cancellation is not inferred from that request alone.
The controller checks for live jobs before resuming. Resumed resource traces
are labelled by session, not promoted to uninterrupted latency measurements.
No new video-quality result, seed CI or benchmark improvement is established.

## 2026-09-23 Combined-Control Pass and V3

The CECSL combined math-CLIP/math-CUT3R/isolated-RNG pair now passes: all 33
saved images through first eviction match exactly, with no pre-eviction trace
divergences. Diffusion noise matches throughout all 12 actions; reconstruction
end-state RNG first differs at step 8 and conditioning/output at step 9, after
eviction. Final resident counts are 32 per payload component for GeoCov versus
49 for unbounded, with no pending evictions or illegal retrievals.

The next experiment is a fresh matched 60-second pair, not another short
diagnostic. [Transfer v3](VMEM_CONTROLLED_TRANSFER_V3.md) implements these same
controls without tensor fingerprinting and includes recovery identity checks.
The cases, seeds and GeoCov scores are unchanged. Added CPU tests await CECSL;
no tests or experiments ran on the Mac. The new long-run and CUDA resume paths
are not yet validated. Earlier quality results remain negative, not superseded
by this repeatability pass. B still does not bound total process or disk memory.

## 2026-09-23 Math Reconstruction Pass

The user-run math-attention CUT3R probe now matches all preprocessing (40),
prediction (30) and aligned-output (19) arrays exactly, within and across two
processes. The recorded controls, RNG, attention flags and weight checks pass.
This is a repeatability result for the tested five images, not a video-quality
result or a universal determinism guarantee.

The generation runner now has opt-in `--cut3r-attention math`, scoped to CUT3R
inference and restored before alignment. Its default is still native. GeoCov
scores, model weights and existing results are unchanged. The subsequent
[49-frame combined-attention pair](VMEM_GENERATION_DEBUG.md#combined-attention-generation-check)
passes as recorded above. The v3 manifest is now implemented; its source lock
must be created on CECSL after testing and pulling the new implementation.

## 2026-09-23 Fixed-Input Reconstruction Result

The user passed all 16 original reconstruction-probe tests and completed two
fresh CECSL processes, each repeating fixed-input CUT3R reconstruction twice.
Matched inputs, settings, loaded weights, recorded environments and RNG states
produce identical preprocessing but differing raw predictions (24/30 arrays)
and aligned outputs (16/19 arrays), within and across processes. This reproduces
an inference-stage variability source without GeoCov, before alignment or surfel
merging. It does not identify a kernel or explain the VBench quality direction.
See [the evidence and next control](VMEM_RECONSTRUCTION_PROBE.md).

A subsequent math-attention probe passes as recorded above, and an opt-in
generation hook is now implemented. There is still no new quality result or
corrected 60-second rerun. The later combined-control result is recorded above.

## 2026-09-23 Budget-Crossing Diagnostic

The user completed the math-CLIP/isolated-RNG 12-action pair on CECSL. All
recorded initial and sampler diffusion noise matches, including after eviction;
the final GeoCov metadata reports 49 durable frames and exactly 32 resident RGB,
latent, embedding, intrinsic and depth payloads, with owned arrays and no pending
evictions. The pre-eviction output check still fails: conditioning first differs
at step 5, and frame 21 is the first unequal decoded image, before first eviction
at step 7 after frame 32. Both arms retrieve only eligible frames. See the
[full result and limits](VMEM_GENERATION_DEBUG.md#budget-crossing-result-2026-09-23).

This validates the tested noise isolation and final residency, not improved
quality or deterministic geometry. The fixed-input reconstruction probe has
since reused these saved images; its result is recorded above. GeoCov
scoring and the old VBench measurements are unchanged. No local experiments
were run, and no new long-generation protocol has been frozen.

## 2026-09-21 Quality-Drop Review

The complete user-supplied Oxford quality table favors unbounded on five
dimensions. The [implementation review](VMEM_IMPLEMENTATION_REVIEW.md) found
pre-eviction divergence in the saved traces and a shared diffusion/reconstruction
RNG stream, so the difference cannot yet be isolated to pruning. The downloaded
resident traces still show legal retrieval and 32 live payloads per component
after the budget binds. No generation code or scoring coefficients were changed
by that initial review. The user now reports all 21 pairing/pixel-audit tests
passed on CECSL. Hash-verified decoded PNGs confirm a small real difference at
generated frame 1 (MAE 0.169186 and RMSE 0.447214 on the 0-255 channel scale),
before eviction or reconstruction could affect generation. Its cause and later
quality impact remain unresolved. Newly added opt-in
[generation diagnostics](VMEM_GENERATION_DEBUG.md) record conditioning/noise
and support isolated phase/action RNG for short runs. The subsequent full
CECSL suite reported 122 tests: 119 passed and three VBench writer tests skipped;
all generation-debug CPU tests passed. User-supplied logs now show both
unbounded `observe` repeats and GeoCov-32 completing two GPU actions and export.
Both user-run fingerprint comparisons (unbounded repeat and unbounded/GeoCov)
first differ in the initial CLIP embeddings. Input, initial VAE latent, recorded
RNG states and initial/sampler noise match. This localizes the earliest
discrepancy to image encoding, not eviction. The user's subsequent standalone
[CLIP probes](VMEM_CLIP_PROBE.md) show variable native embeddings with matching
inputs/weights and bitwise-stable math-profile embeddings within/across two
fresh processes. The opt-in CLIP-only `--clip-attention math` integration has
now passed the two-action generation-output check: recorded events match in
both unbounded-repeat and unbounded/GeoCov comparisons, and all nine A/C saved
frames have hash-verified identical decoded pixels. Geometry still differs
(679 versus 678 surfels after the first update), while selected contexts and
reconstruction input indices match. Debug phase RNG hashes do not hash geometry.
See the [measured results and reconstruction follow-up](VMEM_CLIP_PROBE.md#completed-generation-check-2026-09-21).
The particular kernel cause and long-run quality impact remain unresolved;
isolated-mode post-eviction GPU results are now recorded in the update above.
A [fixed-input reconstruction probe](VMEM_RECONSTRUCTION_PROBE.md) is now
available to localize the remaining variation without generating more video.
It loads only CUT3R weights; original CPU/GPU results are now recorded above.
These changes do not change the old videos or GeoCov
scores. Default RNG behavior is unchanged; source hashes are new.

## 2026-09-17 Resident Storage Validation

New `--frame-storage resident` releases evicted RGB, latent, embedding,
intrinsic and depth payloads after durable PNG output. Navigator image history
is disabled; resume restores only retained images; MP4 export streams from disk.
See [the v2 contract and CECSL validation gate](VMEM_RESIDENT_MEMORY.md).
The user reports the updated CPU tests passed and supplied a **validated CECSL
pause/resume smoke**: 49 durable/video frames at 576x576, 13 fps; resumed after
10 actions; 32 resident entries in each of the five payload components; owned
array storage. The subsequent resident 60-second Oxford pair also passed the
user-run inventory (**1/15 validated pairs**), and resource plots were generated
for one case. The plots/CSVs and videos still await inspection here; no numeric
savings or quality improvement is claimed. Other GPU occupancy was reported at
launch, so timing comparisons are potentially contended. No local tests or
generation were run. The original transfer-v1 Oxford pair remains an
eligibility-only result, separate from this resident-v2 pair.

The audit below describes **legacy storage**, still the default for old
manifests. Its "no release" entries are not the resident-mode contract.
Resident mode still retains small pose/focal and diagnostic history, and does
not establish constant total RAM/VRAM or a fixed surfel count.

## Historical Legacy Audit

2026-09-08 update: new runs now have per-action PNGs and rolling recovery
checkpoints; see [recovery ownership and restart notes](VMEM_RESTART.md).
This does not release histories or change B's meaning. `runner_frame_hashes`
adds one SHA-256 string per durable frame. Checkpoint serialization/copying can
increase host peaks and disk use, and its cost is logged separately in
`recovery_trace.jsonl`. Restoring NumPy/PIL objects may change backing-storage
layout, while CUDA allocator peaks reset per process; default resource plots
exclude resumed pairs. The user reports the updated unit tests pass and supplied
resumed-smoke metadata/status showing 13 frames, two restored actions and
`complete`. This confirms reported recovery completion, not bitwise equivalence
or full resource/video validation. The detailed CPU pass below predates these
additions. No local tests/generation were run.

Audit date: 2026-09-06; validation update: 2026-09-07. Static code audit of our
VMem fork. The user reported all 38 CPU tests passing on CECSL, including the
new resource-accounting tests. Real GPU instrumentation and generation remain
unvalidated. All tests and experiments are run by the user on CECSL, not on the
coding Mac.

## What Is Bounded

For adapted GeoCov (`slam_covisibility`, B=32,
`surfel_indexed_view_memory`), the post-update **eligible archive has at most
32 frames, including frame 0 and the newest endpoint**. The prospective update
can contain the old bank plus four newly generated frames. Reconstruction uses
that retained-plus-new set. The index drops evicted frame references and surfels
with no surviving references. Its surfel count is not explicitly capped at 32
or any other fixed number.

This is not a bound on total resident frame payload, RAM, VRAM, disk output, or
measured retrieval time. In particular, the following full-history lists remain
allocated in both arms. They can make total storage grow even if the eligible
bank stays flat. No payload release or disk-backed history change is included
in this experiment update.

## Ownership Table

Sources: [pipeline](modeling/pipeline.py), [Navigator](navigation.py),
[frame bank](modeling/memory_policies.py), [runner](scripts/run_vmem_demo_actions.py).
Counts and byte estimates will be recorded under the component names below in
`resource_trace.jsonl`; no measured values are available yet.

| Component | Future use in the audited runner | Does eviction release it? | Growth and ownership |
|---|---|---|---|
| `pil_frames` | Reconstruction images; full-video export | No | One PIL image per global frame. Under GeoCov, evicted images are no longer reconstruction candidates, but remain for export and indexed access. |
| `latents` | Selected generation references | No | CPU NumPy history. Evicted entries are not eligible for later reference selection, but their slots and storage remain. |
| `encoder_embeddings` | Selected generation references and GeoCov appearance scores | No | CPU NumPy history; same global frame indexing as latents. |
| `c2ws` | Retrieval ranking, conditioning, controller geometry and reconstruction poses | No | All commanded per-frame camera matrices. The controller currently materializes the full array before selecting retained indices. |
| `Ks` | Intrinsics for selected conditioning frames | No | Global per-frame history, including shared/view-backed arrays. |
| `surfel_depths` | Prior depths when the selected reconstruction inputs all have priors | No | Global slots, overwritten for reconstructed frames; old evicted payload remains. New inputs often prevent use of the prior-depth branch, but it remains part of the API. |
| `surfel_Ks` | Reconstruction focal history, **full-history focal averaging in retrieval** | No | Global slots, overwritten for reconstructed frames. Evicted entries can still affect future retrieval; cannot simply discard them while claiming identical behavior. |
| `surfels` | Scene rendering and spatial merge/retrieval | Yes, orphan entries | Surviving geometry depends on observations and merging. No explicit fixed surfel-count limit; Python objects and backing arrays are counted. |
| `surfel_to_timestep` | Surfel-to-frame voting index | Yes | Evicted frame references removed and surviving surfel IDs reindexed. Reference count is not the eligible-frame count. |
| `memory_buffer` | Eligibility, controller scores and selection statistics | Yes | `_frames` and `_stats` entries removed together. At most B after each GeoCov update; temporary prospective bank may exceed B. |
| `poses`, `focal_lengths` | Legacy slots initialized by reset | Not by eviction | No appends found on the current demo-action generation path; recorded even when empty. |
| `_dino_feature_cache` | Optional MCE/K-center policies, not either transfer arm | No eviction hook | Included defensively; absent in the unbounded/GeoCov pair. Do not confuse this optional cache with GeoCov's CLIP appearance term. |
| `retrieval_trace`, `memory_events` | Diagnostics/export, not image conditioning | No | In-memory trace lists grow. Eligible-index lists can make unbounded retrieval diagnostics grow faster than frame count. |
| `navigator_frames` | Navigator export/undo API; not the runner's primary export | No | Separate list, mostly aliases of `pil_frames`; shared pixel storage counted once. The first returned batch also includes the initial frame. |
| `navigator_poses` | Navigator pose-history export/undo | No | One pose record per action, not per generated frame. |
| `navigator_current_pose`, `navigator_current_K` | Next action | Replaced/constant | Small current-state arrays; may alias other recorded state. |
| `runner_action_records` | Logging and export | No | One record per completed action, also streamed as `actions.partial.jsonl`. |
| `runner_planned_actions` | Drives the frozen trajectory | No | O(planned actions), allocated before generation, constant during that run. |
| `runner_initial_image` | Initial conditioning; runner retains the variable | No | One input tensor retained for the run, generally on the selected GPU. |
| `weights_*` | VMem, VAE, CLIP and CUT3R inference; optional DINO | No | Parameters and registered buffers, deduplicated. Fixed-size baseline, not growing archive data. |
| `denoiser`, `sampler`, `_active_target_frame_indices` | Diffusion schedule/configuration and current target indices | No | Fixed/small runtime state, including denoiser sigmas. Model wrapper aliases the already-counted model. |

Configuration objects, scalar settings, imported libraries, CUDA context/kernel
caches and native model bookkeeping are not exhaustively counted by the
component estimator. They are not identified as growing frame histories. CUT3R
scene/alignment state, generated batches, renderer buffers and merge octrees are
local/transient rather than pipeline histories, but can dominate peak usage.

## Why Payload Release Is Deferred

The bounded path could eventually drop unused latent/embedding/depth payloads
and stream output images, using stable global IDs with disk-backed storage or
tombstones. This is more than removing items from a list:

- List lengths and global frame IDs drive chunking, retrieval, reconstruction,
  output, and Navigator undo. Deleting list entries would shift those IDs.
- PIL aliases in Navigator would keep pixels resident after dropping only the
  pipeline reference. Retained NumPy/Tensor views can keep an entire batch
  allocation alive; deleting one slice need not release its backing storage.
- Full-history `surfel_Ks` averaging is still a generation dependency. An online
  sufficient statistic would have to handle overwritten historical entries and
  preserve the intended numerical behavior.
- `view_context` and unbounded modes still require historical reconstruction
  inputs. A shared storage abstraction would need explicit per-mode contracts.

Keep that engineering change out of the frozen transfer pair. Report measured
component reductions, when observed, and residual growing histories separately.
Even controller work is not established constant time: whole-history pose
conversion and focal averaging remain, and rendering cost depends on surfels.

## Measurement Contract

[resource_audit.py](modeling/resource_audit.py) appends a schema record, an initial
snapshot and a step record after each generation/reconstruction/budget update.
Each step contains `before_update` and `after_update`, eligible IDs/counts,
protected IDs, surfel count and frame-reference count. It logs actual descriptor
branches and the actual `reconstruction_input_indices` used at each step.
Descriptor counts appear in `appearance_descriptor_sources`; unbounded has no
appearance score. Those counts describe the prospective scoring bank, not only
the final survivors.

For every tracked component, report item/live-item counts, Python bytes, unique
CPU and CUDA backing-storage bytes, logical PIL pixel bytes, and their sum.
View-backed allocations and aliased frames are attributed once, to the first
component visited. Thus a component's bytes are an attribution, not the sum of
each entry's logical tensor size. PIL bytes are an estimate, not its exact native
allocation. An empty/aliased component can legitimately add zero payload bytes.

Process RSS is current Linux `/proc/self/statm` resident pages, with a psutil
fallback. CUDA allocated/reserved values and cumulative peaks refer only to the
selected device's **PyTorch allocator in this process**. They exclude other
jobs and non-PyTorch CUDA allocations. They are not `nvidia-smi` totals, and peaks
are not reset per stage. Component estimates are not expected to equal RSS or
CUDA allocation: snapshots are taken while some local/transient values still
exist, and allocators retain cached space.

| Timing field | Boundaries |
|---|---|
| `retrieval` | Entire `get_context_info`: surfel rendering/voting, pose ranking, slot preparation, device transfers and retrieval trace recording. |
| `generation` | Conditioning assembly through diffusion, decoding, new embedding computation and persistent frame/latent/pose appends. |
| `reconstruction` | Entire `construct_and_store_scene`: CUT3R/alignment, depth/focal updates and surfel construction/merging. Does not include the caller's input-list assembly. |
| `memory_update` | Score construction, protected-frame eviction, surfel pruning and memory-event recording. Unbounded returns immediately, but has the same timed boundary. |
| `accounting_seconds` | Component inspection and process-memory queries for both snapshots, outside the four stage timers. Does not include JSON serialization/file writes. |
| Action `wall_seconds` | Navigator action call plus action-record construction; includes stage work and resource tracing, excludes the subsequent partial-action write. |

Stage timers use `perf_counter` and synchronize the selected CUDA device before
starting and before stopping. They are synchronized wall times, not kernel-only
latencies. Their sum is not full end-to-end wall time: preparation, accounting,
file I/O and boundary waits can lie outside these intervals. Initialization and
model loading are separate from generation stages. The first two generation
steps are marked warm-up by convention, not evidence that all startup effects
have ended. Metadata records `model_load_seconds` and `initialization_seconds`
separately; initialization includes initial image/latent/embedding setup, but
not the runner's input-image preprocessing. Scene reconstruction begins with
the first generated action. A resumed process restores state instead of calling
Navigator initialization; its recovery loading/copying is part of total wall time.
The first step uses one reference plus padded target slots; subsequent
steps use the normal four reference/four target layout.

The profiler itself costs CPU time and temporary host memory, proportional to
the structures inspected. It never clones GPU tensors to estimate their bytes.
Both arms must use it. Plot `accounting_seconds` alongside stage measurements
when deciding whether overhead is material; RSS still includes observer effects.
Pre/post snapshots are update-boundary samples, not per-component peak sampling.
Navigator and runner logs are updated after the pipeline returns, so their
snapshot item counts lag the current action by one; pipeline frame counts do not.
JSONL appends preserve earlier completed steps on ordinary failure. SIGKILL or a
node loss may leave an incomplete final line or an unfinished run status. For
new recovery-enabled runs, completed PNGs and the last committed state can
survive too. New process segments have separate warm-up labels; old code's
first-two-global-steps convention remains in its original traces.

## Supported Claims and Pending Evidence

Supported by code inspection: eligibility and index-reference pruning, protected
frames included in B, shared generation/retrieval code across arms, and the
reconstruction-input intervention. This is end-to-end controller transfer.

Pending CECSL evidence: actual enforcement, component/RSS/VRAM growth curves,
retrieval and reconstruction latency, full-video integrity, paired quality and
revisit behavior. There is no measured resource saving or quality improvement
yet. Physical memory bounds, constant-time retrieval, generated coverage of new
space, equivalence to untouched upstream VMem, and superiority of GeoCov's rule
over other matched-budget rules remain unestablished. The last question belongs
to the controlled spatial-policy comparisons in MemCam.
