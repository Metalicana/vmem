# VMem Memory Ownership and Measurement Audit

Date: 2026-09-06. Static code audit of our VMem fork. The new instrumentation
has not been executed or validated. All tests and experiments are to be run by
the user on CECSL, not on the coding Mac.

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
separately; initialization includes initial image/latent setup and first scene
reconstruction, but not the runner's input-image preprocessing.
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
node loss may leave an incomplete final line or an unfinished run status.

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
