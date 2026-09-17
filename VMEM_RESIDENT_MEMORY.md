# Resident Frame Memory: Transfer v2

2026-09-16. **The CECSL resident pause/resume smoke passed**, based on the user's
supplied `storage-smoke` validator output. It validated 49 durable/video frames
at 576x576 and 13 fps (3.769231 seconds), resume after 10 actions, exactly 32
resident entries in each of `pil_frames`, `latents`, `encoder_embeddings`, `Ks`
and `surfel_depths`, and `arrays_own_storage: true`. The user also reported the
updated CPU tests passed; a new count/transcript was not supplied.

This supersedes the September 11 resume failure on `3a95795` (whose 57 CPU
tests had passed). Restoration now normalizes non-owning arrays before checking
ownership. The new smoke's parent is
`outputs/vmem_resident_smoke_restorefix/resident_smoke_restorefix_pan_45_A12_slam_covisibility_B32_20260916_133928`;
the supplied validation output does not name the resumed attempt directory.

No tests, dry runs, generation or metrics were executed on the Mac. The earlier
validated transfer-v1 Oxford pair applies to legacy storage, not resident v2.

## Oxford Pair: Validated

2026-09-17. The user supplied successful completion logs for both resident-v2
Oxford `pan_45` arms (exit code 0, status `complete`) and the subsequent audit:
**Validated pairs: 1/15**. The resource plotter reported **1 measured case**.
This is one matched case, not completion of the full transfer suite.

Root: `outputs/vmem_transfer_v2_resident`. Completed attempt directories:

- `transfer_v2_oxford_pan_45_unbounded_pan_45_A195_unbounded_20260916_140544`
- `transfer_v2_oxford_pan_45_geocov32_pan_45_A195_slam_covisibility_B32_20260917_104717`

Inventory: `outputs/vmem_transfer_v2_resident_inventory.json`.
Plots/CSVs: `outputs/vmem_transfer_v2_resident_analysis`.
The validator checks completed videos, provenance, commanded actions, resource
traces, resident-bank constraints and retrieval eligibility. The actual plots,
CSVs and videos have not yet been inspected in this workspace; no numeric
savings or quality improvement is asserted from these success messages alone.

The user reports roughly 23 GB of other GPU occupancy at the earlier launch.
Co-tenancy throughout each arm is not established by the supplied logs. Treat
these timings as potentially contended, not a controlled idle-GPU speedup
measurement. Preserve the runs and inspect their resource/video results before
launching the remaining cases. Quality metrics and constant total RAM/VRAM
claims remain unestablished.

For quality scores on this existing pair, use [VMEM_QUALITY.md](VMEM_QUALITY.md).
The new evaluator follows the MemCam/WorldMem VBench-Long custom-input protocol;
it does not regenerate the videos. Scoring and evaluator tests remain pending
CECSL execution.

## Contract

Use `--frame-storage resident`. Transfer v2 sets this explicitly in both arms.
The default `legacy` mode remains for compatibility with earlier manifests and
the interactive application; it does not release evicted payloads.

After each completed action, the GeoCov bank retains at most B frames including
protected frames. `pil_frames`, `latents`, `encoder_embeddings`, `Ks` and
`surfel_depths` have live payloads only at eligible global IDs. Evicted slots are
`None`, not shifted indices or disk-backed retrieval candidates. The optional
DINO feature cache drops evicted entries too. Array payloads own their backing
allocations; one depth slice no longer pins an old reconstruction batch.

The unbounded arm uses the same storage code but keeps all frame payloads since
all views remain eligible and are needed for full-history reconstruction.

An action proceeds in this order:

1. Retrieve and generate using the previous bank; reconstruct retained plus new
   frames in the bounded arm, or all history in the unbounded arm.
2. Run the unchanged controller and prune surfel references/orphans. Record
   pending payload evictions. The prospective bounded bank has at most B+4
   frame IDs; model/reconstruction temporaries are additional workspace.
3. Return from generation. The headless Navigator does not retain images, and
   the runner does not retain the returned batch.
4. Atomically save and hash new PNGs. If saving fails, do not release unwritten
   payloads or commit a new checkpoint.
5. Clear evicted payload slots and verify the resident bank. Only then take the
   post-update resource snapshot and optionally commit recovery state.

The next generation action refuses to start with uncommitted output/evictions.
Resident mode requires four-frame actions, indexed-view memory, no reconstruction
window override and no full-history visualization. Interactive undo and live
policy changes are not supported in this headless storage mode.

## What Still Grows

This is a bound on **resident frame payloads**, not constant total process RAM.
Stable-ID list slots, commanded `c2ws`, focal history `surfel_Ks`, action/pose
records, frame hashes and diagnostic traces still grow. Focal entries are small
independent arrays; keeping their exact sequence preserves the existing
full-history focal average used by retrieval. No focal statistic, camera path,
GeoCov coefficient, visual descriptor, retrieval rule or reconstruction input
selection was changed to improve results.

Surfels and their references are pruned by the existing rule, not given a new
fixed surfel-count limit. Model weights, transient workspace, native allocators
and caches can dominate RSS/CUDA peaks even when frame payloads plateau. Output
PNGs, videos and rolling checkpoints consume disk space outside the active bank.
The estimator's list/object overhead includes the growing tombstone metadata.

## Recovery And Measurement

`vmem_recovery_v2` checkpoints contain retained payloads (plus tombstones and
small historical state), the retained image IDs, RNG and hashes of all output
frames. Resume verifies/copies every PNG into a new attempt but decodes only
the retained images into RAM. It never makes an evicted frame eligible again.
Deserialized resident arrays are given independent backing allocations when
needed, including cached descriptors and small focal entries, before the strict
bank/ownership checks. This preserves values, dtypes, global IDs and tombstones;
it does not reload evicted payloads or change the controller. Legacy restoration
is unchanged.
Old recovery-v1 checkpoints cannot be resumed with this source version. Keep
them and their original code/locks intact as historical evidence.

Recovery also requires identical source hashes. The September 11 ownership fix
changes those hashes: preserve the paused `3a95795` smoke and its failed resume,
but start a fresh smoke under the fixed code rather than bypassing identity
checks. Do not edit code or pull changes between the new pause and resume.

The final MP4 is streamed from PNGs, independently of the active bank.
`frame_manifest.json` records every PNG hash. No original output is deleted.

`vmem_resources_v2` adds per-component resident IDs/counts, logical payload bytes,
array-ownership status, durable-frame count, and pending evictions. In resident
mode `after_update_boundary` is `durable_output_and_payload_release`, after
Navigator and generation locals have gone out of scope. Before-update snapshots
still occur inside reconstruction's caller, before policy eviction. These
boundaries differ from the old pilot; do not mix versions' resource curves.

The four original phase timers keep their boundaries. `payload_eviction` times
payload release and its bank checks separately. PNG/checkpoint I/O stays in
`recovery_trace.jsonl`; stage sums are not full wall time. Warm-up is marked per
process session. Resumed pairs remain excluded from default resource plots.

The inventory validator checks payload IDs/counts against both the prospective
and retained banks, owned backing arrays, absence of Navigator image aliases,
all PNG hashes, and video/provenance checks. Plotting adds resident-payload and
stage-time panels, and cumulative peak allocated CUDA alongside reserved bytes.

## CECSL Validation Gate

After pushing from the Mac and pulling on CECSL, run the CPU tests there:

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
```

Tests include real bank-update/pruning methods with synthetic frame payloads,
49-frame eviction, collection of old image/depth references, ownership and
score/bank parity, output streaming, and recovery that decodes only 32 images.
The recovery regressions also force buffer-backed float16/float32 arrays after
loading a checkpoint, check value/dtype preservation and tombstones, then
continue generation with synthetic payloads through 49 output frames. Separate
checks cover release of oversized backing batches and untouched legacy storage.
They are not a neural-generation equivalence test or a measured RAM benchmark.

After the tests pass, choose a GPU using a fresh `nvidia-smi`. Selection does not
reserve VRAM or protect other jobs from contention. Run the short smoke first:
The output roots below are separate from the failed pre-fix smoke attempts.

```bash
GPU=0
CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/run_vmem_demo_actions.py \
  --image test_samples/oxford.jpg --run-id resident_smoke_restorefix \
  --trajectory pan_45 --num-actions 12 --seed 501 \
  --memory-policy slam_covisibility --memory-budget 32 \
  --frame-storage resident --inference-steps 50 --surfel-niter 400 \
  --stop-after-actions 10 --output-root outputs/vmem_resident_smoke_restorefix

PAUSED=$(ls -dt outputs/vmem_resident_smoke_restorefix/resident_smoke_restorefix_* | head -n 1)
```

Confirm the new attempt reports `paused` after 10 actions, then resume without
changing the checkout or settings:

```bash
CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/run_vmem_demo_actions.py \
  --image test_samples/oxford.jpg --run-id resident_smoke_restorefix \
  --trajectory pan_45 --num-actions 12 --seed 501 \
  --memory-policy slam_covisibility --memory-budget 32 \
  --frame-storage resident --inference-steps 50 --surfel-niter 400 \
  --resume-from "$PAUSED" --output-root outputs/vmem_resident_smoke_restorefix_resumed

RESUMED=$(ls -dt outputs/vmem_resident_smoke_restorefix_resumed/resident_smoke_restorefix_* | head -n 1)
python scripts/audit_vmem_runs.py storage-smoke --run-dir "$RESUMED"
```

Expected: validated, 49 video/durable frames, 32 resident entries in each payload
component, owned arrays, and resume after 10 actions. The old 3-action recovery
test never exceeded B=32; it is not sufficient for this change. Stop if any gate
fails. Do not start another hour-long rollout merely to debug ownership.

## Freeze And Launch The New Pair

Only after the smoke passes, build a **new** protocol file and lock on CECSL:

```bash
python scripts/build_vmem_transfer_manifest.py --version v2
python scripts/audit_vmem_runs.py freeze manifests/vmem_transfer_v2.jsonl \
  --output outputs/locks/transfer_v2_resident.json
```

The builder keeps the five scenes, three paths, seeds, duration, inference
settings and paired arms from v1. Only identifiers and explicit resident storage
change. Both builder and freezer refuse to overwrite existing files.

Run Oxford rows 0 and 1 sequentially, under the same new lock:

```bash
mkdir -p outputs/vmem_transfer_v2_resident
nohup env CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/run_vmem_demo_manifest.py \
  manifests/vmem_transfer_v2.jsonl --job-indices 0 1 \
  --experiment-lock outputs/locks/transfer_v2_resident.json \
  --output-root outputs/vmem_transfer_v2_resident \
  >> outputs/vmem_transfer_v2_resident/controller.log 2>&1 < /dev/null &
```

After both finish:

```bash
python scripts/audit_vmem_runs.py inventory \
  --lock outputs/locks/transfer_v2_resident.json \
  --output-root outputs/vmem_transfer_v2_resident \
  --output outputs/vmem_transfer_v2_resident_inventory.json
python scripts/plot_vmem_resources.py \
  --inventory outputs/vmem_transfer_v2_resident_inventory.json \
  --output outputs/vmem_transfer_v2_resident_analysis
```

Expect 1/15 validated pairs, not 15/15. Review the real curves and videos before
the remaining cases. Preserve failures and poor generations. Do not compare a
new GeoCov run with the old unbounded resource measurements or tune the scoring
rule based on quality outcomes. Quality metrics remain separate and pending.
