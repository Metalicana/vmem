# Resident Frame Memory: Transfer v2

2026-09-09. Implementation added; CECSL unit tests and GPU validation are
**pending**. No tests, dry runs, generation or metrics were executed on the Mac.
Prior test passes and the validated Oxford pair apply to the earlier code.

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
Old recovery-v1 checkpoints cannot be resumed with this source version. Keep
them and their original code/locks intact as historical evidence.

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
They are not a neural-generation equivalence test or a measured RAM benchmark.

After the tests pass, choose a GPU using a fresh `nvidia-smi`. Selection does not
reserve VRAM or protect other jobs from contention. Run the short smoke first:

```bash
GPU=0
CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/run_vmem_demo_actions.py \
  --image test_samples/oxford.jpg --run-id resident_smoke \
  --trajectory pan_45 --num-actions 12 --seed 501 \
  --memory-policy slam_covisibility --memory-budget 32 \
  --frame-storage resident --inference-steps 50 --surfel-niter 400 \
  --stop-after-actions 10 --output-root outputs/vmem_resident_smoke

PAUSED=$(ls -dt outputs/vmem_resident_smoke/resident_smoke_* | head -n 1)
CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/run_vmem_demo_actions.py \
  --image test_samples/oxford.jpg --run-id resident_smoke \
  --trajectory pan_45 --num-actions 12 --seed 501 \
  --memory-policy slam_covisibility --memory-budget 32 \
  --frame-storage resident --inference-steps 50 --surfel-niter 400 \
  --resume-from "$PAUSED" --output-root outputs/vmem_resident_smoke_resumed

RESUMED=$(ls -dt outputs/vmem_resident_smoke_resumed/resident_smoke_* | head -n 1)
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
