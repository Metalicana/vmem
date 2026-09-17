# VMem Paired Quality Evaluation

This evaluates the existing, validated videos. No generation, training, GeoCov
tuning, or change to generation source hashes is involved. Implementation and
CPU regression tests are authored; no tests or scoring were run on the Mac.
Numerical quality results remain pending CECSL execution.

## Metrics

Use the six dimensions supported by the official
[VBench-Long custom-input path](https://github.com/Vchitect/VBench/blob/master/vbench2_beta_long/README.md).

| Dimension | Interpretation |
|---|---|
| `aesthetic_quality` | Predicted visual appeal; higher is better |
| `imaging_quality` | MUSIQ technical image quality; higher is better |
| `subject_consistency` | Subject appearance consistency proxy; higher is better |
| `background_consistency` | Background appearance consistency proxy; higher is better |
| `motion_smoothness` | Temporal motion smoothness proxy; higher is better |
| `dynamic_degree` | Amount/presence of motion, not a monotonic quality ranking |

The first two directly address appearance degradation. Report all six
separately, with GeoCov minus unbounded, not a newly averaged total score.
Static/collapsed outputs can score well on consistency and smoothness. Camera
movement also legitimately changes appearance. These are not GT fidelity,
revisit geometry, pose accuracy, or the full prompt-based VBench benchmark.
One Oxford pair is a diagnostic case, not a statistically powered transfer claim.

## Protocol

- Select the exact arms from `audit_vmem_runs.py`'s validated matched inventory,
  never search for the best-scoring attempt. A missing arm is not a zero score.
- Copy each original MP4 unchanged into an isolated staging directory. Originals,
  PNG histories, recovery checkpoints and generation metadata are untouched.
- Use `long_custom_input`, `--dev_flag`, upstream slow-fast configs and
  `imaging_quality_preprocessing_mode=longer`. No semantic splitting or static
  filtering. This follows the MemCam/WorldMem custom-input approach.
- Port their clip-grouping adapter into `scripts/run_vmem_vbench_long.py`.
  It fixes grouping by the original full basename; feature models, equal-clip
  aggregation and upstream imaging-quality normalization are unchanged.
- For integer-fps 60.08-second VMem videos, the inspected upstream splitter
  retains source fps and creates 31 26-frame clips. The last covers frames
  `[755, 781)`, duplicating 25 frames from the preceding clip. Record that overlap,
  preserve it identically in both arms and do not claim 31 independent videos.
  Reject unexpected clip counts, lengths, fps or dimensions. Derived clips are
  re-encoded by upstream, so they are not lossless copies of source pixels.
- Evaluate one dimension and one arm per process, sequentially. Aesthetic and
  imaging quality run first for both arms. Both arms use the same evaluation
  seed (0); this is not a promise of cross-device bitwise determinism.
- Save evaluator/config source hashes, installed package versions, input hashes,
  generation provenance and cache environment before scoring. Recheck evaluator
  sources around each subprocess and refuse summaries after source drift.
  Weight caches are reused, not exhaustively hashed: retain the VBench environment
  and caches, and do not mutate them during a matched evaluation.

## CECSL Commands

Push/pull the added scripts first. These changes do not require regenerating
either video. CPU-only tests (standard library, no VBench models needed):

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -p 'test_vmem_quality.py' -v
```

Activate the existing **VBench-capable environment used for WorldMem/MemCam**,
often `conda activate vbench`, and stay in `~/vmem`. Do not install VBench's
dependencies into the generation environment. Defaults assume `~/VBench` is the
same checkout used for those evaluations. This environment needs working
VBench-Long dependencies (including DreamSim, PySceneDetect and MoviePy's
`moviepy.editor`, supplied by MoviePy 1.x), plus the `ffmpeg` and `ffprobe`
binaries. Model downloads may occur on a cold cache. The runner does not install
anything or modify the VBench checkout.

After tests pass, choose an available GPU with a fresh `nvidia-smi`, then:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/evaluate_vmem_quality.py run \
  --inventory outputs/vmem_transfer_v2_resident_inventory.json \
  --case-id transfer_v2_oxford_pan_45 \
  --vbench-root "$HOME/VBench" \
  --output outputs/vmem_transfer_v2_resident_quality
```

This runs all six dimensions on both videos, not new generation. Resource use
depends on the installed evaluators; GPU selection is not a memory reservation.
Do not change generation/evaluation checkouts or model caches mid-evaluation.
Existing output roots are refused to avoid mixing stale results. Failures and
partial results remain on disk. On failure, inspect the printed `evaluate.log`
path; do not delete evidence or silently count a missing score as zero. A retry
uses a new output root and does not regenerate videos.

To print the current table, including any explicitly incomplete dimensions:

```bash
python scripts/evaluate_vmem_quality.py summarize \
  --output outputs/vmem_transfer_v2_resident_quality
```

## Artifacts

- `paired_scores.csv`, `summary.json`: one row per case/dimension, both normalized
  upstream scores and GeoCov-minus-unbounded differences. No cross-case pooling
  or new total quality score. Imaging-quality scores are divided by 100 once,
  matching upstream top-level results; raw files retain upstream native units.
- `clip_scores.csv`: time-indexed aesthetic, imaging, smoothness and dynamic
  scores, with the overlapping tail flagged. Subject/background results instead
  keep upstream fused per-video records and within-/between-clip components in
  their raw result JSON; no fake per-clip fused curve is constructed.
- `<case>/<policy>/clip_coverage.json`: split indices, nominal frame/time ranges,
  decoded format checks and hashes of all derived clips.
- `<case>/<policy>/<dimension>/`: untouched upstream result/full-info JSON and
  evaluator stdout/stderr. `jobs.json` preserves pending/failed/completed states.
- `evaluation_spec.json`: frozen input selection, source/config hashes, package
  versions and evaluation settings. These are metric-run records, not edits to
  the existing generation inventory or generation lock.

Tests cover paired selection, missing/duplicate arms, filename grouping,
normalization, full-duration split coverage and tail overlap, malformed/missing
results and signed paired differences. They do not exercise neural evaluators
or certify the remote VBench environment; that requires the actual CECSL run.
