# VMem Paired Quality Evaluation

This evaluates the existing, validated videos. No generation, training, GeoCov
tuning, or change to generation source hashes is involved. Implementation and
CPU regression tests are authored; no tests or scoring were run on the Mac.
The user has supplied a complete six-dimension Oxford table from CECSL.
An implementation review found experimental-control issues; see below before
attributing these differences to the eviction rule.

## 2026-09-21 Complete Table and Attribution Warning

After updating PEFT from 0.7.1 to 0.10.0, the user confirmed that pretrained
DreamSim loaded on GPU 1 and supplied the following repeat-evaluation table.
`pip check` still flagged Decord's platform support; actual metric execution
subsequently produced all six paired scores. Raw repeat-evaluation files have
not been downloaded or independently checked here.

| Dimension | Unbounded | GeoCov-32 | GeoCov minus unbounded |
|---|---:|---:|---:|
| Aesthetic quality | 0.584697 | 0.576434 | -0.008263 |
| Imaging quality | 0.728850 | 0.716370 | -0.012481 |
| Subject consistency | 0.958394 | 0.947820 | -0.010574 |
| Background consistency | 0.949241 | 0.943383 | -0.005858 |
| Motion smoothness | 0.977673 | 0.974621 | -0.003052 |
| Dynamic degree | 1.000000 | 1.000000 | 0.000000 |

These supersede the partial table for reporting this repeat attempt, not the
historical files. Scores and differences are independently rounded. The small
changes in the repeated first three scores also mean evaluation is not shown
to be bitwise reproducible; their cause is not established here.

The [implementation review](VMEM_IMPLEMENTATION_REVIEW.md) found that the
generation pair already differs in geometry and retrieval before any eviction,
and reconstruction consumes the same CPU RNG used for future diffusion noise.
The five negative quality/consistency differences are observations for these
two videos, not an isolated estimate of the controller's effect. Do not erase
them or claim that fixing experimental controls is guaranteed to improve them.

## 2026-09-21 Oxford Partial Results

The user supplied the evaluator's partial table for
`transfer_v2_oxford_pan_45`, output root
`outputs/vmem_transfer_v2_resident_quality_compat2`. Six jobs completed: both
arms for each of the following three dimensions. Values and differences below
are copied from the printed table (each independently rounded).

| Dimension | Unbounded | GeoCov-32 | GeoCov minus unbounded |
|---|---:|---:|---:|
| Aesthetic quality | 0.584465 | 0.576284 | -0.008180 |
| Imaging quality | 0.728686 | 0.716447 | -0.012239 |
| Subject consistency | 0.958416 | 0.947994 | -0.010421 |

All three favor unbounded in this one scene/path/seed. These proxies support
the user's impression of degradation on this pilot, not an aggregate transfer
conclusion, a significance claim, or an equivalence/non-inferiority result.
The raw result JSON and resource curves have not been inspected locally.

Unbounded background consistency failed while loading DreamSim's adapter:
`LoraConfig.__init__() got an unexpected keyword argument 'layer_replication'`.
Neither background-consistency arm has a completed score; motion smoothness
and dynamic degree were not reached. Missing scores are not zeros. The passed
import/writer preflight did not exercise neural-model loading. User-reported
tests passed with one skip, and preflight passed after installing
`scenedetect==0.6.7.1`.

This failure requires compatible PEFT config support, not removal of fields
from the checkpoint or replacement of DreamSim by a different feature model.
[PEFT 0.10.0](https://github.com/huggingface/peft/blob/v0.10.0/src/peft/tuners/lora/config.py)
includes `layer_replication`. The following targeted fix is proposed for CECSL,
not yet runtime-validated. Do not change an environment while another evaluation
uses it. Keep the failed attempt and its recorded package versions intact:

```bash
conda activate vbench
python -m pip install --no-deps "peft==0.10.0"
python -m pip check
```

`--no-deps` leaves Torch, Transformers and other installed packages untouched;
it does not prove their compatibility. Resolve any reported relevant dependency
conflict before proceeding. On the available GPU, check actual model loading
with the same cache used by VBench's background-consistency evaluator:

```bash
CUDA_VISIBLE_DEVICES=1 python - <<'PY'
from pathlib import Path
from dreamsim import dreamsim

model, _ = dreamsim(pretrained=True, device="cuda", cache_dir=str(Path.home() / ".cache"))
print("DreamSim pretrained model loaded successfully")
PY
```

If that succeeds, rerun all six dimensions on the same saved videos in a fresh
root, `outputs/vmem_transfer_v2_resident_quality_peft010`, using the `run` command
below and GPU 1 if still available. This gives one complete attempt with a single
recorded environment instead of silently merging scores across dependency
versions. No video regeneration is required. The proposed PEFT change, model
load and repeat evaluation have not been run on the Mac.

## TorchVision Compatibility

The first CECSL attempts failed before producing scores: `vmem` lacked Decord;
after switching to `vbench`, VBench-Long could not import
`torchvision.io.write_video`. TorchVision deprecated that PyAV-based API for
removal in 0.24 ([official 0.23 source](https://github.com/pytorch/vision/blob/v0.23.0/torchvision/io/video.py)).

The adapter now supplies a process-local PyAV video-only writer when the native
API is absent. It uses integer-fps RGB frames, `libx264`, `yuv420p`, encoder
defaults and packet flushing, matching the legacy settings used by VBench here.
It does not resize, change fps, add audio, or edit installed TorchVision/VBench
files. A working native writer is left in place. Both arms use the same selected
writer; its name, PyAV/TorchVision versions and FFmpeg library versions are
recorded in `evaluation_spec.json` and the evaluator logs. Bitwise equivalence
across different encoder versions is not asserted.

The runner now checks requested metric imports and a tiny CPU encode/decode
round trip **before creating its output directory**. This is an environment
check, not a quality measurement or a check of model-weight availability. It
does not load scoring models. A later cold-cache model download can still fail.
No PyTorch/TorchVision downgrade is required for this missing-API error.

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
`conda activate vbench` on CECSL, and stay in `~/vmem`. Do not install VBench's
dependencies into the generation environment. Defaults assume `~/VBench` is the
same checkout used for those evaluations. This environment needs working
VBench-Long dependencies (including DreamSim, PySceneDetect and MoviePy's
`moviepy.editor`, supplied by MoviePy 1.x), plus the `ffmpeg` and `ffprobe`
binaries. Model downloads may occur on a cold cache. The runner does not install
anything or modify the VBench checkout.

After pulling the writer fix, check that environment explicitly without models:

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -p 'test_vmem*.py' -v
CUDA_VISIBLE_DEVICES="" python scripts/run_vmem_vbench_long.py \
  --vbench-root "$HOME/VBench" --check
```

The encoding tests require CPU PyTorch and PyAV; the native-writer parity test
skips when the removed TorchVision API is unavailable. The `--check` command
must pass in the real evaluation environment even if optional tests skipped.
If it specifically reports missing PyAV, install it **only in `vbench`** with
`python -m pip install av`, then repeat the check. Other import errors should be
diagnosed from their tracebacks before changing packages.

After tests pass, choose an available GPU with a fresh `nvidia-smi`, then:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/evaluate_vmem_quality.py run \
  --inventory outputs/vmem_transfer_v2_resident_inventory.json \
  --case-id transfer_v2_oxford_pan_45 \
  --vbench-root "$HOME/VBench" \
  --output outputs/vmem_transfer_v2_resident_quality_compat
```

This runs all six dimensions on both videos, not new generation. Resource use
depends on the installed evaluators; GPU selection is not a memory reservation.
Do not change generation/evaluation checkouts or model caches mid-evaluation.
Existing output roots are refused to avoid mixing stale results. Failures and
partial results remain on disk. On failure, inspect the printed `evaluate.log`
path; the runner also echoes the tail of that log so the child traceback is
visible without another command. Do not delete evidence or silently count a
missing score as zero. A retry
uses a new output root and does not regenerate videos.

To print the current table, including any explicitly incomplete dimensions:

```bash
python scripts/evaluate_vmem_quality.py summarize \
  --output outputs/vmem_transfer_v2_resident_quality_compat
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
