# VMem on Newton

2026-09-25 implementation handoff. **No Newton job, environment installation,
test, generation or evaluation was run by this coding session.** Only static
inspection was performed on the Mac. The CECSL artifacts are not Newton results.

## Scope

Use one allocated H100, sequential arms, inherited Slurm CUDA mask, and a new
Newton lock/result root. KEEPSAKE-B32 retains the internal `slam_covisibility` /
GeoCov identifiers. The comparison is the audited fork's unbounded archive
versus its B32 controller, not a pristine-upstream reproduction.

The new smoke manifest takes the first **18 Oxford pan_45 actions** of transfer
v3: 73 frames, 13 fps, 576x576, approximately 5.62 seconds. It crosses eviction
after action 8 and returns to the commanded initial pose at action 18. Both
arms retain 50 denoising steps, 400 reconstruction iterations, four context /
four target slots, resident storage, isolated RNG, math CLIP/CUT3R attention,
seed 501, and checkpoints every five actions. No reader, scoring coefficient,
model precision, reconstruction window or model checkpoint was changed here.

This is an explicitly separate hardware/implementation smoke, **not** a
shortened replacement for the 60-second experiment. It does not run VBench.
The long modes use the unchanged v3 manifest: 195 actions / 781 frames per arm.

## 1. Transfer the Code

On the Mac, review and commit/push the intended files yourself, including the
previously untracked results scripts. This session did not commit or push.
`git status --short` shows files that a normal remote pull will not yet receive.
Required new files include `VMEM_NEWTON.md`, `VMEM_RESULTS.md`,
`requirements-newton-constraints.txt`,
`manifests/vmem_newton_smoke_v1.jsonl`, `scripts/{run_vmem_results.py,
run_vmem_results.sh,report_vmem_results.py,vmem_slurm.py,profile_vmem_newton.py,
setup_vmem_newton.py}`, `tests/{test_results_workflow.py,test_newton_workflow.py}`,
and the four `slurm/newton_vmem_{common.sh,setup.sbatch,gpu_smoke.sbatch,results.sbatch}` files.

On Newton:

```bash
cd "$HOME/vmem"
git status --short
git rev-parse HEAD
conda env list
```

If the repo is absent, clone `https://github.com/Metalicana/vmem.git` into
`$HOME/vmem`. Pull with `git pull --ff-only` only after checking that no job is
using this checkout and there is no conflicting work. Do not reset it.
Do not reuse CECSL locks/checkpoints by default. `$HOME` may resolve to a Lustre
path; use `readlink -f` to check aliases.

## 2. Dedicated Environment

Only when `$HOME/.conda/envs/vmem` does not already exist:

```bash
conda create -p "$HOME/.conda/envs/vmem" -c conda-forge python=3.10 pip ffmpeg
```

Inspect an existing `vmem` environment before using it. The installer refuses
other environment paths/names and refuses to reinstall after its success marker.
It never installs into `memcam`, `vbench`, or `dfot`.

The user reported `cuda/cuda-12.6.0` as available on Newton. Check the active
toolkit, not just the driver's CUDA version in `nvidia-smi`:

```bash
module list
module avail cuda
```

Use the available CUDA 12.6 toolkit and matching `cu126` wheel. From a login
shell, submit setup as follows (do not submit again inside a live GPU allocation):

```bash
export VMEM_CUDA_MODULE=cuda/cuda-12.6.0
export VMEM_CUDA_WHEEL=cu126
sbatch slurm/newton_vmem_setup.sbatch
```

The job loads that module without purging the module environment and requires
matching `nvcc`. If `VMEM_COMPILER_MODULE` is set, all Newton wrappers load that
compiler module before CUDA, including generation jobs that may need its runtime
libraries. The installer resolves explicit `CC` / `CXX` (otherwise `gcc` / `g++`
on PATH), then compiles/links/runs a C++17 probe and compiles a small CUDA source
with the same `CC` that PyTorch passes to nvcc. This runs before any pip install;
compiler paths/versions are saved in `setup.json`. It is not a CUDA kernel smoke.
The installer then installs Torch 2.7.0 / torchvision 0.22.0 first, keeps NumPy
1.24.4, installs the remaining repo requirements under the Newton compatibility
constraints, then
builds curope with `--no-build-isolation`. The extension imports Torch at build
time and supplies its own architecture flags; no claim of a single-architecture
build or fixed build time is made. Compiler errors are not hidden.
These Torch/CUDA wheel combinations are listed by
[PyTorch](https://pytorch.org/get-started/previous-versions/).

The constraints fix the Hugging Face dependency family and YAML version, not
every transitive dependency. This is **not yet a validated fully pinned Newton
environment**. The installer records commands, commit, installer hash, nvcc,
constraint copies/hashes, the runtime pip install report (if resolution succeeds),
`pip check`, `pip-freeze.txt`, and `resolved-pins.txt` in a unique
`$HOME/vmem_results/environment/setup_*` directory. Only a successful record
is suitable for replay; the Python installer accepts `--constraints` pointing
to that record's resolved pins, in addition to the compatibility constraints.
Conflicting replay pins fail resolution instead of overriding the baseline.
Keep its matching Torch wheel selection and
rebuild the editable extension. Do not edit a working environment during jobs.

Installation runs a real allocated CUDA matrix multiplication, curope forward /
inverse CUDA kernels, VMem/CUT3R imports, PNG/MP4 I/O, and the CPU unit tests.
No model generation occurs. The success marker is
`$HOME/.conda/envs/vmem/vmem_setup_complete.json`; setup logs are
`vmem_setup_JOB_ID.{out,err}`. An interrupted install is not a success.

### Dependency-resolution retry, 2026-09-25

The user-reported attempt `setup_20260925T152739_195226Z` failed in the runtime
pip install, before curope compilation or generation. Pip backtracked into
`ruamel.yaml==0.15.77`, whose build script raised `NameError: Str` on Python 3.10.
There was also a concrete incompatible candidate family: the selected
Transformers required safetensors >=0.8.0, while `huggingface-hub[torch]`
enabled its NumPy extra. [Safetensors 0.8.0 metadata](https://raw.githubusercontent.com/huggingface/safetensors/v0.8.0/bindings/python/pyproject.toml)
requires NumPy >=1.24.6 for that extra, conflicting with VMem's 1.24.4 pin.

The Newton-only constraints now use safetensors 0.6.2, Transformers 4.51.3,
Hub 0.34.4, Diffusers 0.35.1, Gradio 5.49.1 and ruamel.yaml 0.18.6. Their
declared dependency ranges were inspected; this is not a claim that installation
or inference has passed. Safetensors and ruamel.yaml must have wheels, so pip
cannot fall back to the failing old YAML source build. Other dependencies still
resolve normally, and `pip check` plus the existing CUDA/import/tests remain
mandatory. No `--no-deps` bypass or change to the shared `requirements.txt` is used.
Sources: [safetensors](https://raw.githubusercontent.com/huggingface/safetensors/v0.6.2/bindings/python/pyproject.toml),
[Transformers](https://raw.githubusercontent.com/huggingface/transformers/v4.51.3/setup.py),
[Hub](https://raw.githubusercontent.com/huggingface/huggingface_hub/v0.34.4/setup.py),
[Diffusers](https://raw.githubusercontent.com/huggingface/diffusers/v0.35.1/setup.py),
[Gradio](https://raw.githubusercontent.com/gradio-app/gradio/gradio@5.49.1/requirements.txt),
[ruamel.yaml](https://pypi.org/project/ruamel.yaml/0.18.6/).

After publishing/pulling this patch, retry in the existing dedicated environment.
Keep the failed record, do not replay its partial `resolved-pins.txt`, and do not
delete the environment or reinstall Torch by hand. In a still-live Slurm GPU
shell (the wrapper checks the allocation/mask):

```bash
(
  set -euo pipefail
  cd "$HOME/vmem"
  module unload cuda/cuda-13.1.0
  module load cuda/cuda-12.6.0
  hash -r
  export VMEM_CUDA_MODULE=cuda/cuda-12.6.0 VMEM_CUDA_WHEEL=cu126
  export VMEM_PYTHON="$HOME/.conda/envs/vmem/bin/python"
  export VMEM_FETCH_WEIGHTS=no
  mkdir -p "$HOME/vmem_results/environment"
  LOG="$HOME/vmem_results/environment/setup_retry_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S).log"
  bash slurm/newton_vmem_setup.sbatch 2>&1 | tee "$LOG"
)
```

This runs setup in the current allocation, not a nested job. It does not launch
videos. If the allocation expired, use the `sbatch` setup command above instead.
Local validation of this patch is static only; no install or tests run on the Mac.

### Compiler failure after dependency installation

The user-reported retry `setup_20260925T154155_717102Z` successfully installed
the Python runtime dependencies, then failed compiling curope. Its nvcc host
compiler could not execute `cc1plus`; the separate `c++` compile hit PyTorch's
explicit GCC >=9 check. The CUDA 12.6 toolkit alone does not provide a suitable
host C++ toolchain. Do not undo the successful dependency installation, suppress
compiler version checks, or recreate the environment.

Inspect available compiler modules, without guessing a site-specific name:

```bash
module -t avail 2>&1 | grep -Ei 'gcc|gnu|llvm'
command -v gcc g++ c++
printf 'CC=%s CXX=%s\n' "${CC:-unset}" "${CXX:-unset}"
```

Choose an available compatible GCC toolchain (11 or 12 are within
[CUDA 12.6's supported range](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-installation-guide-linux/index.html#host-compiler-support-policy)),
load it, and set `CC` and `CXX` to that module's `gcc` and `g++` executable paths.
Set `VMEM_COMPILER_MODULE` to the actual module name for subsequent jobs, then
retry the same setup wrapper. Existing satisfied packages are reused; compiler
and import/kernel checks still must pass. The preflight reproduces both build
routes because [PyTorch 2.7 BuildExtension](https://github.com/pytorch/pytorch/blob/v2.7.0/torch/utils/cpp_extension.py)
uses `CC` for nvcc's `-ccbin`, independently of `CXX`.

No successful curope build, complete setup, or Newton generation is established
by these two failed attempts. Compiler-preflight regression tests were added
but not run on the Mac.

### Download Weights

After setup succeeds, authenticate interactively with access approval for
`liguang0115/vmem`, then cache and hash all four generation dependencies:

```bash
export HF_HOME="$HOME/hf_cache"
"$HOME/.conda/envs/vmem/bin/hf" auth login
"$HOME/.conda/envs/vmem/bin/python" scripts/setup_vmem_newton.py \
  --weights-only --output "$HOME/vmem_results/environment"
```

This downloads VMem, CUT3R, the actual VAE subfolder, and the configured OpenCLIP
weights, without instantiating a model or generating on the login node. It
records cache paths/hashes. Use a permitted data-transfer/compute shell if the
site restricts large downloads on login nodes. Never paste tokens into scripts
or logs. Older Hugging Face installations may expose `huggingface-cli login`
instead of `hf auth login`; use the executable supplied by this environment.

## 3. Allocated Smoke and Matched Pair

The setup already includes the kernel/import smoke. Repeat it on another node
or after an environment change with:

```bash
sbatch slurm/newton_vmem_gpu_smoke.sbatch
```

After it passes and weights are cached, the single matched-pilot command is:

```bash
sbatch slurm/newton_vmem_results.sbatch
```

It performs another allocation smoke, verifies but does not change settings, creates
the new experiment lock once, runs CPU tests, generates the two 18-action videos,
validates artifacts and pre-eviction pixels, and produces:

```text
$HOME/vmem_results/newton_smoke_v1/
  experiment_lock.json
  generation/                 every attempt, raw traces, PNGs, checkpoints
  reporting/workflow_status.json
  reporting/newton_profile.json
  reporting/report/index.html
  reporting/report/resources.csv
  reporting/report/resource_steps.csv
  reporting/results.zip
```

The profile requires both videos to decode to 73 frames, matching provenance,
an equal hash-verified pre-eviction prefix, legal retrieval, measured action
times, and final resident payload counts of 73 versus 32 with no pending
evictions. The report includes both videos, resource curves, warm-up-separated
phase latency, action timing, allocator peaks, sampled RSS and commanded-revisit
self-consistency. VBench scores are explicitly **not requested** for this smoke.

The constant late-action-rate projection in the JSON is arithmetic guidance,
not an ETA: it excludes checkpoint/export overhead, lost work, and growth in
unbounded reconstruction. Short-run peaks do not certify 195-action memory fit.

## 4. Approve Longer Work Only After Profiling

Inspect `newton_profile.json` and the videos first. Then explicitly approve the
60-second Oxford pair:

```bash
VMEM_MODE=oxford60 VMEM_APPROVE_LONG=yes sbatch slurm/newton_vmem_results.sbatch
```

The gate revalidates the actual smoke artifacts, current generation source /
config hashes and Python package identity, and checks that long Oxford settings
match apart from IDs/duration. It does not auto-approve from a favorable metric.
The long workflow runs six existing VBench-Long dimensions sequentially using
`$HOME/.conda/envs/vbench/bin/python` and `$HOME/VBench`. It does not install or
modify that environment. Failed metrics retain completed generation and their
own logs. No video truncation, hidden batching/aggregation change or automatic
precision reduction is applied to make a metric fit.

Full 15-case study, only when separately approved:

```bash
VMEM_MODE=suite60 VMEM_APPROVE_LONG=yes sbatch slurm/newton_vmem_results.sbatch
```

Both long modes share `$HOME/vmem_results/transfer_v3/generation` and its new
lock. Their reporting folders are `oxford60_reporting` / `suite60_reporting`.
Validated Oxford generation is reused by the suite. Metric reports are separate.
The complete 30-video study is not promised to finish in one allocation.

## Scheduling, Signals and Resumption

Defaults use the user-reported `highgpu` partition and typed
`gpu:nvidia_h100_80gb_hbm3:1`, 8 CPUs, 96G host RAM, four hours for results jobs.
This inventory has not been checked live. If choosing `normal`, change **both**
partition and GRES type to its listed `gpu:nvidia_h100_pcie:1`; do not mix types.
Do not run nested `srun` inside an allocated shell.

The controller uses foreground `run --gpu inherit`; it rejects detached start
and physical GPU selection inside Slurm. GPU children preserve the scheduler's
mask exactly and use logical device 0. A same-process CUDA kernel check precedes
generation/evaluation, preventing the runner's CPU fallback on a broken
allocation. Admission checks CUDA free/total bytes on that device, not all node
GPUs or NVML index 0. Slurm can remap indices inside cgroups; see
[Slurm GRES](https://slurm.schedmd.com/gres.html).

`VMEM_MIN_FREE_MIB=70000` is an explicit H100 admission threshold, not a promise
of future fit. Thresholds above total capacity or above current free memory fail
promptly. No other process is terminated. A node-level CUDA failure should be
reported with job/node logs to ARCC rather than hidden by changing model code.

`--signal=B:USR1@600` warns the foreground controller before the time limit.
It stops only its child process group, marks `paused`, and exits 75. The latest
**already committed** recovery checkpoint survives; this does not force a new
checkpoint from the interrupted action. SIGTERM/interactive cancellation records
`cancelled` with exit 130 when cleanup is possible. SIGKILL/node failure may leave
stale running status; Slurm accounting and artifact validation remain necessary.
Signal delivery details follow [Slurm sbatch](https://slurm.schedmd.com/sbatch.html).

Resubmit the **same command/mode/root** after the job has stopped. There is no
automatic requeue loop. The controller locks both reporting and generation roots,
selects the latest compatible atomic checkpoint, writes a new attempt, and only
skips completed runs after validation. An attempt with no checkpoint or invalid
provenance stops for review. Do not erase locks or failed attempts to bypass this.
Disk space must accommodate PNGs, checkpoints, atomic replacements and resumed
copies. An OOM is not solved by blindly resubmitting at the same memory size.

Cross-node allocation details are recorded in events and stage logs. Package
identity is frozen; a resume on new hardware is not guaranteed bitwise equivalent
to uninterrupted execution and is not an uninterrupted timing measurement.

## Monitor and Download

Replace `JOB_ID` with sbatch's returned numeric ID:

```bash
squeue --me
sacct -j JOB_ID -X --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
tail -n 80 -F vmem_results_JOB_ID.out vmem_results_JOB_ID.err
jq . "$HOME/vmem_results/newton_smoke_v1/reporting/workflow_status.json"
```

Use `scancel JOB_ID` to cancel. Ctrl-C on `tail` only stops viewing. The local
PID on a login node is not the compute-node controller PID. A ZIP or empty queue
does not imply success: require `complete`, validated inventory and profile.

On the Mac, after the smoke finishes:

```bash
scp ab575577@newton.ist.ucf.edu:vmem_results/newton_smoke_v1/reporting/results.zip \
  "$HOME/Downloads/vmem_newton_smoke.zip"
```

For the long Oxford result use
`vmem_results/transfer_v3/oxford60_reporting/results.zip`. Unzip and open
`report/index.html`; no server is required.

## Remaining Evidence

Pending user execution: dependency resolution/build, Newton CPU/GPU smoke logs,
H100 matched generation, Slurm timeout/resume integration, visual report review,
actual throughput/peaks and quality results. Added CPU tests cover mask isolation,
foreground guards, admission thresholds, subprocess wrapping, unchanged protocol
prefix, requirement splitting and profile accounting. They were not run here.

B32 bounds post-update resident frame payload counts, not all surfels, histories,
transient reconstruction, total RAM/VRAM, or disk. The study remains an end-to-end
controller transfer test with constrained revisits, not sustained exploration,
upstream equivalence or proof of superiority to other matched-budget policies.
