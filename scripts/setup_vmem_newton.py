"""Install only in a dedicated Newton vmem environment; execute on an allocation."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
COMPATIBILITY_CONSTRAINTS = REPO / "requirements-newton-constraints.txt"


def runtime_requirements(text):
    excluded = {"torch==2.7.0", "torchvision==0.22.0",
                "--extra-index-url https://download.pytorch.org/whl/nightly/cu124",
                "-e ./extern/CUT3R/src/croco/models/curope"}
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not excluded.issubset(lines):
        raise ValueError("requirements.txt changed; review Torch/index/extension split before installing")
    result = [line for line in lines if line not in excluded]
    if any(line.startswith("-") for line in result):
        raise ValueError("Unreviewed pip option in runtime requirements")
    return "\n".join(result) + "\n"


def command(args, **kwargs):
    print(shlex.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def stage_constraints(output, replay=None):
    """Use immutable per-attempt copies for installation and provenance."""
    sources = [(COMPATIBILITY_CONSTRAINTS, "vmem-pins.txt")]
    if replay is not None:
        sources.append((Path(replay).resolve(), "replay-pins.txt"))
    arguments, records = [], []
    for source, name in sources:
        contents = source.read_bytes()
        target = output / name
        target.write_bytes(contents)
        arguments.extend(["-c", target])
        records.append({"source": str(source), "snapshot": str(target),
                        "sha256": hashlib.sha256(contents).hexdigest()})
    return arguments, records


def fetch_weights(output):
    from huggingface_hub import hf_hub_download, snapshot_download
    from open_clip.pretrained import get_pretrained_cfg, download_pretrained
    from omegaconf import OmegaConf
    from run_vmem_results import digest, save
    config = OmegaConf.load(REPO / "configs/inference/inference.yaml")
    files = {"vmem": hf_hub_download(config.model.model_path, "vmem_weights.pth"),
             "cut3r": hf_hub_download(config.surfel.model_path, "cut3r_512_dpt_4_64.pth")}
    vae = Path(snapshot_download("Manojb/stable-diffusion-2-1-base", allow_patterns=["vae/*"]))
    for path in sorted((vae / "vae").rglob("*")):
        if path.is_file():
            files["vae/" + path.name] = str(path)
    if "vae/config.json" not in files or not any(
            key.startswith("vae/") and key.endswith((".bin", ".safetensors")) for key in files):
        raise ValueError("VAE snapshot is missing its config or weights")
    cfg = get_pretrained_cfg("ViT-H-14", "laion2b_s32b_b79k")
    if not cfg:
        raise ValueError("OpenCLIP lacks the VMem pretrained configuration")
    files["clip"] = download_pretrained(cfg)
    save(output, {key: {"path": str(path), "sha256": digest(path)} for key, path in files.items()})


def main():
    from run_vmem_results import exclusive, save, stamp, digest
    from vmem_slurm import allocation
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cuda", choices=("cu126", "cu128"))
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--constraints", type=Path, help="Prior successful resolved-pins.txt for replay")
    cli.add_argument("--fetch-weights", action="store_true")
    cli.add_argument("--weights-only", action="store_true", help="Download/hash weights only; no install, model allocation or GPU work")
    args = cli.parse_args()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    expected_prefix = Path(os.environ.get("VMEM_ENV", str(Path.home() / ".conda/envs/vmem"))).resolve()
    if Path(sys.prefix).resolve() != expected_prefix or expected_prefix.name != "vmem":
        raise ValueError("Refusing installation outside the dedicated VMEM_ENV (directory name must be vmem)")
    if sys.version_info[:2] != (3, 10):
        raise ValueError("VMem setup requires Python 3.10")
    if args.weights_only:
        output = args.output.resolve() / ("weights_" + stamp() + ".json")
        fetch_weights(output)
        print(f"Cached checkpoint paths/hashes: {output}", flush=True)
        return
    allocation()
    if args.cuda is None:
        raise ValueError("Installation requires --cuda cu126 or cu128 matching nvcc")
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise ValueError("Load an available CUDA 12.6 or 12.8 toolkit module first; do not guess a module name")
    toolkit = command([nvcc, "--version"], capture_output=True, text=True).stdout
    release = re.search(r"release (\d+\.\d+)", toolkit)
    required = {"cu126": "12.6", "cu128": "12.8"}[args.cuda]
    if release is None or release.group(1) != required:
        raise ValueError(f"Selected {args.cuda} requires matching nvcc {required}; got {toolkit}")
    os.environ["CUDA_HOME"] = str(Path(nvcc).resolve().parent.parent)
    os.environ.setdefault("MAX_JOBS", os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    output = args.output.resolve() / ("setup_" + stamp())
    output.mkdir(parents=True)
    with exclusive(expected_prefix / ".vmem_setup.lock"):
        ready = expected_prefix / "vmem_setup_complete.json"
        if ready.exists():
            raise ValueError(f"Environment already recorded at {ready}; run smoke, do not reinstall under active jobs")
        pip = [sys.executable, "-m", "pip"]
        record = {"status": "installing", "python": sys.executable, "cuda_wheel": args.cuda,
                  "nvcc": toolkit, "allocation": allocation(), "requirements_sha256": digest(REPO / "requirements.txt"),
                  "installer_sha256": digest(Path(__file__)),
                  "git_commit": command(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
        save(output / "setup.json", record)
        try:
            constraint_args, record["constraints"] = stage_constraints(output, args.constraints)
            runtime = output / "runtime-requirements.txt"
            runtime.write_text(runtime_requirements((REPO / "requirements.txt").read_text()))
            save(output / "setup.json", record)
            command([*pip, "install", "torch==2.7.0", "torchvision==0.22.0", *constraint_args,
                     "--index-url", "https://download.pytorch.org/whl/" + args.cuda])
            command([sys.executable, REPO / "scripts/vmem_slurm.py", "--gpu-only",
                     "--output", output / "cuda.json"], timeout=180)
            command([*pip, "install", *constraint_args,
                     "--only-binary=ruamel.yaml,safetensors", "--report", output / "runtime-install.json",
                     "setuptools", "wheel", "packaging", "ninja", "-r", runtime])
            command([*pip, "install", *constraint_args, "--no-build-isolation", "-e",
                     REPO / "extern/CUT3R/src/croco/models/curope"])
            command([*pip, "check"])
            command([sys.executable, REPO / "scripts/vmem_slurm.py", "--output", output / "gpu_import_smoke.json"], timeout=300)
            command([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
            if args.fetch_weights:
                fetch_weights(output / "weights.json")
            record.update(status="installed_and_smoke_passed", weights_cached=args.fetch_weights,
                          cuda_build=__import__("torch").version.cuda)
        except BaseException as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            result = subprocess.run([*pip, "freeze", "--all"], capture_output=True, text=True)
            (output / "pip-freeze.txt").write_text(result.stdout + result.stderr)
            pins = sorted({f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()
                           if d.metadata.get("Name") and d.metadata["Name"].lower() != "curope"})
            (output / "resolved-pins.txt").write_text("\n".join(pins) + "\n")
            save(output / "setup.json", record)
            print(json.dumps({"setup_record": str(output), "status": record["status"]}), flush=True)
        save(ready, {"record": str(output / "setup.json"), **record})


if __name__ == "__main__":
    main()
