#!/usr/bin/env python
"""Pre-fetch the PET-MAD-DOS model weights used by the "Predict DOS" feature.

The running CrystalNexus server never downloads anything: it loads the files
this script puts into a project-local directory. Run it once at initial setup
(needs internet, ~145 MB):

    python scripts/download_dos_model.py            # download + verify
    python scripts/download_dos_model.py --check    # verify only (no network)

The files are fetched from the Hugging Face repo lab-cosmo/pet-mad-dos at a
pinned commit and verified against the SHA256 values below, so the weights are
reproducible. Only the Python standard library is used.

Target directory: $CRYSTALNEXUS_DOS_MODEL_DIR, default <project>/models/pet-mad-dos.
Exit code: 0 on success, 1 on any failure.
"""
import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

REPO_ID = "lab-cosmo/pet-mad-dos"
# Exact commit of the HF repo the manifest below was taken from.
REVISION = "7bb2f08a94bfff7dc745af48580f5abf09bd441b"

# local file name -> (path inside the HF repo, size in bytes, SHA256)
MANIFEST: Dict[str, tuple] = {
    "pet-mad-dos-v1.0.ckpt": (
        "models/pet-mad-dos-v1.0.ckpt", 134896386,
        "2faff6887ec5efbf09cb103032b867b1a11092765c54aa10781c3b7bcad8a23d"),
    "bandgap-model.pt": (
        "models/bandgap-model.pt", 4740530,
        "0bbaadb987efef04723765c9b4aa20e6fa04cd828b46dd766508d2babf735a25"),
    "fermi-model.pt": (
        "models/fermi-model.pt", 4740887,
        "8f4aa1bf1e313fb46090b499f45b4de9c4e3340f726dfd7855edb3362ebaabe4"),
}

ENV_VAR = "CRYSTALNEXUS_DOS_MODEL_DIR"
INSTALL_HINT = "python scripts/download_dos_model.py"


def default_model_dir() -> Path:
    env = os.getenv(ENV_VAR)
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent.parent / "models" / "pet-mad-dos"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_file(path: Path, size: int, sha256: str, full: bool = True) -> Optional[str]:
    """Return None if the file is valid, else a short problem description.

    full=False only compares the file size (fast; used by the server at
    start-of-request). full=True also verifies the SHA256.
    """
    if not path.is_file():
        return "missing"
    actual = path.stat().st_size
    if actual != size:
        return f"size mismatch (expected {size}, found {actual})"
    if full and sha256_of(path) != sha256:
        return "SHA256 mismatch"
    return None


def check_model_dir(model_dir: Path, full: bool = True,
                    manifest: Optional[Dict[str, tuple]] = None) -> List[str]:
    """Return a list of problems ('<file>: <problem>'); empty means all valid."""
    manifest = MANIFEST if manifest is None else manifest
    problems = []
    for name, (_, size, sha) in manifest.items():
        problem = check_file(Path(model_dir) / name, size, sha, full=full)
        if problem:
            problems.append(f"{name}: {problem}")
    return problems


def _format_mb(n: float) -> str:
    return f"{n / 1e6:.1f} MB"


def _download_one(url: str, dest: Path, size: int, sha256: str) -> None:
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    done = 0
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
            last_pct = -1
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                done += len(chunk)
                pct = int(done * 100 / size) if size else 100
                if pct != last_pct and (pct % 5 == 0 or done == size):
                    last_pct = pct
                    print(f"\r    {_format_mb(done)} / {_format_mb(size)} ({pct}%)",
                          end="", flush=True)
        print()
        if done != size:
            raise RuntimeError(f"size mismatch (expected {size}, got {done})")
        if h.hexdigest() != sha256:
            raise RuntimeError("SHA256 mismatch")
        os.replace(part, dest)
    finally:
        if part.exists():
            part.unlink()


def download(model_dir: Path) -> int:
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model directory: {model_dir}")
    print(f"Source: huggingface.co/{REPO_ID} @ {REVISION}")
    for name, (repo_path, size, sha) in MANIFEST.items():
        dest = model_dir / name
        if check_file(dest, size, sha, full=True) is None:
            print(f"[ok]   {name} already present and valid, skipping")
            continue
        url = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/{repo_path}"
        print(f"[get]  {name} ({_format_mb(size)})")
        try:
            _download_one(url, dest, size, sha)
        except urllib.error.URLError as e:
            print(f"\nERROR: could not reach huggingface.co ({e.reason}). "
                  "Check your internet connection / proxy, or copy the models "
                  "directory from another machine and run with --check.",
                  file=sys.stderr)
            return 1
        except OSError as e:
            print(f"\nERROR: failed to download {name}: {e}", file=sys.stderr)
            return 1
        except RuntimeError as e:
            print(f"\nERROR: {name}: {e}. The corrupted download was discarded.",
                  file=sys.stderr)
            return 1
        print(f"[done] {name} verified (SHA256 ok)")
    problems = check_model_dir(model_dir, full=True)
    if problems:
        print("ERROR: verification failed:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print("All PET-MAD-DOS model files are present and verified.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="verify existing files only (size + SHA256); no network access")
    parser.add_argument("--dir", type=Path, default=None,
                        help=f"target directory (default: ${ENV_VAR} or <project>/models/pet-mad-dos)")
    args = parser.parse_args(argv)
    model_dir = args.dir if args.dir else default_model_dir()

    if args.check:
        problems = check_model_dir(model_dir, full=True)
        if problems:
            print(f"PET-MAD-DOS model files in {model_dir} are NOT valid:\n  "
                  + "\n  ".join(problems), file=sys.stderr)
            print(f"Run: {INSTALL_HINT}", file=sys.stderr)
            return 1
        print(f"PET-MAD-DOS model files in {model_dir} are present and valid.")
        return 0
    return download(model_dir)


if __name__ == "__main__":
    sys.exit(main())
