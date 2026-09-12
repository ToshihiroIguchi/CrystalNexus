"""Manual benchmark for main.run_relaxation() across structures/optimizers.

Not a test: it loads CHGNet and takes minutes. Run manually and paste the
markdown table into the PR description so speedups stay auditable.

Usage:
    venv\\Scripts\\python.exe scripts/bench_relax.py \
        [--sizes 2x2x2] [--optimizers LBFGS,FIRE] [--fmax 0.1] [--max-steps 100]
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main  # noqa: E402
from pymatgen.core import Structure  # noqa: E402

DEFAULT_STRUCTURES = [
    "Metals/Cu.cif",
    "Metals/Nd2Fe14B.cif",
    "Oxides/BaTiO3(tetragonal).cif",
    "Oxides/ZrO2.cif",
    "Oxides/Fe3O4.cif",
]


def _parse_size(size_str):
    """Parse "2x2x2" into (2, 2, 2)."""
    return tuple(int(x) for x in size_str.lower().split("x"))


def _final_max_force(trajectory):
    if trajectory is None or not getattr(trajectory, "forces", None):
        return None
    forces = np.array(trajectory.forces[-1])
    return float(np.linalg.norm(forces, axis=1).max())


def _final_energy(trajectory):
    if trajectory is None or not getattr(trajectory, "energies", None):
        return None
    return float(trajectory.energies[-1])


def run_benchmark(structure_paths, sizes, optimizers, fmax, max_steps):
    calculator = asyncio.run(main.chgnet_manager.get_calculator())

    rows = []
    for rel_path in structure_paths:
        cif_path = REPO_ROOT / "sample_cif" / rel_path
        if not cif_path.exists():
            print(f"# skipping {rel_path}: file not found", file=sys.stderr)
            continue

        for size in sizes:
            base_structure = Structure.from_file(cif_path)
            structure = base_structure.copy()
            if size != (1, 1, 1):
                structure.make_supercell(list(size))
            n_atoms = len(structure)
            size_str = "x".join(str(s) for s in size)

            for optimizer_name in optimizers:
                deadline = time.monotonic() + 3600  # effectively unbounded for a bench run
                start = time.monotonic()
                result = main.run_relaxation(
                    calculator, structure,
                    fmax=fmax, steps=max_steps, optimizer_name=optimizer_name,
                    deadline=deadline, relax_cell=True, progress_slot=None,
                )
                wall_s = time.monotonic() - start

                trajectory = result.get("trajectory")
                rows.append({
                    "structure": f"{rel_path} ({size_str})",
                    "n_atoms": n_atoms,
                    "optimizer": optimizer_name,
                    "frames": len(trajectory) if trajectory else 0,
                    "opt_steps": result.get("optimizer_steps", 0),
                    "wall_s": wall_s,
                    "final_E_eV": _final_energy(trajectory),
                    "final_maxF": _final_max_force(trajectory),
                })
                print(
                    f"# done: {rel_path} {size_str} {optimizer_name} "
                    f"({wall_s:.1f}s, {result.get('optimizer_steps', 0)} steps)",
                    file=sys.stderr,
                )
    return rows


def print_markdown_table(rows):
    header = ["structure", "n_atoms", "optimizer", "frames", "opt_steps",
              "wall_s", "final_E_eV", "final_maxF"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        final_e = f"{row['final_E_eV']:.4f}" if row["final_E_eV"] is not None else "N/A"
        final_f = f"{row['final_maxF']:.4f}" if row["final_maxF"] is not None else "N/A"
        print(
            f"| {row['structure']} | {row['n_atoms']} | {row['optimizer']} | "
            f"{row['frames']} | {row['opt_steps']} | {row['wall_s']:.1f} | "
            f"{final_e} | {final_f} |"
        )


def main_cli():
    parser = argparse.ArgumentParser(
        description="Benchmark main.run_relaxation() across structures/optimizers/sizes.")
    parser.add_argument("--sizes", default="2x2x2",
                         help="Comma-separated supercell sizes, e.g. '2x2x2,1x1x1'")
    parser.add_argument("--optimizers", default="LBFGS,FIRE",
                         help="Comma-separated ASE optimizer names")
    parser.add_argument("--fmax", type=float, default=0.1,
                         help="Force convergence threshold in eV/A")
    parser.add_argument("--max-steps", type=int, default=100,
                         help="Maximum optimizer steps")
    parser.add_argument("--structures", default=",".join(DEFAULT_STRUCTURES),
                         help="Comma-separated sample_cif-relative paths")
    args = parser.parse_args()

    sizes = [_parse_size(s) for s in args.sizes.split(",")]
    optimizers = [o.strip() for o in args.optimizers.split(",")]
    structure_paths = [s.strip() for s in args.structures.split(",")]

    rows = run_benchmark(structure_paths, sizes, optimizers, args.fmax, args.max_steps)
    print_markdown_table(rows)


if __name__ == "__main__":
    main_cli()
