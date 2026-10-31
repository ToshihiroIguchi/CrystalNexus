"""Finite-strain elastic tensor and polycrystalline moduli from an MLIP.

``compute_elastic`` is a synchronous, model-agnostic routine: it only needs an
ASE calculator that provides forces and stress. It must be run on the MLIP
executor thread by the caller. ``elastic_moduli`` is a pure function of the
6x6 stiffness matrix so it can be unit-tested without any model.
"""
import time

import numpy as np

from .base import ase_stress_to_gpa
from .executor import abort_requested

# Voigt order (same as ASE): xx, yy, zz, yz, xz, xy
_VOIGT_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
N_STRAINED_CELLS = 12
# Force tolerance (eV/A) for the fixed-cell ion relaxation in each strained cell.
# It must be tight: for soft modes (e.g. ferroelectrics) the strain-induced
# forces are often below a loose threshold such as 0.01 eV/A, which would skip
# relaxation entirely and return a clamped-ion (overestimated) stiffness.
ELASTIC_ION_FMAX = 0.001
ELASTIC_ION_MAX_STEPS = 500
DEFAULT_STRAIN = 0.0025  # fraction (0.25 %)
# Strain-consistency criterion: diagonal C_ii at delta vs delta/2 may differ by
# at most max(CONSISTENCY_ABS_GPA, CONSISTENCY_REL * |C_ii|).
CONSISTENCY_ABS_GPA = 3.0
CONSISTENCY_REL = 0.05


class ElasticTimeout(Exception):
    """Raised when the wall-clock deadline is exceeded."""


def _strain_matrix(component, delta):
    """3x3 symmetric strain for one Voigt component (engineering shear)."""
    i, j = _VOIGT_PAIRS[component]
    eps = np.zeros((3, 3))
    if i == j:
        eps[i, i] = delta
    else:
        eps[i, j] = eps[j, i] = delta / 2.0
    return eps


def _tensor_at_strain(atoms, calculator, strain, fmax, max_steps, deadline,
                      progress_cb, offset, total):
    """Raw symmetrized 6x6 stiffness (GPa) from 12 strained cells of magnitude `strain`.

    Returns (C, n_ion_relax_steps, n_unconverged_cells). Cell numbering for
    progress/timeout messages starts after `offset` and runs up to `total`.
    """
    from ase.optimize import FIRE

    base_cell = np.array(atoms.get_cell())
    total_steps = 0
    n_unconverged = 0
    counter = offset
    stress_gpa = {}  # (component, sign) -> Voigt-6 stress in GPa

    for j in range(6):
        for sign in (+1, -1):
            counter += 1
            if progress_cb is not None:
                progress_cb(counter, total)
            if abort_requested():
                raise ElasticTimeout(f"aborted before strained cell {counter}/{total}")
            if deadline is not None and time.monotonic() > deadline:
                raise ElasticTimeout(
                    f"time limit reached before strained cell {counter}/{total}")
            eps = _strain_matrix(j, sign * strain)
            a = atoms.copy()
            a.calc = calculator
            a.set_cell(base_cell @ (np.eye(3) + eps), scale_atoms=True)

            opt = FIRE(a, logfile=None)

            def _check_deadline(counter=counter):
                if abort_requested():
                    raise ElasticTimeout(
                        f"aborted during strained cell {counter}/{total}")
                if deadline is not None and time.monotonic() > deadline:
                    raise ElasticTimeout(
                        f"time limit reached during strained cell {counter}/{total}")

            opt.attach(_check_deadline, interval=1)
            opt.run(fmax=fmax, steps=max_steps)
            total_steps += int(opt.get_number_of_steps())
            max_force = float(np.linalg.norm(a.get_forces(), axis=1).max())
            if max_force > fmax:
                n_unconverged += 1
            stress_gpa[(j, sign)] = np.asarray(
                ase_stress_to_gpa(a.get_stress(voigt=True)), dtype=float)

    C = np.zeros((6, 6))
    for j in range(6):
        C[:, j] = (stress_gpa[(j, +1)] - stress_gpa[(j, -1)]) / (2.0 * strain)
    return 0.5 * (C + C.T), total_steps, n_unconverged


def strain_consistency(C_full, C_half):
    """Compare two stiffness matrices computed at strain delta and delta/2.

    Inconsistent if any diagonal element C_ii differs by more than
    max(CONSISTENCY_ABS_GPA, CONSISTENCY_REL * max(|C_ii|, |C_ii'|)).
    Returns (max_abs_diff_GPa, worst_component, consistent) where the worst
    component is the one with the largest absolute difference.
    """
    d_full = np.diag(np.asarray(C_full, dtype=float))
    d_half = np.diag(np.asarray(C_half, dtype=float))
    diff = np.abs(d_full - d_half)
    tol = np.maximum(CONSISTENCY_ABS_GPA,
                     CONSISTENCY_REL * np.maximum(np.abs(d_full), np.abs(d_half)))
    k = int(np.argmax(diff))
    return float(diff[k]), f"C{k + 1}{k + 1}", bool(np.all(diff <= tol))


def compute_elastic(atoms, calculator, strain=DEFAULT_STRAIN, fmax=ELASTIC_ION_FMAX,
                    max_steps=ELASTIC_ION_MAX_STEPS,
                    deadline=None, progress_cb=None, check_consistency=True):
    """Compute the 6x6 elastic tensor (GPa) of an already relaxed structure.

    Args:
        atoms: ASE Atoms (relaxed). It is not modified.
        calculator: ASE calculator with forces and stress.
        strain: strain magnitude (fraction, e.g. 0.0025 = 0.25 %).
        fmax: force tolerance (eV/A) for the fixed-cell ion relaxations. The
            default is deliberately tight (ELASTIC_ION_FMAX): with soft modes,
            strain-induced forces below a loose threshold would skip the
            relaxation and give clamped-ion values.
        max_steps: maximum FIRE steps per strained cell.
        deadline: ``time.monotonic()`` value after which ElasticTimeout is raised
            (applies across both passes).
        progress_cb: called as ``progress_cb(k, total)`` before strained cell k
            (1-based); total is 24 with the consistency check, else 12.
        check_consistency: also compute the tensor at strain/2 and compare.

    Returns a dict with C_GPa, moduli, stable, min_eigenvalue_GPa, strain,
    n_strained_cells, n_ion_relax_steps (summed over passes),
    n_unconverged_cells and consistency (None when disabled). The primary
    result is always the tensor at ``strain``.
    """
    if strain <= 0:
        raise ValueError("strain must be positive")

    total = 2 * N_STRAINED_CELLS if check_consistency else N_STRAINED_CELLS
    C, steps, unconv = _tensor_at_strain(
        atoms, calculator, strain, fmax, max_steps, deadline, progress_cb, 0, total)

    consistency = None
    if check_consistency:
        C_half, steps2, unconv2 = _tensor_at_strain(
            atoms, calculator, strain / 2.0, fmax, max_steps, deadline,
            progress_cb, N_STRAINED_CELLS, total)
        steps += steps2
        unconv += unconv2
        max_diff, worst, ok = strain_consistency(C, C_half)
        consistency = {
            "strain_half": float(strain / 2.0),
            "max_abs_diff_GPa": max_diff,
            "consistent": ok,
            "worst_component": worst,
        }

    result = elastic_moduli(C)
    result.update({
        "strain": float(strain),
        "n_strained_cells": total,
        "n_ion_relax_steps": steps,
        "n_unconverged_cells": unconv,
        "consistency": consistency,
    })
    return result


def _kg_derived(K, G):
    """Young's modulus and Poisson's ratio from bulk and shear moduli."""
    denom = 3.0 * K + G
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return None, None
    E = 9.0 * K * G / denom
    nu = (3.0 * K - 2.0 * G) / (2.0 * denom)
    return float(E), float(nu)


def elastic_moduli(C):
    """Polycrystalline moduli and stability from a 6x6 stiffness matrix (GPa).

    Returns dict(C_GPa, moduli{K,G,E,nu: {voigt, reuss, hill}}, stable,
    min_eigenvalue_GPa). Reuss (and therefore Hill) values are None when C is
    singular or not positive definite.
    """
    C = np.asarray(C, dtype=float)
    if C.shape != (6, 6):
        raise ValueError("C must be a 6x6 matrix")
    C = 0.5 * (C + C.T)

    eig = np.linalg.eigvalsh(C)
    min_eig = float(eig.min())
    stable = bool(min_eig > 0)

    c11, c22, c33 = C[0, 0], C[1, 1], C[2, 2]
    c12, c13, c23 = C[0, 1], C[0, 2], C[1, 2]
    c44, c55, c66 = C[3, 3], C[4, 4], C[5, 5]
    K_v = (c11 + c22 + c33 + 2.0 * (c12 + c13 + c23)) / 9.0
    G_v = (c11 + c22 + c33 - (c12 + c13 + c23) + 3.0 * (c44 + c55 + c66)) / 15.0

    K_r = G_r = None
    if stable:
        try:
            S = np.linalg.inv(C)
            s_diag = S[0, 0] + S[1, 1] + S[2, 2]
            s_off = S[0, 1] + S[0, 2] + S[1, 2]
            K_r = 1.0 / (s_diag + 2.0 * s_off)
            G_r = 15.0 / (4.0 * s_diag - 4.0 * s_off
                          + 3.0 * (S[3, 3] + S[4, 4] + S[5, 5]))
            if not (np.isfinite(K_r) and np.isfinite(G_r)):
                K_r = G_r = None
        except np.linalg.LinAlgError:
            K_r = G_r = None

    if K_r is not None:
        K_h, G_h = 0.5 * (K_v + K_r), 0.5 * (G_v + G_r)
    else:
        K_h = G_h = None

    def _entry(K, G):
        if K is None or G is None:
            return None, None, None, None
        E, nu = _kg_derived(float(K), float(G))
        return float(K), float(G), E, nu

    cols = {"voigt": _entry(K_v, G_v), "reuss": _entry(K_r, G_r),
            "hill": _entry(K_h, G_h)}
    moduli = {name: {scheme: cols[scheme][idx] for scheme in cols}
              for idx, name in enumerate(("K", "G", "E", "nu"))}

    return {
        "C_GPa": C.tolist(),
        "moduli": moduli,
        "stable": stable,
        "min_eigenvalue_GPa": min_eig,
    }
