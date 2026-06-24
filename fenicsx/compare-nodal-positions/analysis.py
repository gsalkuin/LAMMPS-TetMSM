from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REF_GEOMETRY = HERE.parent / "cube-N16.lam"


@dataclass(frozen=True)
class Case:
    name: str
    dump_path: Path
    fem_path: Path


CASES = (
    Case(
        name="msm_nu0p25",
        dump_path=HERE / "comp-clamp-N16-nu-0.25.dump",
        fem_path=HERE / "fem-comp-nu-0.450.data",
    ),
    Case(
        name="msm_nu0p45",
        dump_path=HERE / "comp-clamp-N16-nu-0.45.dump",
        fem_path=HERE / "fem-comp-nu-0.450.data",
    ),
)


def parse_lammps_data_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read atom ids and xyz positions from the Atoms section of a LAMMPS data file."""
    ids: list[int] = []
    positions: list[list[float]] = []
    in_atoms = False

    with path.open() as fh:
        for line in fh:
            stripped = line.strip()
            if not in_atoms:
                if stripped.startswith("Atoms"):
                    in_atoms = True
                continue

            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if not parts[0].isdigit():
                break

            ids.append(int(parts[0]))
            positions.append([float(parts[2]), float(parts[3]), float(parts[4])])

    if not ids:
        raise ValueError(f"No atoms found in {path}")

    return np.asarray(ids, dtype=int), np.asarray(positions, dtype=float)


def parse_lam_reference_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read atom ids and reference xyz positions from cube-N16.lam."""
    return parse_lammps_data_positions(path)


def parse_last_dump_frame(path: Path) -> tuple[int, np.ndarray, np.ndarray, list[str]]:
    """Read only the final frame from a LAMMPS custom dump."""
    timestep: int | None = None
    atom_columns: list[str] | None = None
    rows: list[list[str]] = []

    with path.open() as fh:
        for line in fh:
            stripped = line.strip()
            if stripped == "ITEM: TIMESTEP":
                timestep = int(next(fh).strip())
                atom_columns = None
                rows = []
                continue

            if stripped.startswith("ITEM: ATOMS"):
                atom_columns = stripped.split()[2:]
                continue

            if atom_columns is not None and stripped and not stripped.startswith("ITEM:"):
                rows.append(stripped.split())

    if timestep is None or atom_columns is None or not rows:
        raise ValueError(f"No dump frame found in {path}")

    col = {name: i for i, name in enumerate(atom_columns)}
    for required in ("id", "x", "y", "z"):
        if required not in col:
            raise ValueError(f"{path} is missing required dump column {required!r}")

    ids = np.asarray([int(row[col["id"]]) for row in rows], dtype=int)
    positions = np.asarray(
        [[float(row[col["x"]]), float(row[col["y"]]), float(row[col["z"]])] for row in rows],
        dtype=float,
    )

    return timestep, ids, positions, atom_columns


def order_by_id(ids: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(ids)
    return ids[order], values[order]


def require_same_ids(*id_arrays: np.ndarray) -> np.ndarray:
    first = id_arrays[0]
    for ids in id_arrays[1:]:
        if len(ids) != len(first) or not np.array_equal(ids, first):
            raise ValueError("Atom ids do not match between MSM, FEM, and reference files")
    return first


def write_last_frame_dump(path: Path, timestep: int, ids: np.ndarray, positions: np.ndarray) -> None:
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    pad = 1.0e-4

    with path.open("w") as fh:
        fh.write("ITEM: TIMESTEP\n")
        fh.write(f"{timestep}\n")
        fh.write("ITEM: NUMBER OF ATOMS\n")
        fh.write(f"{len(ids)}\n")
        fh.write("ITEM: BOX BOUNDS ss ss ss\n")
        for lo, hi in zip(mins - pad, maxs + pad):
            fh.write(f"{lo:.12g} {hi:.12g}\n")
        fh.write("ITEM: ATOMS id type x y z\n")
        for atom_id, xyz in zip(ids, positions):
            fh.write(f"{atom_id:d} 1 {xyz[0]:.12g} {xyz[1]:.12g} {xyz[2]:.12g}\n")


def write_error_dump(
    path: Path,
    timestep: int,
    ids: np.ndarray,
    positions: np.ndarray,
    errors: np.ndarray,
) -> None:
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    pad = 1.0e-4

    with path.open("w") as fh:
        fh.write("ITEM: TIMESTEP\n")
        fh.write(f"{timestep}\n")
        fh.write("ITEM: NUMBER OF ATOMS\n")
        fh.write(f"{len(ids)}\n")
        fh.write("ITEM: BOX BOUNDS ss ss ss\n")
        for lo, hi in zip(mins - pad, maxs + pad):
            fh.write(f"{lo:.12g} {hi:.12g}\n")
        fh.write("ITEM: ATOMS id type x y z error\n")
        for atom_id, xyz, err in zip(ids, positions, errors):
            fh.write(
                f"{atom_id:d} 1 {xyz[0]:.12g} {xyz[1]:.12g} {xyz[2]:.12g} {err:.12g}\n"
            )


def compare_case(case: Case, ref_ids: np.ndarray, ref_positions: np.ndarray) -> dict[str, float]:
    timestep, msm_ids, msm_positions, _ = parse_last_dump_frame(case.dump_path)
    fem_ids, fem_positions = parse_lammps_data_positions(case.fem_path)

    msm_ids, msm_positions = order_by_id(msm_ids, msm_positions)
    fem_ids, fem_positions = order_by_id(fem_ids, fem_positions)

    ids = require_same_ids(msm_ids, fem_ids, ref_ids)

    error_vectors = msm_positions - fem_positions
    error_norms = np.linalg.norm(error_vectors, axis=1)

    fem_displacement = fem_positions - ref_positions
    fem_displacement_norms = np.linalg.norm(fem_displacement, axis=1)
    max_fem_displacement = float(fem_displacement_norms.max())
    if max_fem_displacement == 0.0:
        raise ValueError("Cannot nondimensionalize errors: max FEM displacement is zero.")

    msm_displacement = msm_positions - ref_positions
    max_msm_displacement = float(np.linalg.norm(msm_displacement, axis=1).max())

    nondim_error_norms = error_norms / max_fem_displacement
    mean_error = float(nondim_error_norms.mean())
    max_error = float(nondim_error_norms.max())

    fem_displacement_l2 = float(np.linalg.norm(fem_displacement))
    relative_error = (
        float(np.linalg.norm(error_vectors) / fem_displacement_l2)
        if fem_displacement_l2
        else np.nan
    )

    write_last_frame_dump(HERE / f"{case.name}.dump", timestep, ids, msm_positions)
    write_error_dump(
        HERE / f"{case.name}_error.dump",
        timestep,
        ids,
        msm_positions,
        nondim_error_norms,
    )

    return {
        "timestep": float(timestep),
        "nodes": float(len(ids)),
        "mean_error": mean_error,
        "max_error": max_error,
        "max_fem_displacement": max_fem_displacement,
        "max_msm_displacement": max_msm_displacement,
        "relative_error": float(relative_error),
    }


def main() -> None:
    ref_ids, ref_positions = parse_lam_reference_positions(REF_GEOMETRY)
    ref_ids, ref_positions = order_by_id(ref_ids, ref_positions)

    for case in CASES:
        metrics = compare_case(case, ref_ids, ref_positions)
        print(f"\n{case.name}")
        print(f"  last timestep: {metrics['timestep']:.0f}")
        print(f"  nodes: {metrics['nodes']:.0f}")
        print(f"  FEM file: {case.fem_path}")
        print(f"  last-frame dump: {HERE / f'{case.name}.dump'}")
        print(f"  OVITO error dump: {HERE / f'{case.name}_error.dump'}")
        print(f"  nondimensional mean error: {metrics['mean_error']:.8e}")
        print(f"  nondimensional max error: {metrics['max_error']:.8e}")
        print(f"  max FEM displacement: {metrics['max_fem_displacement']:.8e}")
        print(f"  max MSM displacement: {metrics['max_msm_displacement']:.8e}")
        print(f"  relative L2 error: {metrics['relative_error']:.8e}")


if __name__ == "__main__":
    main()
