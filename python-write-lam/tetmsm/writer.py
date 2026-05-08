"""tetmsm.writer

Writes LAMMPS data files for `bond_style tetmsm` + `improper_style tetmsm`.

- Atom style: `hybrid full sphere`
- Pipeline writes topology only:
  - Bonds = unique mesh edges (1 bond type)
  - Impropers = tetrahedra (1 improper type per elastic material region)

Material parameters (G, nu) are set via `improper_coeff` in the LAMMPS input
script, not in the data file.

See: DATAFILE_INSTRUCTIONS.md
"""

from __future__ import annotations

import numpy as np


def write_lam(list_of_composites, filename: str) -> str:
    """Write a LAMMPS `.lam` file for the current (single) tetmsm model.

    Parameters
    ----------
    list_of_composites
        List of `Composite` objects (usually length 1).
    filename
        Output path. If it does not end with `.lam`, `.lam` is appended.

    Returns
    -------
    str
        The output path without the `.lam` extension (for backward
        compatibility with older callers).
    """
    if not list_of_composites:
        raise ValueError("list_of_composites must be non-empty")

    if filename.endswith(".lam"):
        outpath = filename
    else:
        outpath = f"{filename}.lam"

    outbase = outpath[:-4] if outpath.endswith(".lam") else outpath

    V_into_xyz = []
    V_into_mol_ID = []
    V_into_atom_type = []
    V_into_q = []
    V_into_diam = []
    V_into_rho = []

    bonds_all = []
    impropers_all = []

    rigid_id_all = []

    mol_off = 0
    atom_off = 0
    imp_type_off = 0
    rigid_off = 0

    for comp in list_of_composites:
        V_into_xyz.append(np.asarray(comp.V_into_xyz, dtype=float))
        V_into_atom_type.append(np.asarray(comp.V_into_atom_type, dtype=int))
        V_into_q.append(np.asarray(comp.V_into_q, dtype=float))
        V_into_diam.append(np.asarray(comp.V_into_diam, dtype=float))
        V_into_rho.append(np.asarray(comp.V_into_rho, dtype=float))

        # mol-ID (LAMMPS: molecule-ID)
        mol = np.asarray(comp.V_into_part_ID, dtype=int) + mol_off
        V_into_mol_ID.append(mol)
        mol_off += int(comp.n_parts)

        # Bonds: (type, i, j) with 0-indexed atom indices and 0-indexed types
        if getattr(comp, "bonds", None) is not None and len(comp.bonds) > 0:
            b = np.asarray(comp.bonds, dtype=int).copy()
            b[:, 1:] += atom_off
            bonds_all.append(b)

        # Impropers: (type, n1, n2, n3, n4) with 0-indexed atom indices
        if getattr(comp, "impropers", None) is not None and len(comp.impropers) > 0:
            imp = np.asarray(comp.impropers, dtype=int).copy()
            imp[:, 0] += imp_type_off
            imp[:, 1:] += atom_off
            impropers_all.append(imp)
            imp_type_off += int(getattr(comp, "n_improper_types", 0))

        # Optional rigid group IDs file
        if getattr(comp, "V_into_rigid_ID", None) is not None:
            rid = np.asarray(comp.V_into_rigid_ID, dtype=int).copy()
            # ensure elastic atoms have rigid-ID 0
            at = np.asarray(comp.V_into_atom_type, dtype=int)
            rid[np.isin(at, [1, 2])] = 0
            # keep rigid IDs unique across composites
            nonzero = rid != 0
            rid[nonzero] += rigid_off
            rigid_off = max(rigid_off, int(rid.max()) if rid.size else 0)
            rigid_id_all.append(rid)

        atom_off += int(comp.n_atoms)

    V_into_xyz = np.concatenate(V_into_xyz, axis=0)
    V_into_atom_type = np.concatenate(V_into_atom_type, axis=0)
    V_into_mol_ID = np.concatenate(V_into_mol_ID, axis=0)
    V_into_q = np.concatenate(V_into_q, axis=0)
    V_into_diam = np.concatenate(V_into_diam, axis=0)
    V_into_rho = np.concatenate(V_into_rho, axis=0)

    N = int(V_into_xyz.shape[0])

    bonds = np.concatenate(bonds_all, axis=0) if bonds_all else None
    impropers = np.concatenate(impropers_all, axis=0) if impropers_all else None

    NB = int(bonds.shape[0]) if bonds is not None else 0
    NI = int(impropers.shape[0]) if impropers is not None else 0

    n_atom_types = int(V_into_atom_type.max()) if V_into_atom_type.size else 0
    n_improper_types = int(imp_type_off)

    # Point particles: keep diameter as-is (default 0.0).
    # `model.py` stores per-atom mass in `V_into_rho` when diameter==0.
    # If you want LAMMPS to use that mass, set it in the input script, e.g.:
    #   variable m atom density
    #   set atom * mass v_m
    diam_out = V_into_diam.copy()
    dens_out = V_into_rho.copy()

    # Bounding box (pad by half-range on each side)
    x, y, z = V_into_xyz[:, 0], V_into_xyz[:, 1], V_into_xyz[:, 2]
    bounds = []
    for coords in (x, y, z):
        mn = float(np.min(coords))
        mx = float(np.max(coords))
        r = float(mx - mn)
        if r == 0.0:
            r = 1.0
        bounds.append((mn - 0.5 * r, mx + 0.5 * r))

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("LAMMPS data file for tetmsm\n\n")

        f.write(f"{N} atoms\n")
        f.write(f"{NB} bonds\n")
        f.write(f"{NI} impropers\n\n")

        f.write(f"{n_atom_types} atom types\n")
        f.write("1 bond types\n")
        f.write(f"{n_improper_types} improper types\n\n")

        (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds
        f.write(f"{xlo} {xhi} xlo xhi\n")
        f.write(f"{ylo} {yhi} ylo yhi\n")
        f.write(f"{zlo} {zhi} zlo zhi\n\n")

        # Masses: required by LAMMPS (placeholder values).
        # For point particles (diameter = 0), `model.py` stores per-atom mass in the
        # last column; set it in the input script if needed:
        #   variable m atom density
        #   set atom * mass v_m
        f.write("Masses\n\n")
        for t in range(1, n_atom_types + 1):
            f.write(f"{t} 1.0\n")
        f.write("\n")

        # Atoms: atom-ID atom-type x y z mol-ID q diameter rho
        f.write("Atoms  # hybrid full sphere\n\n")
        ids = (1 + np.arange(N, dtype=int))
        atoms = np.column_stack(
            (
                ids,
                V_into_atom_type,
                V_into_xyz,
                V_into_mol_ID,
                V_into_q,
                diam_out,
                dens_out,
            )
        )
        f.write(("%d %d %.16g %.16g %.16g %d %.16g %.16g %.16g\n" * N) % tuple(atoms.ravel()))
        f.write("\n")

        if bonds is not None and NB > 0:
            f.write("Bonds\n\n")
            b = bonds.copy()
            b[:, 0] += 1  # type
            b[:, 1:] += 1  # atom ids
            b_ids = (1 + np.arange(NB, dtype=int))[:, None]
            b_out = np.concatenate([b_ids, b], axis=1)
            f.write(("%d %d %d %d\n" * NB) % tuple(b_out.ravel()))
            f.write("\n")

        if impropers is not None and NI > 0:
            f.write("Impropers\n\n")
            imp = impropers.copy()
            imp[:, 0] += 1  # type
            imp[:, 1:] += 1  # atom ids
            imp_ids = (1 + np.arange(NI, dtype=int))[:, None]
            imp_out = np.concatenate([imp_ids, imp], axis=1)
            f.write(("%d %d %d %d %d %d\n" * NI) % tuple(imp_out.ravel()))
            f.write("\n")

    if rigid_id_all:
        rigid_ids = np.concatenate(rigid_id_all, axis=0)
        with open(outbase + "-rigid-ID.txt", "w", encoding="utf-8") as f:
            f.write(f"{N}\n")
            data = np.column_stack((1 + np.arange(N, dtype=int), rigid_ids))
            f.write(("%d %d\n" * N) % tuple(data.ravel()))

    print(f"Generated {outpath}")
    if rigid_id_all:
        print(f"Generated {outbase}-rigid-ID.txt")

    return outbase
