"""Example: generate LAMMPS `.lam` files for the single tetmsm model."""

import numpy as np

from tetmsm.geometry import TetMesh
from tetmsm.model import Composite
from tetmsm.writer import write_lam

# Elastic constants (used in LAMMPS input script):

# material properties
E = 1.16e3
nu = 0.49
G = E / (2.0 * (1.0 + nu))

rho = 2.01

# dimensions
L = 25.8
W = 1.21
h = 0.49

I = W * h**3 / 12
A = W * h

# magnetic properties
Mr = 0.941
dBmax = 0.1437

M_vec = np.array([Mr, 0, 0])

print(f"E={E:g} Pa, nu={nu:g} -> G={G:g} Pa")
print(f"LAMMPS: improper_coeff * {G:.16g} {nu:.16g}")

for N in [4, 8]:
    msh_file = f"Yan-beam-N{N}"

    tet_mesh = TetMesh(msh_file)

    # CFL-like estimate based on P-wave speed (units: meters, seconds in `units si`).
    # dt_safe = cfl * h_min / c_p
    edge_lengths = tet_mesh.E_lengths(tet_mesh.E_into_V).reshape(-1)
    h_min = float(edge_lengths.min())
    h_max = float(edge_lengths.max())
    M = 2.0 * G * (1.0 - nu) / (1.0 - 2.0 * nu)  # P-wave modulus
    c_p = (M / rho) ** 0.5
    dt_safe = 0.1 * h_min / c_p

    # bF = tet_mesh.bF_into_V_per_block[0]
    # print(type(bF), len(bF))
    # print(type(bF[0]), np.asarray(bF[0]).shape)

    # NOTE: `elastic_per_cell_block` controls elastic vs rigid blocks.
    comp = Composite(tet_mesh, rho_per_cell_block=[rho], elastic_per_cell_block=[True], M_vec_per_cell_block=[M_vec])
    outbase = write_lam([comp], filename=msh_file + "-Mx")

    NB = 0 if comp.bonds is None else int(comp.bonds.shape[0])
    NI = 0 if comp.impropers is None else int(comp.impropers.shape[0])

    print(f"\nWrote: {outbase}.lam")
    print(f"Atoms: {comp.n_atoms}")
    print(f"Bonds: {NB} (types: {comp.n_bond_types})")
    print(f"Impropers: {NI} (types: {comp.n_improper_types})")
    print(f"h_min={h_min:.6g} m, h_max={h_max:.6g} m")
    print(f"h_min={h_min:.6g} m, c_p={c_p:.6g} m/s -> dt_safe≈{dt_safe:.6g} s (cfl=0.1)")
    print(f"LAMMPS: timestep {dt_safe:.16g}")

    # Mesh stats
    T_vols = tet_mesh.T_vols()
    print(f"Mesh: {tet_mesh.nT} tets, {tet_mesh.nE} edges, {tet_mesh.nV} vertices")
    print(
        f"Volume: min={T_vols.min():.6f}, max={T_vols.max():.6f}, "
        f"std/mean={T_vols.std()/T_vols.mean():.4f}"
    )
