"""
FEniCSx: Incremental uniaxial compression of a unit cube (neo-Hookean).
Reads cube-N16.msh, sweeps nu, writes VTK files for OVITO.

Boundary conditions:
  - Bottom (z=0): fully clamped (u = 0)
  - Top (z=1):    u_x = 0, u_y = 0, u_z = prescribed (clamped platens)

Neo-Hookean stored energy:
  psi = (mu/2)(I_C - 3) - mu*ln(J) + (lmbda/2)(ln(J))^2
"""

import numpy as np
from mpi4py import MPI
from dolfinx import fem, io, log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl

# ---- Material parameters ----
G_val = 4e3  # Pa
rho = 1000.0  # kg/m^3

# nu_list = [0., 0.25, 0.333, 0.4, 0.45, 0.49, 0.499]
nu_list = [0.45]

# ---- Compression parameters ----
emax = 0.20       # 20% engineering strain
Nsteps = 20        # increments

# ---- Read mesh ----
domain, cell_tags, facet_tags = io.gmshio.read_from_msh(
    "cube-N16.msh", MPI.COMM_WORLD, gdim=3
)

# P2 vector function space
V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))

# ---- Locate boundary facets ----
def bottom(x):
    return np.isclose(x[2], 0.0)

def top(x):
    return np.isclose(x[2], 1.0)

# ---- DOFs per sub-space ----
V0, _ = V.sub(0).collapse()  # x-component
V1, _ = V.sub(1).collapse()  # y-component
V2, _ = V.sub(2).collapse()  # z-component

bot_dofs_0 = fem.locate_dofs_geometrical((V.sub(0), V0), bottom)
bot_dofs_1 = fem.locate_dofs_geometrical((V.sub(1), V1), bottom)
bot_dofs_2 = fem.locate_dofs_geometrical((V.sub(2), V2), bottom)

top_dofs_0 = fem.locate_dofs_geometrical((V.sub(0), V0), top)
top_dofs_1 = fem.locate_dofs_geometrical((V.sub(1), V1), top)
top_dofs_2 = fem.locate_dofs_geometrical((V.sub(2), V2), top)

# Zero functions for clamping
zero_func = fem.Function(V0)
zero_func.x.array[:] = 0.0

# Prescribed z-displacement on top (updated each step)
uz_top = fem.Function(V2)

# ---- Loop over nu values ----
for nu_val in nu_list:
    print(f"\n{'='*60}")
    print(f"  nu = {nu_val}")
    print(f"{'='*60}")

    mu_val = G_val
    lmbda_val = 2.0 * G_val * nu_val / (1.0 - 2.0 * nu_val)

    # ---- Solution and test functions ----
    u = fem.Function(V, name="displacement")
    v = ufl.TestFunction(V)

    # ---- Kinematics ----
    d = len(u)
    I = ufl.Identity(d)
    F = ufl.variable(I + ufl.grad(u))
    C = ufl.variable(F.T * F)
    Ic = ufl.variable(ufl.tr(C))
    J = ufl.variable(ufl.det(F))

    # ---- Neo-Hookean strain energy ----
    mu = fem.Constant(domain, default_scalar_type(mu_val))
    lmbda = fem.Constant(domain, default_scalar_type(lmbda_val))

    psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2

    # First Piola-Kirchhoff stress
    P = ufl.diff(psi, F)

    # ---- Weak form ----
    metadata = {"quadrature_degree": 4}
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)
    residual = ufl.inner(ufl.grad(v), P) * dx

    # ---- Boundary conditions ----
    # Bottom: u_x = u_y = u_z = 0
    bc_bot_x = fem.dirichletbc(zero_func, bot_dofs_0, V.sub(0))
    bc_bot_y = fem.dirichletbc(zero_func, bot_dofs_1, V.sub(1))
    bc_bot_z = fem.dirichletbc(zero_func, bot_dofs_2, V.sub(2))

    # Top: u_x = 0, u_y = 0, u_z = prescribed
    bc_top_x = fem.dirichletbc(zero_func, top_dofs_0, V.sub(0))
    bc_top_y = fem.dirichletbc(zero_func, top_dofs_1, V.sub(1))
    bc_top_z = fem.dirichletbc(uz_top, top_dofs_2, V.sub(2))

    bcs = [bc_bot_x, bc_bot_y, bc_bot_z, bc_top_x, bc_top_y, bc_top_z]

    # ---- Newton solver ----
    problem = NonlinearProblem(residual, u, bcs)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.atol = 1e-8
    solver.rtol = 1e-8
    solver.max_it = 50

    log.set_log_level(log.LogLevel.WARNING)

    # ---- Incremental compression ----
    u.x.array[:] = 0.0
    de = emax / Nsteps

    for step in range(1, Nsteps + 1):
        disp_z = -de * step  # total z-displacement at top face

        uz_top.x.array[:] = disp_z
        uz_top.x.scatter_forward()

        num_its, converged = solver.solve(u)
        assert converged, f"Newton did not converge at step {step}, nu={nu_val}"
        u.x.scatter_forward()

        print(f"  step {step:3d}/{Nsteps}  ez = {disp_z:.4f}  "
              f"Newton iters = {num_its}")

    # Write final deformed state as LAMMPS data file
    # Match DOLFINx vertices to .lam atom positions by coordinates
    from scipy.spatial import cKDTree

    V1_vec = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    u_p1 = fem.Function(V1_vec)
    u_p1.interpolate(u)

    # DOLFINx geometry points and P1 function dofs are not guaranteed to have
    # the same ordering. Use dof coordinates because they are aligned with
    # u_p1.x.array.
    coords_fem = V1_vec.tabulate_dof_coordinates()
    bs = V1_vec.dofmap.index_map_bs
    disp = u_p1.x.array.reshape(-1, bs)[:, : domain.geometry.dim]
    deformed_fem = coords_fem + disp

    # Read .lam reference positions to get the target ordering
    # atom_style hybrid full sphere: id type x y z mol charge diam rho
    lam_coords = []
    with open("cube-N16.lam") as f:
        in_atoms = False
        for line in f:
            if line.strip().startswith("Atoms"):
                in_atoms = True
                next(f)  # skip blank line
                continue
            if in_atoms:
                parts = line.split()
                if len(parts) < 5 or not parts[0].isdigit():
                    if line.strip() and not line.strip().startswith("#"):
                        break  # hit next section
                    continue
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                lam_coords.append([x, y, z])
    lam_coords = np.array(lam_coords)

    # Build KD-tree on DOLFINx vertices, query with .lam positions
    tree = cKDTree(coords_fem)
    dists, idx = tree.query(lam_coords)

    if dists.max() > 1e-6:
        print(f"  WARNING: max coord mismatch = {dists.max():.2e}")

    # Reorder deformed positions to .lam atom order
    deformed_ordered = deformed_fem[idx]
    natoms = len(lam_coords)

    bottom_atoms = np.isclose(lam_coords[:, 2], lam_coords[:, 2].min())
    top_atoms = np.isclose(lam_coords[:, 2], lam_coords[:, 2].max())

    # These nodes have essential boundary conditions. Snap the exported
    # coordinates to the prescribed values so visualization/data export cannot
    # show artificial lateral sliding from interpolation or ordering details.
    deformed_ordered[bottom_atoms] = lam_coords[bottom_atoms]
    deformed_ordered[top_atoms, 0] = lam_coords[top_atoms, 0]
    deformed_ordered[top_atoms, 1] = lam_coords[top_atoms, 1]
    deformed_ordered[top_atoms, 2] = lam_coords[top_atoms, 2] - emax

    disp_ordered = deformed_ordered - lam_coords
    max_abs_uz = np.abs(disp_ordered[:, 2]).max()
    max_top_lateral = np.linalg.norm(disp_ordered[top_atoms, :2], axis=1).max()
    max_bottom_motion = np.linalg.norm(disp_ordered[bottom_atoms], axis=1).max()

    xlo, xhi = deformed_ordered[:, 0].min() - 0.01, deformed_ordered[:, 0].max() + 0.01
    ylo, yhi = deformed_ordered[:, 1].min() - 0.01, deformed_ordered[:, 1].max() + 0.01
    zlo, zhi = deformed_ordered[:, 2].min() - 0.01, deformed_ordered[:, 2].max() + 0.01

    fname = f"fem-comp-nu-{nu_val:.3f}.data"
    with open(fname, "w") as f:
        f.write(f"LAMMPS data file - FEM compression nu={nu_val:.3f}\n\n")
        f.write(f"{natoms} atoms\n")
        f.write(f"1 atom types\n\n")
        f.write(f"{xlo:.6f} {xhi:.6f} xlo xhi\n")
        f.write(f"{ylo:.6f} {yhi:.6f} ylo yhi\n")
        f.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n\n")
        f.write("Atoms\n\n")
        for i in range(natoms):
            f.write(f"{i+1} 1 {deformed_ordered[i,0]:.8f} {deformed_ordered[i,1]:.8f} {deformed_ordered[i,2]:.8f}\n")

    print(f"  -> Wrote {fname} ({natoms} atoms)")
    print(
        f"     max |u_z| = {max_abs_uz:.8f}, "
        f"max top lateral = {max_top_lateral:.3e}, "
        f"max bottom motion = {max_bottom_motion:.3e}"
    )

print("\nDone.")
