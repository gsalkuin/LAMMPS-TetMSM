"""
Liu-style local-global solver for tetrahedral MSM, extended with:

- Golec volume penalty via a rest-config geometric stiffness matrix
  on the left-hand side and lagged nonlinear residual on the right-hand side
- Anderson acceleration on the fixed-point map
- energy-based backtracking for robustness
- secant continuation for magnetic load stepping

Springs: local step = direction projection (vector normalization)
         global step = back-substitution with pre-factored matrix
Volume penalty: rest-config stiffness + lagged nonlinear residual
External forces: gravity + FMC charges in B-gradient

The matrix is CONSTANT — factored once, reused forever.
Each local-global iteration costs O(nnz) for the back-solve.
"""

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import splu


class LocalGlobalSolver:
    def __init__(self, pos, tets, bonds, charges, G, nu, rho=1.0):
        self.nv = len(pos)
        self.nb = len(bonds)
        self.nt = len(tets)
        self.ndof = 3 * self.nv

        self.pos = pos.astype(np.float64).copy()
        self.pos0 = self.pos.copy()
        self.tets = tets
        self.bonds = bonds
        self.q = charges.astype(np.float64)
        self.G, self.nu = G, nu
        self.kv = G * (4 * nu - 1) / (1 - 2 * nu)
        self.L_beam = float(np.max(pos[:, 0]))

        # Rest volumes. The volume implementation requires positive V0.
        self.V0 = self._volumes(pos)
        invalid = np.where(self.V0 <= 0.0)[0]
        if len(invalid):
            raise ValueError(
                f"{len(invalid)} tetrahedra have non-positive reference volume; "
                "check vertex ordering"
            )
        print(f"  V0 range: [{self.V0.min():.4e}, {self.V0.max():.4e}]")

        # Springs: match lammps-src/volume/bond_tetmsm.cpp.
        # k_e = sqrt(2)/5 * G * sum_adjacent_tets(l_eff)
        r = pos[bonds[:, 1]] - pos[bonds[:, 0]]
        self.r0 = np.linalg.norm(r, axis=1)
        if np.any(self.r0 <= 0.0):
            raise ValueError("zero-length bond in reference configuration")
        self.kE = self._spring_stiffness()
        print(f"  kE range: [{self.kE.min():.4e}, {self.kE.max():.4e}]")

        # Lumped mass
        self.mass = np.zeros(self.nv)
        np.add.at(self.mass, self.tets.ravel(), np.repeat(rho * self.V0 / 4, 4))

        self.M_diag = np.repeat(self.mass, 3)
        self.vel = np.zeros_like(pos)
        self.fixed_mask = np.zeros(self.ndof, dtype=bool)
        self.last_converged = False

        self._assemble_L_J()

    # -------------------------------------------------------------- mesh
    def _volumes(self, x):
        a, b, c, d = (x[self.tets[:, k]] for k in range(4))
        return np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6

    def _volume_gradients(self, x):
        v = self.tets
        x0, x1, x2, x3 = x[v[:, 0]], x[v[:, 1]], x[v[:, 2]], x[v[:, 3]]
        g = np.empty((self.nt, 4, 3), dtype=x.dtype)
        g[:, 1] = np.cross(x2 - x0, x3 - x0) / 6
        g[:, 2] = np.cross(x3 - x0, x1 - x0) / 6
        g[:, 3] = np.cross(x1 - x0, x2 - x0) / 6
        g[:, 0] = -(g[:, 1] + g[:, 2] + g[:, 3])
        return g

    def _spring_stiffness(self):
        edge_to_bond = {}
        for e, bond in enumerate(self.bonds):
            key = tuple(sorted((int(bond[0]), int(bond[1]))))
            if key in edge_to_bond:
                raise ValueError(f"duplicate bond for edge {key}")
            edge_to_bond[key] = e

        leff_sum = np.zeros(self.nb)
        leff = np.cbrt(12.0 * self.V0 / np.sqrt(2.0))
        for tet, tet_leff in zip(self.tets, leff):
            for a in range(3):
                for b in range(a + 1, 4):
                    key = tuple(sorted((int(tet[a]), int(tet[b]))))
                    try:
                        edge = edge_to_bond[key]
                    except KeyError as exc:
                        raise ValueError(
                            f"tetrahedron edge {key} is missing from Bonds"
                        ) from exc
                    leff_sum[edge] += tet_leff

        missing = np.where(leff_sum == 0.0)[0]
        if len(missing):
            raise ValueError(
                f"{len(missing)} bonds are not edges of any tetrahedron"
            )
        return (np.sqrt(2.0) / 5.0) * self.G * leff_sum

    # ------------------------------------------------- constant matrices
    def _assemble_L_J(self):
        """L = stiffness-weighted Laplacian + rest-config volume stiffness,
        J = spring-direction coupling."""
        n, s3 = self.ndof, 3 * self.nb

        # Spring Laplacian
        dims = np.tile(np.arange(3), self.nb)
        bi = np.repeat(self.bonds[:, 0], 3)
        bj = np.repeat(self.bonds[:, 1], 3)
        edge = np.repeat(np.arange(self.nb), 3)
        kd = np.repeat(self.kE, 3)

        ri = 3 * bi + dims
        rj = 3 * bj + dims
        rs = 3 * edge + dims

        rL_spring = np.concatenate([ri, rj, ri, rj])
        cL_spring = np.concatenate([ri, rj, rj, ri])
        vL_spring = np.concatenate([kd, kd, -kd, -kd])

        rJ = np.concatenate([ri, rj])
        cJ = np.concatenate([rs, rs])
        vJ = np.concatenate([kd, -kd])

        spring_L = csr_matrix(
            (vL_spring, (rL_spring, cL_spring)), shape=(n, n)
        )

        # Rest-config volume stiffness is B^T diag(kv/V0) B, where each
        # row of B contains one tetrahedron's 12 volume-gradient entries.
        g = self._volume_gradients(self.pos0)
        B_rows = np.repeat(np.arange(self.nt), 12)
        B_cols = (
            3 * self.tets[:, :, None] + np.arange(3)[None, None, :]
        ).reshape(-1)
        B = csr_matrix((g.reshape(-1), (B_rows, B_cols)), shape=(self.nt, n))
        weighted_B = B.multiply((self.kv / self.V0)[:, None])
        self.K_vol_rest = (B.T @ weighted_B).tocsr()

        self.L_mat = spring_L + self.K_vol_rest
        self.J_mat = csr_matrix((vJ, (rJ, cJ)), shape=(n, s3))

    # --------------------------------------------------------------- BC
    def clamp(self, x_max):
        mask = self.pos0[:, 0] < x_max
        for i in np.where(mask)[0]:
            self.fixed_mask[3 * i: 3 * i + 3] = True

    # -------------------------------------------------------- prefactor
    def prefactor(self, h=1.0):
        """Pre-factor the free-DOF block of A. Call once after clamp()."""
        self.h = h
        self.h2 = h * h
        n = self.ndof

        M_sp = diags(self.M_diag, 0, shape=(n, n), format="csr")
        A = M_sp + self.h2 * self.L_mat

        self.free_dofs = np.flatnonzero(~self.fixed_mask)
        self.fixed_dofs = np.flatnonzero(self.fixed_mask)
        if len(self.free_dofs) == 0:
            raise ValueError("all degrees of freedom are fixed")

        A_ff = A[self.free_dofs][:, self.free_dofs].tocsc()
        self.A_free_fixed = A[self.free_dofs][:, self.fixed_dofs].tocsr()
        self.A_factor = splu(A_ff)
        self.x_fixed = self.pos0.ravel()[self.fixed_dofs].copy()
        n_fixed = self.fixed_mask.sum() // 3
        print(f"  prefactored {len(self.free_dofs)} free DOFs, nnz={A_ff.nnz}, "
              f"{n_fixed} fixed verts, h={h}")

    # -------------------------------------------------------- local step
    def _local_step(self, x):
        """d_i = r_i * (p_{i1} - p_{i2}) / ||p_{i1} - p_{i2}||"""
        # J is assembled with +k at vertex i and -k at vertex j.
        r = x[self.bonds[:, 0]] - x[self.bonds[:, 1]]
        dist = np.linalg.norm(r, axis=1, keepdims=True)
        dist = np.maximum(dist, 1e-15)
        d = (self.r0[:, None] / dist) * r
        return d.ravel()

    # --------------------------------------------------- volume forces
    def _volume_forces(self, x):
        f = np.zeros_like(x)
        g = self._volume_gradients(x)
        strain_v = (self._volumes(x) - self.V0) / self.V0
        coeff = (-self.kv * strain_v)[:, None]
        for local in range(4):
            np.add.at(f, self.tets[:, local], coeff * g[:, local])
        return f

    # ------------------------------------------------- external forces
    def _ext_forces(self, x, dB, gravity):
        f = np.zeros_like(x)
        f[:, 0] += self.mass * gravity
        if abs(dB) > 1e-15:
            Lb = self.L_beam
            f[:, 0] += self.q * (-0.5 * dB * (x[:, 0] - Lb / 2))
            f[:, 1] += self.q * (-0.5 * dB * x[:, 1])
            f[:, 2] += self.q * (dB * x[:, 2])
        return f

    def _objective(self, x, y, dB, gravity):
        inertia = 0.5 * np.sum(self.mass[:, None] * (x - y) ** 2)

        edge = x[self.bonds[:, 0]] - x[self.bonds[:, 1]]
        stretch = np.linalg.norm(edge, axis=1) - self.r0
        spring = 0.5 * np.dot(self.kE, stretch * stretch)

        strain_v = (self._volumes(x) - self.V0) / self.V0
        volume = 0.5 * self.kv * np.dot(self.V0, strain_v * strain_v)

        potential = -np.dot(self.mass * gravity, x[:, 0])
        if abs(dB) > 1e-15:
            potential += np.sum(
                self.q
                * (
                    0.25 * dB * (x[:, 0] - self.L_beam / 2) ** 2
                    + 0.25 * dB * x[:, 1] ** 2
                    - 0.5 * dB * x[:, 2] ** 2
                )
            )
        return inertia + self.h2 * (spring + volume + potential)

    def _fixed_point_candidate(self, x, y_flat, dB, gravity, anderson_m=0,
                               x_hist=None, g_hist=None, f_hist=None):
        """One local-global map evaluation for residual checking."""
        d = self._local_step(x)
        f_vol = self._volume_forces(x).ravel()
        f_ext = self._ext_forces(x, dB, gravity).ravel()
        x_flat = x.ravel()
        rhs = (self.h2 * self.J_mat.dot(d)
               + self.M_diag * y_flat
               + self.h2 * (f_vol + f_ext)
               + self.h2 * self.K_vol_rest.dot(x_flat))
        rhs_free = rhs[self.free_dofs]
        if len(self.fixed_dofs):
            rhs_free -= self.A_free_fixed.dot(self.x_fixed)
        candidate = np.empty(self.ndof)
        candidate[self.free_dofs] = self.A_factor.solve(rhs_free)
        candidate[self.fixed_dofs] = self.x_fixed

        if anderson_m >= 2 and x_hist is not None and g_hist is not None and f_hist is not None:
            x_hist.append(x_flat.copy())
            g_hist.append(candidate.copy())
            f_hist.append(candidate - x_flat)
            m = min(anderson_m, len(f_hist))
            if m >= 2:
                F = np.column_stack(f_hist[-m:])
                G = np.column_stack(g_hist[-m:])
                ones = np.ones((m, 1))
                gram = F.T @ F
                kkt = np.block([
                    [gram + 1e-14 * np.eye(m), ones],
                    [ones.T, np.zeros((1, 1))],
                ])
                rhs_kkt = np.zeros(m + 1)
                rhs_kkt[-1] = 1.0
                try:
                    sol = np.linalg.solve(kkt, rhs_kkt)
                    alpha = sol[:m]
                    candidate = G @ alpha
                except np.linalg.LinAlgError:
                    pass
        return candidate

    def _fixed_point_residual(self, x, y_flat, dB, gravity):
        cand = self._fixed_point_candidate(x, y_flat, dB, gravity, anderson_m=0)
        return np.sqrt(np.mean((cand.reshape(-1, 3) - x) ** 2))

    # ----------------------------------------------------- one timestep
    def step(self, dB=0.0, gravity=9.8e-3, n_iters=5, damping=1.0,
             energy_relax=0.0, anderson_m=0):
        x = self.pos.copy()
        h = self.h
        x_hist = []
        g_hist = []
        f_hist = []

        # Inertia: y = pos + h * damped_velocity
        y = self.pos + h * self.vel * (1 - damping)
        y_flat = y.ravel()

        energy = self._objective(x, y, dB, gravity)
        if not np.isfinite(energy):
            raise FloatingPointError("non-finite objective at start of timestep")

        for iteration in range(n_iters):
            candidate = self._fixed_point_candidate(
                x, y_flat, dB, gravity, anderson_m, x_hist, g_hist, f_hist
            )
            if not np.all(np.isfinite(candidate)):
                raise FloatingPointError(
                    f"non-finite global solve result at local-global iteration {iteration}"
                )

            # The volume force is nonlinear and lagged. Backtracking keeps the
            # accelerated update from running away.
            direction = candidate.reshape(-1, 3) - x
            update_norm = np.linalg.norm(direction)
            if update_norm < 1e-12:
                break

            alpha = 1.0
            accepted = False
            for _ in range(20):
                x_trial = x + alpha * direction
                trial_energy = self._objective(x_trial, y, dB, gravity)
                energy_tol = 1e-12 * max(1.0, abs(energy))
                energy_limit = energy + energy_tol + energy_relax * max(1.0, abs(energy))
                if np.isfinite(trial_energy) and trial_energy <= energy_limit:
                    x = x_trial
                    energy = trial_energy
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                raise RuntimeError(
                    "local-global direction was not accepted; "
                    "increase energy_relax, reduce h, or inspect the applied load"
                )
            if alpha * update_norm < 1e-10:
                break

        self.vel = (x - self.pos) / h
        self.pos = x

    # ------------------------------------------------------- quasi-static
    def solve(self, dB=0.0, gravity=9.8e-3, n_steps=30,
              n_iters=5, damping=1.0, tol=5e-7, verbose=True,
              energy_relax=0.0, anderson_m=0):
        converged = False
        for s in range(n_steps):
            x_prev = self.pos.copy()
            self.step(dB=dB, gravity=gravity, n_iters=n_iters,
                      damping=damping, energy_relax=energy_relax,
                      anderson_m=anderson_m)
            dx = np.sqrt(np.mean(np.sum((self.pos - x_prev) ** 2, axis=1)))
            check_res = verbose and (s % 10 == 0 or dx < 10 * tol)
            res = np.inf
            if check_res:
                y = self.pos + self.h * self.vel * (1 - damping)
                res = self._fixed_point_residual(self.pos, y.ravel(), dB, gravity)
            if verbose and s % 10 == 0:
                if np.isfinite(res):
                    print(f"    step {s:3d}: rms|Δx| = {dx:.4e}  res = {res:.4e}")
                else:
                    print(f"    step {s:3d}: rms|Δx| = {dx:.4e}")
            if dx < tol and (not check_res or res < tol):
                if verbose:
                    if np.isfinite(res):
                        print(f"    converged step {s}: rms|Δx| = {dx:.4e}  res = {res:.4e}")
                    else:
                        print(f"    converged step {s}: rms|Δx| = {dx:.4e}")
                converged = True
                break
        self.last_converged = converged
        return converged

    def predict_from(self, x_ref, x_prev, lam_ref, lam_prev, lam_new):
        """Secant continuation predictor for the next load step."""
        if x_ref is None or x_prev is None:
            return
        denom = lam_ref - lam_prev
        if abs(denom) < 1e-15:
            return
        scale = (lam_new - lam_ref) / denom
        self.pos = x_ref + scale * (x_ref - x_prev)
        self.vel = np.zeros_like(self.pos)

    # ----------------------------------------------------------- output
    def tip_deflection(self):
        right = self.pos0[:, 0] > self.L_beam - 0.05
        if not np.any(right):
            return 0.0, 0.0
        tip = self.pos[right].mean(axis=0)
        tip0 = self.pos0[right].mean(axis=0)
        return (tip[0] - tip0[0]) / self.L_beam, (tip[2] - tip0[2]) / self.L_beam

    def write_dump(self, filename):
        with open(filename, "w") as f:
            f.write("ITEM: TIMESTEP\n0\n")
            f.write(f"ITEM: NUMBER OF ATOMS\n{self.nv}\n")
            f.write("ITEM: BOX BOUNDS ff ff ff\n")
            lo = self.pos.min(axis=0) - 2
            hi = self.pos.max(axis=0) + 2
            for d in range(3):
                f.write(f"{lo[d]} {hi[d]}\n")
            f.write("ITEM: ATOMS id type x y z q\n")
            for i in range(self.nv):
                t = 2 if self.fixed_mask[3 * i] else 1
                p = self.pos[i]
                f.write(f"{i+1} {t} {p[0]:.8f} {p[1]:.8f} "
                        f"{p[2]:.8f} {self.q[i]:.8e}\n")

    def save_state(self, filename, rows=None, next_index=0,
                   prev_pos=None, prev_prev_pos=None,
                   prev_lam=None, prev_lam_prev=None):
        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        data = {
            "pos": self.pos,
            "vel": self.vel,
            "last_converged": np.array(bool(self.last_converged)),
            "next_index": np.array(int(next_index)),
            "prev_pos": np.array(prev_pos) if prev_pos is not None else np.empty((0, 3)),
            "prev_prev_pos": np.array(prev_prev_pos) if prev_prev_pos is not None else np.empty((0, 3)),
            "prev_lam": np.array(np.nan if prev_lam is None else float(prev_lam)),
            "prev_lam_prev": np.array(np.nan if prev_lam_prev is None else float(prev_lam_prev)),
            "rows": np.array(rows, dtype=float) if rows is not None else np.empty((0, 3)),
        }
        np.savez_compressed(filename, **data)

    def load_state(self, filename):
        data = np.load(filename, allow_pickle=False)
        self.pos = data["pos"].astype(np.float64).copy()
        self.vel = data["vel"].astype(np.float64).copy()
        self.last_converged = bool(np.asarray(data["last_converged"]).item()) if "last_converged" in data else False
        prev_pos = data["prev_pos"] if "prev_pos" in data else np.empty((0, 3))
        prev_prev_pos = data["prev_prev_pos"] if "prev_prev_pos" in data else np.empty((0, 3))
        prev_lam_arr = np.asarray(data["prev_lam"]) if "prev_lam" in data else np.array(np.nan)
        prev_lam_prev_arr = np.asarray(data["prev_lam_prev"]) if "prev_lam_prev" in data else np.array(np.nan)
        prev_lam = float(prev_lam_arr) if np.isfinite(prev_lam_arr).item() else None
        prev_lam_prev = float(prev_lam_prev_arr) if np.isfinite(prev_lam_prev_arr).item() else None
        rows = data["rows"] if "rows" in data else np.empty((0, 3))
        next_index = int(data["next_index"]) if "next_index" in data else 0
        return {
            "prev_pos": prev_pos if len(prev_pos) else None,
            "prev_prev_pos": prev_prev_pos if len(prev_prev_pos) else None,
            "prev_lam": prev_lam,
            "prev_lam_prev": prev_lam_prev,
            "rows": rows,
            "next_index": next_index,
            "resume_current": next_index < len(rows),
        }


# ============================================================ data reader
def read_lammps_data(filename):
    atoms, bond_ids, tet_ids = {}, [], []
    section = None
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            header = line.split("#")[0].strip()
            if header == "Atoms":
                section = "atoms"; continue
            elif header == "Bonds":
                section = "bonds"; continue
            elif header == "Impropers":
                section = "impropers"; continue
            elif header in ("Masses", "Bond Coeffs", "Improper Coeffs",
                            "Pair Coeffs", "Velocities"):
                section = "skip"; continue
            if section == "atoms":
                p = line.split()
                if len(p) >= 7:
                    atom_id = int(p[0])
                    if atom_id in atoms:
                        raise ValueError(f"duplicate atom id {atom_id}")
                    atoms[atom_id] = (
                        [float(p[2]), float(p[3]), float(p[4])],
                        float(p[6]),
                    )
            elif section == "bonds":
                p = line.split()
                bond_ids.append([int(p[2]), int(p[3])])
            elif section == "impropers":
                p = line.split()
                tet_ids.append([int(p[2]), int(p[3]), int(p[4]), int(p[5])])

    atom_ids = sorted(atoms)
    id_to_index = {atom_id: index for index, atom_id in enumerate(atom_ids)}
    try:
        bonds = [[id_to_index[a], id_to_index[b]] for a, b in bond_ids]
        tets = [[id_to_index[v] for v in tet] for tet in tet_ids]
    except KeyError as exc:
        raise ValueError(f"topology references missing atom id {exc.args[0]}") from exc

    pos = np.array([atoms[atom_id][0] for atom_id in atom_ids], dtype=float)
    charges = np.array([atoms[atom_id][1] for atom_id in atom_ids], dtype=float)
    return pos, np.array(tets, dtype=int), np.array(bonds, dtype=int), charges


# ============================================================ main
if __name__ == "__main__":
    import argparse
    import os
    os.makedirs("dump", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs="?", default="Yan-beam-N4-Mz.lam")
    parser.add_argument("--state-in", dest="state_in", default=None,
                        help="load a saved solver state from an .npz file")
    parser.add_argument("--state-out", dest="state_out", default=None,
                        help="write the last converged solver state to an .npz file")
    args = parser.parse_args()

    data_file = args.data_file
    print(f"Reading {data_file}")
    pos, tets, bonds, charges = read_lammps_data(data_file)
    print(f"  {len(pos)} verts, {len(bonds)} bonds, {len(tets)} tets")
    print(f"  charge range: [{charges.min():.4e}, {charges.max():.4e}]")

    E = 1.16e3; nu = 0.49; G = E / (2 * (1 + nu))

    solver = LocalGlobalSolver(
        pos, tets, bonds, charges, G=G, nu=nu, rho=2.01
    )
    solver.clamp(x_max=0.05)
    solver.prefactor(h=0.5)

    state_meta = {
        "prev_pos": None,
        "prev_prev_pos": None,
        "prev_lam": None,
        "prev_lam_prev": None,
        "rows": [],
        "next_index": 0,
    }
    if args.state_in:
        print(f"Loading restart state: {args.state_in}")
        state_meta = solver.load_state(args.state_in)
        solver.prefactor(h=0.5)
        print(
            f"  restored {len(state_meta['rows'])} sweep rows, "
            f"next index = {state_meta['next_index']}"
        )
    else:
        # Gravity
        print("\nGravity relaxation:")
        solver.solve(dB=0.0, n_steps=1000, n_iters=5, damping=1.0, tol=5e-7,
                     energy_relax=0.0, anderson_m=0)
        dx, dz = solver.tip_deflection()
        print(f"  tip: dx/L = {dx:.6e}, dz/L = {dz:.6e}")
        solver.write_dump("dump/gravity.dump")

    solver.prefactor(h=5.0)

    # Incremental efield
    dBone = 1.437e-3
    dB_vals = [dBone * i for i in [0.01, 0.1, 0.2, 0.5, 0.75, 1, 2, 3, 3.5, 4, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100]]
    out_txt = os.path.join(
        "dump",
        os.path.basename(data_file).replace("Yan-beam", "Yan").replace(".lam", "-deflection.txt"),
    )

    print("\n#   lambda      dx/L          dz/L")
    rows = [tuple(row) for row in state_meta["rows"]]
    prev_prev_pos = state_meta["prev_prev_pos"]
    prev_pos = state_meta["prev_pos"] if state_meta["prev_pos"] is not None else solver.pos.copy()
    prev_lam = state_meta["prev_lam"]
    prev_lam_prev = state_meta["prev_lam_prev"]
    start_index = int(state_meta["next_index"])
    resume_current = bool(state_meta.get("resume_current", False))
    saved_state = False
    if start_index < len(rows):
        rows = rows[:start_index]
    with open(out_txt, "w") as f_txt:
        f_txt.write("# lambda deltax deltaz\n")
        for lam, dx, dz in rows:
            f_txt.write(f"{lam:.16g} {dx:.16g} {dz:.16g}\n")
    for k, dB in enumerate(dB_vals[start_index:], start=start_index):
        lam = dB / dBone
        if prev_lam is not None and prev_lam_prev is not None and not (resume_current and k == start_index):
            solver.predict_from(prev_pos, prev_prev_pos, prev_lam, prev_lam_prev, lam)
        print(f"\nField loading: lambda {lam:.2f}")
        converged = solver.solve(dB=dB, n_steps=5000, n_iters=10, damping=1.0,
                                 tol=1e-6, energy_relax=1e-3, anderson_m=3,
                                 verbose=True)
        dx, dz = solver.tip_deflection()
        print(f"  lambda {lam:6.2f}: dx/L = {dx:14.6e}, dz/L = {dz:14.6e}")
        rows.append((lam, dx, dz))
        with open(out_txt, "a") as f_txt:
            f_txt.write(f"{lam:.16g} {dx:.16g} {dz:.16g}\n")
        if converged:
            prev_prev_pos = prev_pos
            prev_pos = solver.pos.copy()
            prev_lam_prev = prev_lam
            prev_lam = lam
            if args.state_out:
                solver.save_state(
                    args.state_out,
                    rows=rows,
                    next_index=k + 1,
                    prev_pos=prev_pos,
                    prev_prev_pos=prev_prev_pos,
                    prev_lam=prev_lam,
                    prev_lam_prev=prev_lam_prev,
                )
                saved_state = True
                print(f"  saved restart state: {args.state_out}")
        else:
            print(f"  WARNING: lambda {lam:.2f} did not converge; stopping sweep")
            if args.state_out:
                solver.save_state(
                    args.state_out,
                    rows=rows,
                    next_index=k,
                    prev_pos=solver.pos.copy(),
                    prev_prev_pos=prev_pos,
                    prev_lam=prev_lam,
                    prev_lam_prev=prev_lam_prev,
                )
                saved_state = True
                print(f"  saved restart state: {args.state_out}")
            solver.write_dump(f"dump/lambda-{lam:.2f}.dump")
            if args.state_out:
                print(
                    f"  Restart from {args.state_out} with --state-in "
                    f"{args.state_out}"
                )
            else:
                print(
                    "  Rerun with --state-out <restart.npz> if you want an "
                    "automatic resume point next time"
                )
            break
        solver.write_dump(f"dump/lambda-{lam:.2f}.dump")
    print(f"\nWrote sweep table: {out_txt}")
    if args.state_out and saved_state:
        print(f"Wrote restart state: {args.state_out}")
