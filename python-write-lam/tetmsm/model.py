"""
Part and Composite classes for tetrahedral mass-spring models.

Generates LAMMPS data files for bond_style tetmsm + improper_style tetmsm.
The Python pipeline writes topology only (bonds = edges, impropers = tets).
All stiffness computation (kappa_E, kappa_V) is done internally by LAMMPS
from the material properties (G, nu) specified in the input script.

Usage:
    bond_style tetmsm
    bond_coeff *                   # no arguments

    improper_style tetmsm
    improper_coeff <type> G nu     # one per (G, nu) pair

Atom types:
  1 = elastic bulk
  2 = elastic boundary
  3 = rigid bulk       (added by _rigidify)
  4 = rigid boundary   (added by _rigidify)
  5 = magnetic         (added by _magnetize)

Magnetic coupling via fictitious surface magnetic charges (div M = 0).
Atom style: hybrid full sphere.
"""
import warnings
import numpy as np
import igl

class _Part:
    def __init__(self, tet_mesh, cell_block_index, rho,
                 is_elastic=True, M_vec=None):
        self.cell_block_index = cell_block_index
        self.rho = rho
        self.magnet_flag = False

        self.T_into_V = tet_mesh.T_into_V_per_block[cell_block_index]
        self.T_into_E_idx = tet_mesh.T_into_E_idx_per_block[cell_block_index]
        self.E_into_V = tet_mesh.E_into_V_per_block[cell_block_index]
        self.E_lengths = tet_mesh.E_lengths(self.E_into_V)
        self.T_vols = tet_mesh.T_vols_per_block()[cell_block_index]
        self.b_loop_Vs = tet_mesh.bV_per_block[cell_block_index]

        if is_elastic:
            self.rigid_flag = False
            self.bond_flag = True
            self.improper_flag = True
            self.bonds = self.E_into_V
            self.impropers = self.T_into_V
            if M_vec is not None:
                warnings.warn("Elastic part is also magnetic.")
                if M_vec.size != 3:
                    raise ValueError("M_vec dimensions must be 3.")
                self.magnet_flag = True
                self.M_vec = M_vec
        else:
            self.rigid_flag = True
            self.bond_flag = False
            self.improper_flag = False
            if M_vec is not None:
                if M_vec.size != 3:
                    raise ValueError("M_vec dimensions must be 3.")
                self.magnet_flag = True
                self.M_vec = M_vec


# ######################################################################
class Composite:
    def __init__(self, tet_mesh, rho_per_cell_block,
                 elastic_per_cell_block=None, M_vec_per_cell_block=None):
        self.tet_mesh = tet_mesh
        self.mesh = tet_mesh.mesh
        self.n_parts = tet_mesh.n_cell_blocks

        if elastic_per_cell_block is None:
            elastic_per_cell_block = [True] * self.n_parts
        elif isinstance(elastic_per_cell_block, (bool, np.bool_)):
            elastic_per_cell_block = [bool(elastic_per_cell_block)] * self.n_parts
        if len(elastic_per_cell_block) != self.n_parts:
            raise ValueError("Number of parts does not match number of elastic flags.")

        if M_vec_per_cell_block is None:
            M_vec_per_cell_block = [None] * self.n_parts
        if len(M_vec_per_cell_block) != self.n_parts:
            raise ValueError("Number of parts does not match number of magnetizations.")

        if isinstance(rho_per_cell_block, (float, np.floating)):
            rho_per_cell_block = [float(rho_per_cell_block)] * self.n_parts
        if len(rho_per_cell_block) != self.n_parts:
            raise ValueError("Number of parts does not match number of densities.")

        self.parts = []
        for i in range(self.n_parts):
            self.parts.append(
                _Part(tet_mesh, i, rho_per_cell_block[i],
                      elastic_per_cell_block[i], M_vec_per_cell_block[i]))

        # ---- Bonds (tetmsm: 1 type, no coefficients) ----
        self.bond_style = "tetmsm"
        self.n_bond_types = 1

        # Use mesh-global unique edges (avoids duplicate bonds at elastic/elastic interfaces).
        elastic_v = np.zeros(tet_mesh.nV, dtype=bool)
        for p in self.parts:
            if p.bond_flag:
                elastic_v[p.T_into_V.flatten()] = True

        if np.any(elastic_v):
            E = np.asarray(tet_mesh.E_into_V, dtype=int)
            mask = elastic_v[E[:, 0]] & elastic_v[E[:, 1]]
            edges = E[mask]
            types = np.zeros((edges.shape[0], 1), dtype=int)
            self.bonds = np.concatenate([types, edges], axis=1)
        else:
            self.bonds = None

        # ---- Impropers (tetmsm: 1 type per elastic material) ----
        self.improper_style = "tetmsm"
        impropers = []
        imp_type = 0
        for p in self.parts:
            if p.improper_flag:
                types = np.full(p.impropers.shape[0], imp_type, dtype=int)[:, None]
                impropers.append(np.concatenate([types, p.impropers], axis=1))
                imp_type += 1
        if impropers:
            self.improper_flag = True
            self.impropers = np.concatenate(impropers, axis=0)
            self.n_improper_types = imp_type
        else:
            self.improper_flag = False
            self.impropers = None
            self.n_improper_types = 0

        # ---- Atoms ----
        self.V_into_xyz = tet_mesh.V_into_xyz
        self.n_atoms = tet_mesh.nV
        self.V_into_q = np.zeros(self.n_atoms, float)
        self.V_into_atom_type = np.ones(self.n_atoms, int)

        # Mark boundary vertices as elastic boundary (type 2).
        # Rigid boundary (type 4) is applied later in _rigidify().
        if self.tet_mesh.bV_per_block:
            bverts = np.unique(np.concatenate(self.tet_mesh.bV_per_block))
            self.V_into_atom_type[bverts] = 2

        self.atom_types = [1, 2]
        self.rigid_flag = False
        self.magnet_flag = False
        self.V_into_diam = np.zeros(self.n_atoms, float)
        self.V_into_rho = np.zeros(self.n_atoms, float)

        # ---- Part IDs ----
        self.V_into_part_ID = np.zeros(self.n_atoms, int)
        self.part_IDs = list(range(1, self.n_parts + 1))
        self.rigid_flags = np.array([p.rigid_flag for p in self.parts], dtype=int)
        for index in np.argsort(self.rigid_flags):
            self.set_part_ID(self.parts[index], self.part_IDs[index])
        if np.any(self.V_into_part_ID == 0):
            print("WARNING: Atoms with mol-ID = 0 found!")

        self._rigidify()
        self.V_into_rigid_ID = None
        self.n_magnets = self._magnetize()
        self.compute_rho_from_mesh()
        self.n_atom_types = len(self.atom_types)

    # ------------------------------------------------------------------
    def set_part_ID(self, part, ID):
        for _, v in np.ndenumerate(part.T_into_V):
            self.V_into_part_ID[v] = ID

    def set_rigid_ID(self, list_of_part_IDs, rigid_ID):
        if self.V_into_rigid_ID is None:
            self.V_into_rigid_ID = np.copy(self.V_into_part_ID)
            self.V_into_rigid_ID = np.where(
                np.isin(self.V_into_atom_type, [1, 2]), 0, self.V_into_rigid_ID)
        self.V_into_rigid_ID = np.where(
            np.isin(self.V_into_part_ID, list_of_part_IDs), rigid_ID, self.V_into_rigid_ID)

    def _rigidify(self):
        """Rigid bulk = type 3, rigid boundary = type 4."""
        for p in self.parts:
            if p.rigid_flag:
                self.rigid_flag = True
                self.V_into_atom_type[p.T_into_V.flatten()] = 3
                self.V_into_atom_type[p.b_loop_Vs] = 4
        if self.rigid_flag:
            self.atom_types += [3, 4]

    def _magnetize(self):
        """Magnetic surface charges. Magnetic atoms = type 5."""
        n = 0
        for p in self.parts:
            if p.magnet_flag:
                self.magnet_flag = True
                n += 1
                bF = self.tet_mesh.bF_into_V_per_block[p.cell_block_index]
                c, _ = igl.orientable_patches(bF)
                bF = igl.orient_outward(self.V_into_xyz, bF, c)[0]
                bF = igl.bfs_orient(bF)[0]
                normals = igl.per_face_normals(self.V_into_xyz, bF, np.array([]))
                areas = igl.doublearea(self.V_into_xyz, bF) / 2.0
                q_per_F = areas * np.sum(p.M_vec[None, :] * normals, axis=1)
                for idx, v in np.ndenumerate(bF):
                    self.V_into_q[v] += q_per_F[idx[0]] / 3.0
        if self.magnet_flag:
            self.atom_types.append(5)
        return n

    def set_rho_per_type(self, t, rho):
        self.V_into_rho[self.V_into_atom_type == t] = rho

    def set_diam_per_type(self, t, diam):
        self.V_into_diam[self.V_into_atom_type == t] = diam

    def compute_rho_from_mesh(self):
        if np.any(self.V_into_rho != 0.):
            print("WARNING: Masses will be reset.")
            self.V_into_rho = np.zeros(self.n_atoms, float)
        for p in self.parts:
            for t, verts in enumerate(p.T_into_V):
                self.V_into_rho[verts] += p.rho * p.T_vols[t] / 4.
        sph_vol = np.pi * self.V_into_diam ** 3 / 6.
        self.V_into_rho = np.where(
            self.V_into_diam == 0., self.V_into_rho, self.V_into_rho / sph_vol)