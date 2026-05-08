"""
Read the .msh file created using Gmsh
"""
import os
import numpy as np
import meshio
import igl

class TetMesh:
    def __init__(self, dot_msh):
        """
        Parameters:
        dot_msh: <string> name of the .msh file without the extension
        NOTE:
        1) If using Gmsh v.4 ASCII format, check 'save all elements' but not 'save parametric coordinates'.
        2) If using physical groups, do not save all elements. Define physical groups for everything you want saved.
        """
        self.mesh = self.read_msh(dot_msh) # meshio.Mesh object

        self.V_into_xyz = self.mesh.points
        self.T_into_V = self.mesh.get_cells_type("tetra")
        self.E_into_V = igl.edges(self.T_into_V)

        etol = 1e-2
        if np.min(self.E_lengths(self.E_into_V)) < etol*np.max(self.E_lengths(self.E_into_V)):
            raise Exception("Edge length too short. This may affect stability in simulations.")

        self.nV = self.V_into_xyz.shape[0]
        self.nT = self.T_into_V.shape[0]
        self.nE = self.E_into_V.shape[0]

        self.T_into_V_per_block = [cell_block.data for cell_block in self.mesh.cells if cell_block.type == "tetra"]

        self.E_into_V_per_block = [igl.edges(T_into_V) for T_into_V in self.T_into_V_per_block]

        self.E_idx_per_block = self._E_per_block_to_global_idx()

        self.T_into_E_idx_per_block = self._T_per_block_to_E_idx()

        self.n_cell_blocks = len(self.T_into_V_per_block)
        self.nT_per_block = [T_into_V.shape[0] for T_into_V in self.T_into_V_per_block]

        self.bF_into_V_per_block = [igl.boundary_facets(T_into_V) for T_into_V in self.T_into_V_per_block]
        self.bV_per_block = []
        for bF in self.bF_into_V_per_block:
            # libigl's python bindings may return either an (nF,3) ndarray or a
            # tuple like (facets, map1, map2). We only need the facets.
            facets = None
            if isinstance(bF, (tuple, list)):
                for item in bF:
                    if isinstance(item, np.ndarray) and item.ndim == 2:
                        facets = item
                        break
                if facets is None and bF:
                    # best-effort fallback
                    facets = np.asarray(bF[0])
            else:
                facets = np.asarray(bF)

            facets = np.asarray(facets)
            if facets.size == 0:
                self.bV_per_block.append(np.array([], dtype=int))
            else:
                # facets is typically (nF, 3) boundary facets. Flatten robustly.
                self.bV_per_block.append(np.unique(facets.reshape(-1)))

    def read_msh(self, dot_msh):
        cwd = os.getcwd()
        msh_file = os.path.join(cwd, dot_msh + '.msh')
        mesh = meshio.read(msh_file)
        return mesh

    # for each block, map the edges to the global edge indices
    def _E_per_block_to_global_idx(self):
        E_global_into_sortedV = np.sort(self.E_into_V, axis=1)
        E_idx_per_block = []
        for E_into_V in self.E_into_V_per_block:
            E_into_sortedV = np.sort(E_into_V, axis=1)

            # same as np.argwhere((E_into_sortedV[:,None,:] == E_global_into_sortedV[None,:,:]).all(axis=-1))[:,-1]
            E_idx = np.searchsorted(np.lexsort(E_global_into_sortedV.T), np.lexsort(E_into_sortedV.T))

            E_idx_per_block.append(E_idx)
        return E_idx_per_block

    # for each tetrahedron, map the 4 vertices to 6 edges using local edge indices
    def _T_per_block_to_E_idx(self):
        T_into_E_idx_per_block = []
        for block, T_into_V in enumerate(self.T_into_V_per_block):
            E_into_V = self.E_into_V_per_block[block]
            E_into_sortedV = np.sort(E_into_V, axis=1)
            nT = np.shape(T_into_V)[0]
            T_into_E_V = np.zeros((nT, 6, 2), dtype=int)
            T_into_E_V[:, :, 0] = T_into_V[:, [0, 0, 0, 1, 2, 3]]
            T_into_E_V[:, :, 1] = T_into_V[:, [1, 2, 3, 2, 3, 1]]
            T_into_E_sortedV = np.sort(T_into_E_V, axis=-1)

            # very memory intensive!
            # T_into_E_idx = np.argwhere((T_into_E_sortedV[:,:,None,:] == E_into_sortedV[None,None,:,:]).all(axis=-1))[:,-1].reshape(nT, 6)

            hshTbl = {}
            for idx, sedge in enumerate(E_into_sortedV):
                hsh = hash(bytes(sedge))
                hshTbl[hsh] = idx
            T_into_E_idx = np.zeros(nT*6, dtype=int)
            for idx, sedge in enumerate(T_into_E_sortedV.reshape(-1, 2)):
                hsh = hash(bytes(sedge))
                T_into_E_idx[idx] = hshTbl[hsh]

            T_into_E_idx_per_block.append(T_into_E_idx.reshape(nT, 6))
        return T_into_E_idx_per_block

    def T_vols(self):
        return igl.volume(self.V_into_xyz, self.T_into_V)

    def T_vols_per_block(self):
        return [igl.volume(self.V_into_xyz, T_into_V) for T_into_V in self.T_into_V_per_block]

    def E_lengths(self, E_into_V):
        V1, V2 = E_into_V[:, 0], E_into_V[:, 1]
        P1, P2 = self.V_into_xyz[V1, :], self.V_into_xyz[V2, :]
        E_lengths = np.linalg.norm(P1 - P2, axis=1)
        return E_lengths[:, None]

    def T_into_E_lengths_per_block(self):
        E_lengths_per_block = [self.E_lengths(E_into_V) for E_into_V in self.E_into_V_per_block]
        T_into_E_lengths_per_block = [E_lengths_per_block[block][T_into_E_idx] for block, T_into_E_idx in enumerate(self.T_into_E_idx_per_block)]
        return T_into_E_lengths_per_block

    def distance(self, V1, V2):
        P1, P2 = self.V_into_xyz[V1], self.V_into_xyz[V2]
        return np.linalg.norm(P1 - P2)

