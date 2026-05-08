# LAMMPS Tetrahedral Mass-Spring Model (`tetmsm`)

Custom LAMMPS `bond_style` and `improper_style` to implement a
tetrahedral mass-spring model (MSM) for isotropic linear-elastic solids
with arbitrary Poisson's ratio. A Python package is additionally
provided to generate LAMMPS data files.

There are two implementations provided: `volume` applies the penalty on
each improper, while `nodal` applies the penalty on each atom.

> **Note:** This project is under active development and not yet in its
> final form.

## Table of contents

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Model](#model)
- [Python pipeline](#python-pipeline)
- [Usage](#usage)
- [Validation](#validation)
- [References](#references)

## Overview

A tetrahedral solid mesh is represented as point masses connected by harmonic
springs (bonds) and subject to a volumetric energy penalty on each tetrahedron
(impropers). The two styles must be used together. The `improper_style` takes
the shear modulus $G$ and Poisson's ratio $\nu$ as coefficients, and the
`bond_style` reads $G$ from the improper style and scans the tetrahedral topology
to compute each bond's stiffness automatically. No bond coefficients must be provided.
However, both bond (edge) and improper (tetrahedron) topologies must be defined
in the LAMMPS data file.

The `tetmsm` Python package contains helpful tools to generate LAMMPS data files.

**Note:** The code has only been tested for a single-material, single-component solid.

## Repository structure

```
README.md
LICENSE
lammps-src/         # Contains two different implementations
  nodal/
    bond_tetmsm.h
    bond_tetmsm.cpp
    improper_tetmsm.h
    improper_tetmsm.cpp
    fix_tetmsm_comm.h
    fix_tetmsm_comm.cpp
    bond_tetmsm.rst
    improper_tetmsm.rst
  volume/
    bond_tetmsm.h
    bond_tetmsm.cpp
    improper_tetmsm.h
    improper_tetmsm.cpp
    bond_tetmsm.rst
    improper_tetmsm.rst
python-write-lam/
  pyproject.toml
  tetmsm/
    __init__.py
    geometry.py
    model.py
    writer.py
    msh/            # Gmsh input meshes
    lam/            # generated LAMMPS data files
    cube.py         # example: generate .lam files for a unit cube .msh
tests/
    nodal/
    volume/
```

## Installation

Clone the git repository.

#### Python package

```bash
cd python-write-lam
pip install -e .
```

#### LAMMPS styles

Copy the `.h`/`.cpp` files from `lammps-src/<volume|nodal>/` to your LAMMPS `src/` folder,
then rebuild. No additional packages are required.

#### LAMMPS documentation

Copy the `.rst` files from `lammps-src/<volume|nodal>/` to your
LAMMPS `doc/src/` folder, then rebuild the docs.

## Model

### Edge springs

Lloyd et al. (2007) derived optimal spring constants for an MSM
that best approximates the finite-element stiffness matrix of a
constant-strain tetrahedron by minimizing the squared difference
between the FEM stiffness matrix and the corresponding entries of
the MSM stiffness matrix.
For an irregular tetrahedron element $\Delta$ of volume $V$, they define an
equivalent edge length $\tilde{l}_\Delta := (12 V / \sqrt{2})^{1/3}$.
Then the spring stiffness of an edge $e$ sums contributions from all
adjacent tetrahedra:

$$
k_e = \frac{2\sqrt{2}}{25} E \sum_{\Delta}^{\text{adj.\ tets}} \tilde{l}_\Delta. \qquad (1)
$$

This is their *optimization approach*. They also introduced a *hybrid approach*
that adds a volume correction term, making the MSM stiffness matrix denser and
closer to a typical FEM stiffness matrix.
One condition of their procedure is that Poisson's ratio must be $\nu = 1/4$
(related to the Cauchy relation for 3D central-force networks).
Our goal is to extend this to arbitrary Poisson's ratio.

Following the arguments of Golec et al. (2020), the energy density of a
Poisson/Cauchy solid ($\nu = 1/4$) is

$$
W^{(e)} = G \left( \frac{1}{2} (\text{tr}\boldsymbol{\varepsilon})^2 + \boldsymbol{\varepsilon} : \boldsymbol{\varepsilon} \right), \qquad (2)
$$

where $G$ is the shear modulus.
Since $G$ alone parameterizes the edge spring stiffness ($E = 2G(1+\nu)$),
we can rewrite Eq. 1 as

$$
k_e = \frac{\sqrt{2}}{5} G \sum_{\Delta}^{\text{adj.\ tets}} \tilde{l}_\Delta, \qquad (3)
$$

which we expect to be valid for arbitrary $\nu$. A unique
edge $e$ of the mesh has bond energy

$$
U_e = \frac{1}{2} k_e \left( r - r_0 \right)^2. \qquad (4)
$$

### Volume penalty

Poisson's ratio can be tuned by adding a volume energy density (Golec et al. 2020)

$$
W^{(v)} = \frac{1}{2} \kappa (\text{tr}\boldsymbol{\varepsilon})^2, \qquad (5)
$$

where $\kappa := K - \tfrac{5}{3}G$, with $K$ being the bulk modulus.
Substituting $K = 2G(1+\nu)/[3(1-2\nu)]$, the volume stiffness becomes

$$
\kappa_V \equiv \kappa = \frac{G\,(4\nu - 1)}{1 - 2\nu}. \qquad (6)
$$

At $\nu = 1/4$, $\kappa_V = 0$ and the model reduces to Lloyd's springs-only MSM.
Two formulations of the discrete volume penalty are provided.

#### Per-tetrahedron (`volume`)

The volume penalty energy is applied independently to each tetrahedron $\Delta$:

$$
U_\Delta^{\mathrm{vol}} = \frac{1}{2}\,\kappa_V\,V_{0,\Delta}\left(1 - \frac{V_\Delta}{V_{0,\Delta}}\right)^2, \qquad (7)
$$

where $V_{0,\Delta}$ is the reference volume and $V_\Delta$ is the current
(signed) volume. The force on vertex $k$ is

$$
\mathbf{f}_k = -\kappa_V\,\frac{V_\Delta - V_{0,\Delta}}{V_{0,\Delta}}\,\frac{\partial V_\Delta}{\partial \mathbf{x}_k}. \qquad (8)
$$

This formulation is simple and local: each improper interaction involves
only its four vertices, with no inter-element communication.
A known limitation is **volumetric locking** as $\nu \to 0.5$: the number of
per-tet constraints (approximately 5–6 per node on structured tetrahedral
meshes) exceeds the three translational DOFs per node, causing
artificial stiffening.

#### Nodal-averaged (`nodal`)

Following the average nodal pressure (ANP) concept of
Bonet and Burton (1998), a nodal volume is defined for each atom $i$:

$$
V_i = \frac{1}{4}\sum_{\Delta \in \mathcal{T}_i} V_\Delta, \qquad (9)
$$

where $\mathcal{T}_i$ is the set of tetrahedra sharing node $i$.
The nodal volumetric strain is $\varepsilon_i = (V_i - V_{0,i})/V_{0,i}$,
and the energy per node is

$$
U_i^{\mathrm{vol}} = \frac{1}{2}\,\kappa_V\,V_{0,i}\,\varepsilon_i^2. \qquad (10)
$$

The force on node $k$ of tetrahedron $\Delta$ is

$$
\mathbf{f}_k^{(\Delta)} = -\frac{S_\Delta}{4}\,\frac{\partial V_\Delta}{\partial \mathbf{x}_k}, \qquad
S_\Delta = \kappa_V \sum_{v \in \mathrm{verts}(\Delta)} \varepsilon_v. \qquad (11)
$$

This formulation reduces the number of independent volumetric
constraints (one per node instead of one per tet), which in FEM
alleviates locking near $\nu = 0.5$. However, the implementation requires
two passes over the improper list—first to accumulate nodal volumes
(with reverse/forward ghost communication), then to compute forces—and
all improper types must share the same $G$ and $\nu$.

> **Note:** In our beam validation tests, the nodal-averaged formulation
> produces systematically incorrect elastic moduli and Poisson's ratio.
> The averaging-before-squaring structure (Jensen's inequality) makes
> the penalty too soft on structured meshes, under-penalizing non-uniform
> volumetric deformation. The **`volume` implementation is recommended**
> for general use.

## Python pipeline

Both implementations can use the same data file. There are three modules:

**`tetmsm.geometry`** reads the mesh and builds connectivity:

- Parses `.msh` files via `meshio` (Gmsh v4 ASCII format).
- Extracts vertices, tetrahedra, and unique edges using `libigl`.
- Supports multiple cell blocks (physical groups in Gmsh) for
  multi-material meshes.
- Builds per-block edge–tet adjacency maps and identifies boundary
  facets and boundary vertices.

**`tetmsm.model`** assembles the physical model:

- `Composite` takes a `TetMesh` plus per-block material flags
  (elastic/rigid) and optional magnetization vectors.
  A `Part` in `model.py` is a [meshio](https://pypi.org/project/meshio/) cell block.
  A `Composite` is a list of cell blocks obtained from one `TetMesh` object.
- Generates the bond list (unique mesh edges among elastic cells) and
  the improper list (tetrahedra, one improper type per material region).
- Assigns atom types automatically: 1 = elastic bulk, 2 = elastic
  boundary, 3 = rigid bulk, 4 = rigid boundary, 5 = magnetic.
- Computes per-vertex mass using a lumped mass approximation.
- For rigid parts, assigns molecule IDs for use with `fix rigid`.
- For magnetic parts, computes fictitious surface charges from the
  magnetization vector $\mathbf{M}$ by projecting onto outward face
  normals (assuming $\nabla \cdot \mathbf{M} = 0$). Charges are
  lumped to vertices.

**`tetmsm.writer`** writes the LAMMPS data file:

- Atom style: `hybrid full sphere` (supports molID, charge, diam, rho).
- Writes `Atoms`, `Bonds`, and `Impropers` sections. 
- No coefficient sections -- material properties are set
  in the LAMMPS input script via `improper_coeff` command.
- Supports multiple `Composite` objects with correct offsets.
- Optionally writes a rigid-ID file for `fix rigid`.

## Usage

### Generating a LAMMPS data file

```python
from tetmsm import TetMesh, Composite, write_lam

tet_mesh = TetMesh("msh/cube-N16")

comp = Composite(
    tet_mesh,
    rho_per_cell_block=[1000.0],       # kg/m^3
    elastic_per_cell_block=[True],
)

write_lam([comp], filename="lam/cube-N16")
```

This reads `msh/cube-N16.msh`, extracts the tetrahedral topology, and
writes `lam/cube-N16.lam`. The material properties ($G$, $\nu$) are
*not* baked into the data file -- they are set in the LAMMPS input
script via `improper_coeff`, so the same `.lam` file can be reused for
different elastic constants.

### Minimal input script

```lammps
units         si
atom_style    hybrid full sphere
read_data     mesh.lam nocoeff

bond_style    tetmsm
bond_coeff    *                  # no arguments

improper_style tetmsm
improper_coeff * 1000.0 0.4     # G [pressure units]  nu

fix           1 all nve
run           10000
```

### Coefficients

| Style | Coefficients |
|---|---|
| `bond_style tetmsm` | none |
| `improper_style tetmsm` | $G$, $\nu$ (per type) |

**`bond_coeff`** accepts no arguments. The `bond_coeff *` command must
still be issued for all bond types, but the stiffness $\kappa_E$ is
computed internally from the improper topology and the shear modulus $G$.

**`improper_coeff`** takes two arguments per type:

| Argument | Symbol | Units | Description |
|---|---|---|---|
| 1 | $G$ | pressure | Shear modulus |
| 2 | $\nu$ | dimensionless | Poisson's ratio |

### Data file format

The data file must contain `Bonds` and `Impropers` sections. Each bond
connects two vertices sharing a tetrahedral edge. Each improper defines
one tetrahedron by its four vertices, ordered so that the signed volume
is positive.

```
Bonds

1 1 1 2
2 1 1 3
3 1 1 4
4 1 2 3
5 1 2 4
6 1 3 4

Impropers

1 1 1 2 3 4
```

A `Bond Coeffs` section is not needed in the data file. An
`Improper Coeffs` section with $G$ and $\nu$ can optionally be included.

### Multi-material meshes

For composite structures with different elastic regions, assign
separate improper types to tetrahedra in each material region:

```lammps
improper_coeff 1 1000.0 0.4    # soft matrix
improper_coeff 2 1e6   0.3     # stiff inclusion
```

Each bond automatically accumulates stiffness contributions from its
adjacent tetrahedra, weighted by their respective shear moduli. No
manual bookkeeping is required.

By default, each `Part` in `model.py` has the same improper type.
For the `volume` implementation, in principle it is possible to
assign a different $G$ and $\nu$ for each improper via types.
However, in the `nodal` implementation, this will not give a consistent
averaging procedure and thus requires the same $G$ and $\nu$ for all
impropers. (This could be relaxed later for composites.)

## Validation

Validation scripts are found in the `tests/` directory. 

## References

1. Lloyd, B., Szekely, G., Harders, M. "Identification of spring
   parameters for deformable object simulation." *IEEE Trans. Vis.
   Comput. Graph.*, 13(5), 1081-1094 (2007).
   [DOI: 10.1109/TVCG.2007.1055](https://doi.org/10.1109/TVCG.2007.1055)

2. Golec, K., Palierne, J.-F., Zara, F., Nicolle, S., Damiand, G.
   "Hybrid 3D mass-spring system for soft tissue simulation."
   *Vis. Comput.*, 36, 809-825 (2020).
   [DOI: 10.1007/s00371-019-01663-0](https://doi.org/10.1007/s00371-019-01663-0)

3. Kot, M., Nagahashi, H., Szymczak, P. "Elastic moduli of simple
   mass spring models." *Vis. Comput.*, 31, 1339-1350 (2015).
   [DOI: 10.1007/s00371-014-1015-5](https://doi.org/10.1007/s00371-014-1015-5)

4. Clemmer, J. T., Monti, S., Lechman, J. B. "A soft granular material
   model for particle-based numerical simulation of deformable linear
   elastic solids." *Soft Matter*, 20, 1702-1718 (2024).
   [DOI: 10.1039/D3SM01158E](https://doi.org/10.1039/D3SM01158E)

5. Bonet, J., Burton, A. J. "A simple average nodal pressure
   tetrahedral element for incompressible and nearly incompressible
   dynamic explicit applications." *Comput. Methods Appl. Mech.
   Engrg.*, 154(1-2), 73-81 (1998).
   [DOI: 10.1016/S0045-7825(97)00111-4](https://doi.org/10.1016/S0045-7825(97)00111-4)

6. Geuzaine, C., Remacle, J.-F. "Gmsh: A 3-D finite element mesh
   generator with built-in pre- and post-processing facilities."
   *Int. J. Numer. Meth. Engng.*, 79(11), 1309-1331 (2009).
   [https://gmsh.info](https://gmsh.info)

7. Schlömer, N. (2024). "meshio: Tools for mesh files (v5.3.5)." Zenodo.
   [https://doi.org/10.5281/zenodo.1288334](https://doi.org/10.5281/zenodo.1288334)

8. Jacobson, A. et al. "libigl: A simple C++ geometry processing library."
   [https://libigl.github.io/](https://libigl.github.io/)