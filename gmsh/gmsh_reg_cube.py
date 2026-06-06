"""
Generate tetrahedral meshes of a unit cube at multiple refinement levels.
"""

import gmsh
import os

layers = [2, 4, 8, 16]

for N in layers:
    gmsh.initialize()
    gmsh.model.add(f"reg-cube-N{N}")

    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1, tag=1)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(3, [1], tag=1, name="Cube")

    # Set N+1 points on every edge for exactly N layers per dimension
    curves = gmsh.model.getEntities(dim=1)
    for _, tag_c in curves:
        gmsh.model.mesh.setTransfiniteCurve(tag_c, N + 1)

    surfaces = gmsh.model.getEntities(dim=2)
    for _, tag_s in surfaces:
        gmsh.model.mesh.setTransfiniteSurface(tag_s)

    gmsh.model.mesh.setTransfiniteVolume(1)

    # Recombine OFF — we want tets, not hexes
    gmsh.option.setNumber("Mesh.RecombineAll", 0)

    gmsh.model.mesh.generate(3)

    fname = f"reg-cube-N{N}.msh"
    gmsh.write(fname)

    # Print stats
    nodes = gmsh.model.mesh.getNodes()
    elems = gmsh.model.mesh.getElements(dim=3)
    n_nodes = len(nodes[0])
    n_tets = len(elems[2][0]) // 4
    print(f"N={N:3d}  nodes={n_nodes:7d}  tets={n_tets:7d}  -> {fname}")

    gmsh.finalize()