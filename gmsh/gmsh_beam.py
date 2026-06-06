"""
Generate tetrahedral meshes of a unit cube at multiple refinement levels.
"""
import gmsh

Lx = Ly = 0.4
Lz = 2.4

layers = [4, 8]

for N in layers:
    gmsh.initialize()
    gmsh.model.add(f"beam-N{N}")

    gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz, tag=1)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(3, [1], tag=1, name="Beam")

    # Force uniform element size everywhere
    h = Lx / N
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

    gmsh.model.mesh.generate(3)

    fname = f"beam-N{N}.msh"
    gmsh.write(fname)

    # Print stats
    nodes = gmsh.model.mesh.getNodes()
    elems = gmsh.model.mesh.getElements(dim=3)
    n_nodes = len(nodes[0])
    n_tets = len(elems[2][0]) // 4
    print(f"N={N:3d}  h={h:.4f}  nodes={n_nodes:7d}  tets={n_tets:7d}  -> {fname}")

    gmsh.finalize()