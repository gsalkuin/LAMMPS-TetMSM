"""
Generate tetrahedral meshes of a unit cube at multiple refinement levels.
"""
import gmsh

layers = [2, 4, 8, 16]

for N in layers:
    gmsh.initialize()
    gmsh.model.add(f"cube-N{N}")

    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1, tag=1)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(3, [1], tag=1, name="Cube")

    # Force uniform element size everywhere
    h = 1.0 / N
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

    gmsh.model.mesh.generate(3)

    fname = f"cube-N{N}.msh"
    gmsh.write(fname)

    # Print stats
    nodes = gmsh.model.mesh.getNodes()
    elems = gmsh.model.mesh.getElements(dim=3)
    n_nodes = len(nodes[0])
    n_tets = len(elems[2][0]) // 4
    print(f"N={N:3d}  h={h:.4f}  nodes={n_nodes:7d}  tets={n_tets:7d}  -> {fname}")

    gmsh.finalize()