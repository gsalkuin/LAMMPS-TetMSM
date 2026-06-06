"""
Generate tetrahedral meshes of a unit cube at multiple refinement levels.
"""
import gmsh

L = 10.
W = 1.
H = 1.

N = 4

gmsh.initialize()
gmsh.model.add(f"beam-10x1x1-N{N}")

gmsh.model.occ.addBox(0, 0, 0, L, W, H, tag=1)
gmsh.model.occ.synchronize()

gmsh.model.addPhysicalGroup(3, [1], tag=1, name="Beam")

# Force uniform element size everywhere
t = H / N
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", t)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", t)

gmsh.model.mesh.generate(3)

fname = f"beam-10x1x1-N{N}.msh"
gmsh.write(fname)

# Print stats
nodes = gmsh.model.mesh.getNodes()
elems = gmsh.model.mesh.getElements(dim=3)
n_nodes = len(nodes[0])
n_tets = len(elems[2][0]) // 4
print(f"N={N:3d}  t={t:.4f}  nodes={n_nodes:7d}  tets={n_tets:7d}  -> {fname}")

gmsh.fltk.run()

gmsh.finalize()
