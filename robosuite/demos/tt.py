import trimesh



mesh = trimesh.load_mesh('/home/joonhyeok/robosuite/robosuite/models/assets/objects/meshes/laptop.stl')

print(mesh.bounding_box.extents)
print(mesh.bounding_box_oriented.primitive.extents)
mesh.show()