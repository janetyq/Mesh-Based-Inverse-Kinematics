import numpy as np

def gen_off(name, vertices, faces, edges):
    '''
    Generates an .off file from a list of vertices, faces, and edges
    name: name of the file
    vertices: list of vertices
    faces: list of faces
    edges: list of edges
    returns: nothing
    '''
    with open('meshes/' + name + '.off', 'w') as file:
        file.write('OFF\n')
        nv = len(vertices)
        nf = len(faces)
        ne = len(edges)
        file.write(f'{nv} {nf} {ne}\n')
        for v in vertices:
            file.write(f'{v[0]} {v[1]} {v[2]}\n')
        for f in faces:
            file.write(f'3 {f[0]} {f[1]} {f[2]}\n')
    print(f'Generated {name}.off')

def read_off(off_file):
    with open("meshes/" + off_file, 'r') as f:
        lines = f.readlines()
        vertices = []
        faces = []
        for i, line in enumerate(lines):
            if i == 0:
                if line.strip() != 'OFF':
                    raise Exception('Not an OFF file')
            elif i == 1:
                nv, nf, ne = [int(x) for x in line.strip().split(' ')]
            elif i <= nv+1:
                vertices.append([float(x) for x in line.strip().split(' ')])
            else:
                faces.append([int(x) for x in line.strip().split(' ')][1:])
    return np.array(vertices), np.array(faces)

def read_obj(file_name):
    vertices = []
    faces = []
    
    with open('meshes/' + file_name, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertex = list(map(float, line[2:].strip().split()))
                vertices.append(vertex)
            elif line.startswith('f '):
                face = line[2:].strip().split()
                face_indices = []
                if len(face) == 4:  # Quadrilateral face
                    # Convert the quadrilateral to two triangular faces
                    vertex_indices = [int(vertex_data.split('/')[0]) - 1 for vertex_data in face]
                    face_indices.append([vertex_indices[0], vertex_indices[1], vertex_indices[2]])
                    face_indices.append([vertex_indices[0], vertex_indices[2], vertex_indices[3]])
                elif len(face) == 3:  # Triangular face
                    face_indices = [[int(vertex_data.split('/')[0]) - 1 for vertex_data in face]]
                else:
                    # Skip faces with more than four vertices or unsupported format
                    continue
                faces.extend(face_indices)
    
    return np.array(vertices), np.array(faces)



if __name__ == '__main__':
    from info import *
    from plotting import *
    # Example usage
    plotting_options = {'cmap': 'Blues', 'boxaspect': True, 'view': True, 'no_extra_vertices': True, "clean": True}
    file_paths = 'cat_000001.obj', 'cat_000011.obj', 'cat_000013.obj', 'cat_000015.obj','cat_000020.obj', 'cat_000024.obj','cat_000033.obj'
    meshes = []
    for file in file_paths:
        vertices, faces = read_obj(file)
        meshes.append(vertices)
    plot_mesh2_multiple(meshes, faces, options=plotting_options, titles=file_paths)
    plt.show()
    
    
    print("done")