import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from matplotlib.gridspec import GridSpec


# TODO
# make plot_mesh plot meshes together

def plot_3d_vertices(vertices):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(-15, 15)
    ax.set_title('3D Scatter Plot')
    plt.show()

    
def plot_mesh(vertices, faces, ax=None, options=None, title=None):
    vertices = np.array(vertices)
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    if options is None:
        options = {}

    if options.get('clean'):
        ax.grid(False)  
        ax.axis('off')
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    if title is not None:
        ax.set_title(title)
    
    if options.get('view'):
        ax.view_init(elev=90, azim=-90)
    
    ax.autoscale(enable=True, axis='both', tight=True)
    if options.get('no_extra_vertices'):
        end = len(vertices)
    else:
        end = -len(faces)
    if options.get('boxaspect'):
        x_range = np.ptp(vertices[:end, 0])
        y_range = np.ptp(vertices[:end, 1])
        z_range = np.ptp(vertices[:end, 2])
        ax.set_box_aspect([x_range, y_range, z_range])
    elif options.get('axlim'):  # Only set axlim if not using boxaspect
        ax.set_xlim(-2, 10)
        ax.set_ylim(-2, 10)
        ax.set_zlim(-5, 5)
        
    
    if options.get('wireframe'):
        linewidth = 2
    else:
        linewidth = 1

    ax.plot_trisurf(vertices[:end, 0], vertices[:end, 1], vertices[:end, 2], triangles=faces[:, :3], 
                    edgecolor='k', linewidth=linewidth, alpha=0.5, cmap=options.get('cmap'))

    return ax

def x_to_vertices(x):
    return x.reshape((3, -1)).T

def plot_x_mesh(x, faces, ax=None, options=None, title=None, scatter_points=None, scatter_indices=None):
    vertices = x_to_vertices(x)
    ax = plot_mesh(vertices, faces, ax=ax, options=options, title=title)
    if scatter_points is not None:
        ax.scatter(scatter_points[:, 0], scatter_points[:, 1], scatter_points[:, 2], c='r', marker='o')
    if scatter_indices is not None:
        ax.scatter(vertices[scatter_indices, 0], vertices[scatter_indices, 1], vertices[scatter_indices, 2], c='r', marker='o')
    
def simple_plot_mesh(mesh):
    vertices, faces = mesh()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces[:, :3], edgecolor='k', linewidth=1, alpha=0.2, facecolor='none')

    x_range = np.ptp(vertices[:, 0])
    y_range = np.ptp(vertices[:, 1])
    z_range = np.ptp(vertices[:, 2])

    ax.set_box_aspect([x_range, y_range, z_range])

def plot_meshes(meshes, faces):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim(-2, 10)
    # ax.set_ylim(-2, 10)
    # ax.set_zlim(-5, 5)
    for mesh in meshes:
        plot_mesh(mesh, faces, ax, wireframe=False)
    return fig, ax

def plot_mesh_face_values(vertices, faces, weights, clean=False, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-2, 8)
    ax.set_ylim(-2, 8)
    ax.set_zlim(-2, 2)
    if clean:
        ax.view_init(elev=90, azim=-90)
        ax.grid(False)  
        ax.axis('off')
    if title:
        plt.title(title)

    # Create a Poly3DCollection object with the mesh faces
    collection = Poly3DCollection(vertices[faces[:, 0:3]], linewidths=0.2)

    # Set facecolors based on the weights
    collection.set_array(weights)
    collection.set_cmap('viridis')

    # Add the collection to the plot
    ax.add_collection(collection)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set colorbar
    cbar = fig.colorbar(collection)

def plot_x_mesh2_multiple(meshes, faces, titles=None, cmaps=None, options=None):
    fig = plt.figure()
    title = ""
    cmap = None
    num_plots = len(meshes)
    gs = GridSpec(1, num_plots, wspace=0.0, hspace=0.0, 
                  top=0.92, bottom=0.08, left=0.0, right=1.0) 
    for i in range(num_plots):
        ax = fig.add_subplot(gs[i], projection='3d')
        if titles:
            title = titles[i]
        if cmaps:
            cmap = cmaps[i]
        options.update({'cmap': cmap})
        plot_x_mesh2(meshes[i], faces, ax=ax, options=options, title=title)

def plot_mesh2_multiple(meshes, faces, titles=None, cmaps=None, options=None):
    fig = plt.figure()
    title = ""
    cmap = None
    num_plots = len(meshes)
    gs = GridSpec(1, num_plots, wspace=0.0, hspace=0.0, 
                  top=0.92, bottom=0.08, left=0.0, right=1.0) 
    for i in range(num_plots):
        ax = fig.add_subplot(gs[i], projection='3d')
        if titles:
            title = titles[i]
        if cmaps:
            cmap = cmaps[i]
        options.update({'cmap': cmap})
        plot_mesh2(meshes[i], faces, ax=ax, options=options, title=title)
