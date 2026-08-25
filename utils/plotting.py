import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

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
                    edgecolor='k', linewidth=linewidth, alpha=options.get('alpha', 0.5), cmap=options.get('cmap'))

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
    
def plot_meshes(meshes, faces, options=None):
    '''
    Plots several meshes (sharing faces) on the same axes
    '''
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    for mesh in meshes:
        plot_mesh(mesh, faces, ax=ax, options=options)
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

# FIGURE HELPERS

FIGURE_OPTIONS = {'clean': True, 'view': True, 'boxaspect': True}

def plot_mesh_row(meshes, faces, cmaps, options=None, titles=None, figsize=None, shared_scale=True):
    '''
    Plots meshes side by side in one row, top-down and without axes.
    cmaps: one colormap name per mesh (e.g. 'Blues' for examples, 'Greens' for generated meshes)
    shared_scale: draw every mesh at the same scale (so a sequence reads as one motion)
    '''
    options = {**FIGURE_OPTIONS, **(options or {})}
    meshes = [np.asarray(m) for m in meshes]
    all_vertices = np.concatenate([m[:-len(faces)] for m in meshes])
    lo, hi = all_vertices.min(axis=0), all_vertices.max(axis=0)
    span = (hi - lo).max()

    fig = plt.figure(figsize=figsize or (3 * len(meshes), 3))
    for i, mesh in enumerate(meshes):
        ax = fig.add_subplot(1, len(meshes), i + 1, projection='3d')
        plot_mesh(mesh, faces, ax=ax, options={**options, 'cmap': cmaps[i]},
                  title=titles[i] if titles else None)
        lo_i, hi_i = mesh[:-len(faces)].min(axis=0), mesh[:-len(faces)].max(axis=0)
        center = (lo_i + hi_i) / 2
        if not shared_scale:
            span = (hi_i - lo_i).max()
        ax.set_xlim(center[0] - span/2, center[0] + span/2)
        ax.set_ylim(center[1] - span/2, center[1] + span/2)
        ax.set_zlim(center[2] - span/2, center[2] + span/2)
        ax.set_box_aspect([1, 1, 1])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    return fig

def plot_mesh_overlay(meshes, faces, cmaps, alphas=None, options=None, figsize=(6, 6)):
    '''
    Plots meshes on top of each other in one axes, top-down and without axes.
    alphas: transparency per mesh, e.g. low for example meshes and high for generated meshes
    '''
    options = {**FIGURE_OPTIONS, **(options or {})}
    if alphas is None:
        alphas = [0.5] * len(meshes)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    for mesh, cmap, alpha in zip(meshes, cmaps, alphas):
        plot_mesh(mesh, faces, ax=ax, options={**options, 'cmap': cmap, 'alpha': alpha})
    all_vertices = np.concatenate([np.asarray(m)[:-len(faces)] for m in meshes])
    lo, hi = all_vertices.min(axis=0), all_vertices.max(axis=0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig
