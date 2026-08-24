# Mesh Based Inverse Kinematics
This repo is a re-implementation of the method described in Mesh-Based Inverse Kinematics by Sumner et al. (2005). At a high level, the method learns from a set of example mesh poses and uses them to generate natural interpolations and new poses. It represents each pose with a feature vector describing the local deformation of each triangle, then interpolates these deformations by separating them into rotation and scale/shear components. It interpolates scale and shear linearly, while rotations are interpolated using an axis-angle representation.

I originally implemented the main part of this project for MIT 6.8410: Shape Analysis in Spring 2023. Since then, I've done some cleanup and small improvements to make the repo a bit easier to follow and use.

## Method

Each example mesh is described by a feature vector: the affine transformation (deformation gradient) of every
face relative to the reference mesh. The feature vector is a linear function of the vertex positions, `f = G x`,
where `G` depends only on the reference mesh. A fourth vertex is added to each face along its normal so that the
transformation is fully determined (Sumner and Popovic 2004).

The example feature vectors span a "feature space" of meaningful poses. Given position constraints on some
vertices, MeshIK finds the vertex positions `x` and blend weights `w` such that `G x` is as close as possible
to the blend of example features `M(w)`:

```
min ||G x - M(w)||    subject to the constrained vertices
x, w
```

Blending features linearly works poorly for large rotations, so each face transformation is split into a
rotation and a shear (polar decomposition). Rotations are blended in log space and shears linearly. This makes
`M(w)` nonlinear in `w`, and the problem is solved with Gauss-Newton iterations.

![Linear feature blend vs nonlinear feature blend](images/tube_linear_vs_nonlinear.png)

## Examples

Interpolating in feature space between two cat poses, with no other constraints, gives intermediate poses of
the run cycle.

![Interpolation between two cat poses](images/cat_interpolation.png)

Run `meshik.py` to see a tube example with position constraints. `examples/make_readme_figures.py` regenerates
the figures.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). `uv sync` creates a project-local `.venv`
with numpy/scipy/matplotlib and the dev tools (pytest).

```
uv sync
uv run python meshik.py                          # run the tube example
uv run pytest                                    # run the tests
uv run python examples/make_readme_figures.py    # regenerate images/
```

`tests/golden/tube.npz` pins the tube example's output; regenerate it with `uv run python tests/make_golden.py`
only when a change in behaviour is intended.

## Code

| File | Contents |
|---|---|
| `meshik.py` | Example script: load meshes, set constraints, solve, plot |
| `model.py` | `MeshIKModel`: everything precomputed from the example meshes (`G`, feature vectors, rotation logs, shears) |
| `methods.py` | The solvers: linear feature blend, nonlinear feature blend (Gauss-Newton), solve for a target feature |
| `matrices.py` | Feature extractor `G`, feature vector <-> face transformations, polar decomposition, rotation log/exp, `M(w)` and its Jacobian |
| `constraints.py` | Build constraints and eliminate constrained vertices from the system |
| `preprocessing.py` | Load meshes, add the fourth vertex to each face |
| `utils/` | OFF/OBJ readers and matplotlib plotting |
| `meshes/` | Tube examples (`.off`) and cat run cycle poses (`meshes/cat/*.obj`) |
| `tests/` | pytest suite |

Meshes are stored as [OFF](https://en.wikipedia.org/wiki/OFF_(file_format)) files (a header line, vertex count
and face count, then one vertex or face per line). OBJ files are also read. The cat poses were exported from a
rigged Blender animation.

## Work in progress

Todo
- Implement rodrigues exponential map
- Vectorize nonlinear combination of feature vectors
- Sparse `G` and the block Cholesky solver from the paper
- Add a nice explanation of the method
