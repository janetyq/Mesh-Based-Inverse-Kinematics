# Mesh Based Inverse Kinematics
Implementation of method described in the paper "Mesh-Based Inverse Kinematics" by Sumner et al. 2005.
This method learns the features of a mesh through example poses and uses these features to generate new 
meaningful poses that satisfy user inputed position constraints. 

Techniques used: mesh processing, linear algebra, optimization

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). `uv sync` creates a project-local `.venv`
with numpy/scipy/matplotlib and the dev tools (pytest).

```
uv sync
uv run python meshik.py   # run the tube example
uv run pytest             # run the tests
```

`tests/golden/tube.npz` pins the tube example's output; regenerate it with `uv run python tests/make_golden.py`
only when a change in behaviour is intended.

## Work in progress

Todo
- Refactor/organize/comment code
- Implement rodrigues exponential map
- Include different ways of solving (ex. interpolation without constraints)
- Vectorize nonlinear combination of feature vectors
- Produce some cool examples
- Add a nice explanation of the method