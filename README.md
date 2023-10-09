# Mesh Based Inverse Kinematics
Implementation of method described in the paper "Mesh-Based Inverse Kinematics" by Sumner et al. 2005.
This method learns the features of a mesh through example poses and uses these features to generate new 
meaningful poses that satisfy user inputed position constraints. 

Techniques used: mesh processing, linear algebra, optimization

Run 'meshik.py' to see an example.

## Work in progress

Todo
- Refactor/organize/comment code
- Implement rodrigues exponential map
- Include different ways of solving (ex. interpolation without constraints)
- Vectorize nonlinear combination of feature vectors
- Produce some cool examples
- Add a nice explanation of the method