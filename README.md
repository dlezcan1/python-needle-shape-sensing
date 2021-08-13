# python-needle-shape-sensing
Author: Dimitri Lezcano

This is a library of functions for needle shape sensing meant.

## Package layout
### numerical
A library of numerical functions like integrations like Euler-Poincare integration.

* `simpson_vec_int`: a function to perform Simpson Vector integration
* `intEP_w0`: Integration of the Euler-Poincare equation given a intrinsic angular deformation

### cost_functions
A library of cost functions for shape-sensing optimizations

* `singlebend_cost`: A cost function to curvature measurements applied for the single-bend and single-layer homogenous tissue needle insertion

### geometry
A library of geometrical functions (mostly for rigid body mechanics)

* `hat`: perform the inverse `vee` operation
* `is_(so2|so3|se3)`: Determine if a matrix is of the for (so(2) | so(3) | se(3))
* `is_(SO2|SO3|SE3)`: Determine if a matrix is of the for (SO(2) | SO(3) | SE(3))
* `is_(skew)symm`: Determine if a matrix is (skew-)symmetric
* `vee`: Convert (SO(2) | SO(3) | SE(3)) matrix -> (so(2) | so(3) | se(3)) element

