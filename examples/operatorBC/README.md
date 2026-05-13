# Operator-First Boundary and Periodic Examples

These examples collect the architecture prototypes that are not specific to the
non-Newtonian benchmark suite.

- `boundary_action_sketch.jl` shows the minimal source-level action vocabulary.
- `residual_bc_sketch.jl` shows a nonlinear boundary residual lowered to a Newton row.
- `fv_residual_bc_laplacian.jl` applies the same residual/Jacobian action to a real FV Laplacian.
- `topological_periodic_stokes.jl` demonstrates periodicity as mesh topology rather than post-assembly matrix surgery.

The examples are deliberately explicit. They are intended to show where the
new architecture reduces bookkeeping without hiding the finite-volume operator
or the linear algebra being solved.
