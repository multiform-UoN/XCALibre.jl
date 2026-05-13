# =============================================================================
# Boundary Action Sketch
# =============================================================================
#
# This is the smallest possible demonstration of the source-level action types.
# A boundary condition describes algebraic intent; the matrix backend consumes it.
# =============================================================================

using XCALibre
using LinearAlgebra
using Printf
using SparseArrays

n = 5
A = spdiagm(0 => ones(n))
b = ones(n)

actions = (
    SetEquationRow(1, 1.0, 10.0),
    SetEquationRow(5, 1.0, 20.0),
)

apply_boundary_actions!(A, b, actions)

x = A \ b

@printf("SetEquationRow demo: x[1]=%.2f, x[5]=%.2f\n", x[1], x[5])
