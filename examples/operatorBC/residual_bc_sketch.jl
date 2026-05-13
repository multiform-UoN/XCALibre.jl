# =============================================================================
# Residual/Jacobian BC Sketch
# =============================================================================
#
# A nonlinear boundary equation B(u)=0 becomes one Newton row:
#
#     B'(u_k) * du = -B(u_k)
#
# This sketch stays algebraic on purpose; fv_residual_bc_laplacian.jl applies the
# same source-level actions to a real finite-volume operator.
# =============================================================================

using XCALibre
using LinearAlgebra
using Printf
using SparseArrays

n = 3
A = spdiagm(0 => ones(n))
state = zeros(n)
state[1] = 2.0

bc = LocalScalarResidualBC(
    1;
    residual = u -> u + u^2 - 10.0,
    jacobian = u -> 1.0 + 2.0*u,
)

for iter in 1:4
    J = copy(A)
    rhs = zeros(n)
    apply_boundary_actions!(J, rhs, boundary_actions(bc, state))

    correction = J \ rhs
    state .+= correction

    @printf("iter %d: u=%.8f, B(u)=%.3e\n",
            iter, state[1], residual_value(bc, state))
end
