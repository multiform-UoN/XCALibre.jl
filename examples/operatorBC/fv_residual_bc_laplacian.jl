# =============================================================================
# ARCHITECTURE SKETCH — Residual/Action BC on a Scalar Laplacian system
# =============================================================================
# This sketch uses DUMMY diagonal assembly (A[i,i] = 4), NOT a real FV operator.
# Its purpose is to show how a residual/Jacobian boundary condition lowers to
# AddDiagonalEntry / AddRHSEntry actions and drives a Newton loop.
#
# For a real FV nonlinear BC integration, see:
#   examples/ADR/nonlinear_adr.jl  (NonLinearRobin with the @define_boundary path)
# =============================================================================

using LinearAlgebra
using Printf
using SparseArrays

# ── Residual/Jacobian BC ─────────────────────────────────────────────────────
# Nonlinear Robin: R(u) = u + u² - target = 0

struct NonlinearRobinBC
    row::Int
    target::Float64
end

function residual_jacobian(bc::NonlinearRobinBC, u::Float64)
    R = u + u^2 - bc.target
    J = 1.0 + 2.0*u
    return R, J
end

# ── Action types and injection ───────────────────────────────────────────────
# Deliberately minimal here to keep the sketch self-contained.
# In XCALibre these correspond to AddDiagonalEntry / AddRHSEntry from
# Discretise_7_boundary_actions.jl.

struct _AddDiag; row::Int; val::Float64; end
struct _AddRHS;  row::Int; val::Float64; end

function inject!(A::SparseMatrixCSC, b::AbstractVector, a::_AddDiag)
    A[a.row, a.row] += a.val
end
function inject!(A::SparseMatrixCSC, b::AbstractVector, a::_AddRHS)
    b[a.row] += a.val
end

function newton_actions(bc::NonlinearRobinBC, u_curr::Float64)
    R, J = residual_jacobian(bc, u_curr)
    return (_AddDiag(bc.row, J), _AddRHS(bc.row, -R))
end

# ── Dummy FV system assembly ─────────────────────────────────────────────────
# 9-cell grid (3×3), interior Laplacian diagonal = 4, RHS = 1.
# The Biharmonic-style diagonal is illustrative only.

function assemble_system!(n, u_vals, bcs)
    A = spzeros(n, n)
    b = zeros(n)
    for i in 1:n
        A[i, i] = 4.0
        b[i] = 1.0
    end
    for bc in bcs
        for action in newton_actions(bc, u_vals[bc.row])
            inject!(A, b, action)
        end
    end
    return A, b
end

# ── Newton loop ──────────────────────────────────────────────────────────────

function run_sketch()
    n = 9
    u = fill(2.0, n)   # initial guess
    bc = NonlinearRobinBC(1, 10.0)

    @info "Newton loop: R(u) = u + u² - 10 = 0, expected root ≈ 2.701562"
    for iter in 1:6
        A, b = assemble_system!(n, u, [bc])
        u .+= A \ b
        R, _ = residual_jacobian(bc, u[1])
        @printf("  iter %d: u[1]=%.8f  R=%.2e\n", iter, u[1], R)
        abs(R) < 1e-12 && break
    end
end

run_sketch()
