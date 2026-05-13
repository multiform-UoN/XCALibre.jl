# =============================================================================
# ARCHITECTURE SKETCH — Monolithic action injection
# =============================================================================
# WARNING: Uses dummy diagonal assembly (A[cID,cID] = 4), NOT a real FV operator.
# Demonstrates how residual/Jacobian BCs lower to actions and inject into a
# monolithic block-structured system.
#
# Uses the XCALibre action vocabulary from Discretise_7_boundary_actions.jl:
#   AddDiagonalEntry, AddRHSEntry, apply_boundary_action!
# =============================================================================

using XCALibre
using LinearAlgebra
using Printf
using SparseArrays
using StaticArrays

# ── Residual BC using Discretise_7 action types ───────────────────────────────

struct RobinResidualBC
    row::Int
    α::Float64
    value::Float64
end

function bc_actions(bc::RobinResidualBC, u_curr::Float64)
    R = bc.α * u_curr - bc.value
    J = bc.α
    return (AddDiagonalEntry(bc.row, J), AddRHSEntry(bc.row, -R))
end

# ── Monolithic Stokes prototype ───────────────────────────────────────────────

function run_stokes_monolithic_demo()
    mesh = adapt(CPU(), UNV2D_mesh(joinpath(pkgdir(XCALibre, "examples", "0_GRIDS"), "quad40.unv"), scale=0.025))
    u = ScalarField(mesh); initialise!(u, 0.0)
    v = ScalarField(mesh); initialise!(v, 0.0)

    bcs_u = [RobinResidualBC(1, 1.0, 1.0)]

    n = length(mesh.cells)
    A = spzeros(2*n, 2*n)
    b = zeros(2*n)

    # Dummy Laplacian diagonal for u and v blocks
    for cID in 1:n
        A[cID,     cID]     = 4.0
        A[cID + n, cID + n] = 4.0
        b[cID] = 1.0
    end

    # Apply BC actions via Discretise_7 API
    for bc in bcs_u
        for action in bc_actions(bc, u.values[bc.row])
            apply_boundary_action!(A, b, action)
        end
    end

    @info "Monolithic assembly (dummy Laplacian + RobinResidualBC)" nnz=nnz(A)
end

run_stokes_monolithic_demo()
