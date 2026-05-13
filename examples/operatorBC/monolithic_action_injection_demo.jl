# =============================================================================
# Prototype: Monolithic Action Injection Demo
# =============================================================================
# WARNING: This script uses dummy operators (A[cID,cID]=4) for assembly
# verification. It does NOT yet use actual FV flux operators.
# =============================================================================

using XCALibre
using LinearAlgebra
using Printf
using SparseArrays
using StaticArrays

# ── 1. Prototype BC Residual Operators ───────────────────────────────────────

struct RobinBC <: ResidualBC
    row::Int
    α::Float64
    β::Float64
    value::Float64
end

function get_residual_actions(bc::RobinBC, u_curr::Float64)
    # Algebraic residual prototype (handles αu - γ = 0). 
    # Note: β∂ₙu term is ignored in this simple algebraic test.
    resid = bc.α * u_curr - bc.value
    jac   = bc.α
    return [AddDiagonal(bc.row, jac), AddSource(bc.row, -resid)]
end

# ── 2. Monolithic Stokes Prototype ──────────────────────────────────────────

function run_stokes_monolithic_demo()
    mesh = adapt(CPU(), UNV2D_mesh(joinpath(pkgdir(XCALibre, "examples", "0_GRIDS"), "quad40.unv"), scale=0.025))
    u = ScalarField(mesh); initialise!(u, 0.0)
    v = ScalarField(mesh); initialise!(v, 0.0)
    p = ScalarField(mesh); initialise!(p, 0.0)
    
    # Boundary: Robin on U, Dirichlet on P
    bcs_u = [RobinBC(1, 1.0, 0.0, 1.0)] 
    
    # Assembly
    n = length(mesh.cells)
    A = spzeros(3*n, 3*n)
    b = zeros(3*n)
    
    # Stokes system: Laplacian blocks (dummy)
    for cID in 1:n
        A[cID, cID] = 4.0   # u-u
        A[cID+n, cID+n] = 4.0 # v-v
        b[cID] = 1.0
    end
    
    # Apply via API
    for bc in bcs_u
        actions = get_residual_actions(bc, u.values[bc.row])
        for a in actions; inject!(A, b, a); end
    end
    
    @info "Monolithic Assembly Complete" nnz=nnz(A)
end

run_stokes_monolithic_demo()
