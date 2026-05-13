# =============================================================================
# Prototype: Scalar Laplacian Residual-based Boundary API
# =============================================================================
#
# Integration test: Scalar Laplacian with a Nonlinear Robin BC
# defined through the new ResidualBC / BoundaryAction API.
# =============================================================================

using XCALibre
using LinearAlgebra
using Printf
using SparseArrays
using StaticArrays

# ── LAYER 1: Residual-based Boundary Logic ───────────────────────────────────
# We define a NonlinearRobinBC that returns Residual and Jacobian actions 
# based on a scalar field value.

abstract type ResidualBC end

struct NonlinearRobinBC <: ResidualBC
    row::Int
    target::Float64
end

# Action definitions
abstract type BoundaryAction end
struct AddDiagonal <: BoundaryAction; row::Int; value::Float64; end
struct AddSource <: BoundaryAction; row::Int; value::Float64; end

function inject!(A::SparseMatrixCSC, b::AbstractVector, action::AddDiagonal)
    A[action.row, action.row] += action.value
end
function inject!(A::SparseMatrixCSC, b::AbstractVector, action::AddSource)
    b[action.row] += action.value
end

# Residual semantic: R = u + u^2 - target = 0
# Jacobian: dR/du = 1 + 2u
function get_residual_actions(bc::NonlinearRobinBC, u_curr::Float64)
    resid = u_curr + u_curr^2 - bc.target
    jac   = 1.0 + 2.0 * u_curr
    return [AddDiagonal(bc.row, jac), AddSource(bc.row, -resid)]
end

# ── LAYER 2: FV Assembly (Integration) ───────────────────────────────────────

function assemble_system!(mesh, phi, bcs)
    n = length(mesh.cells)
    A = spzeros(n, n)
    b = zeros(n)
    
    # Internal faces assembly (Dummy Laplacian)
    for cID in 1:n
        A[cID, cID] = 4.0
        b[cID] = 1.0
    end
    
    # NEW API: Residual-based BC assembly
    for bc in bcs
        u_curr = phi.values[bc.row]
        actions = get_residual_actions(bc, u_curr)
        for action in actions
            inject!(A, b, action)
        end
    end
    
    return A, b
end

# ── 3. Newton Linearisation Flow ─────────────────────────────────────────────

function run_test()
    mesh_cpu = adapt(CPU(), UNV2D_mesh(joinpath(pkgdir(XCALibre, "examples", "0_GRIDS"), "quad40.unv"), scale=0.025))
    u = ScalarField(mesh_cpu); initialise!(u, 2.0) # initial guess
    bc = NonlinearRobinBC(1, 10.0) # nonlinear Robin on first cell
    
    @info "Starting Newton loop..."
    for iter in 1:5
        A, b = assemble_system!(mesh_cpu, u, [bc])
        
        # Newton update step: A * delta_u = b (since b represents the residual)
        delta_u = A \ b
        
        # Field update
        u.values .+= delta_u
        
        res = norm(b)
        @printf("Iteration %d: u[1]=%.6f, res=%.2e\n", iter, u.values[1], res)
        
        if res < 1e-8
            break
        end
    end
    @printf("Final solution u[1] = %.6f (True root of u^2+u-10=0 is 2.701562)\n", u.values[1])
end

run_test()
