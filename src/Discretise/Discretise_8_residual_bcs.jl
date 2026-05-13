# =============================================================================
# Residual-based Boundary Condition Framework
# =============================================================================
#
# This file defines the transition from BC "Surgeons" (A[i,j] += ...) to
# BC "Residual Operators".
#
# Semantic:
#   boundary_face_residual!(bc, face, ...) -> returns residual contribution
#   boundary_face_jacobian!(bc, face, ...) -> returns jacobian contribution (optional)
# =============================================================================

abstract type ResidualBC end

# The Clerk: Dispatches residual/Jacobian to the backend
# Matches the signature of internal face loops
function assemble_boundary_residual!(
    A, b, bc::ResidualBC, face, cell, time, config
)
    # 1. Calculate Residual
    res_contrib = boundary_face_residual!(bc, face, cell, time)
    # 2. Calculate Jacobian (Linearization)
    jac_contrib = boundary_face_jacobian!(bc, face, cell, time)
    
    # 3. Backend Injection (Action Layer)
    # inject_boundary!(A, b, bc.row, res_contrib, jac_contrib)
end

# Example: Neumann BC
struct NeumannBC <: ResidualBC
    row::Int
    flux::Float64
end

@inline function boundary_face_residual!(bc::NeumannBC, face, cell, time)
    # Flux balance: Integral(phi_f * n) = -flux
    return bc.flux * face.area
end

@inline function boundary_face_jacobian!(bc::NeumannBC, face, cell, time)
    # Neumann is often constant flux (zero Jacobian contribution)
    return 0.0
end
