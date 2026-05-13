# =============================================================================
# Experimental Boundary Action API
# =============================================================================
# This infrastructure decouples BC semantic logic (Residual/Jacobian) from
# backend matrix storage (CSR, Monolithic block, etc).
#
# DESIGN:
# 1. ResidualBCs compute mathematical contributions (Residual/Jacobian).
# 2. They return BoundaryActions (AddDiagonal, AddSource, etc).
# 3. Solver consumers inject actions into their specific matrix layout.

export BoundaryAction, SetRow, AddDiagonal, AddSource, inject!
export ResidualBC, get_residual_actions

abstract type BoundaryAction end
struct SetRow <: BoundaryAction; row::Int; value::Float64; end
struct AddDiagonal <: BoundaryAction; row::Int; value::Float64; end
struct AddSource <: BoundaryAction; row::Int; value::Float64; end

abstract type ResidualBC end

# Backend Injectors (The Clerk)
function inject!(A::SparseMatrixCSC, b::AbstractVector, action::SetRow)
    A[action.row, :] .= 0.0
    A[action.row, action.row] = 1.0
    b[action.row] = action.value
end

function inject!(A::SparseMatrixCSC, b::AbstractVector, action::AddDiagonal)
    A[action.row, action.row] += action.value
end

function inject!(A::SparseMatrixCSC, b::AbstractVector, action::AddSource)
    b[action.row] += action.value
end
