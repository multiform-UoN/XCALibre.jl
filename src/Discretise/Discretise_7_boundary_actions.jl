export AbstractBoundaryAction
export SetEquationRow, SetNewtonRow, AddJacobianEntry, AddDiagonalEntry, AddRHSEntry
export AbstractResidualBC, LocalScalarResidualBC
export boundary_actions, residual_value
export apply_boundary_action!, apply_boundary_actions!

# Lightweight action language for examples and prototype backends.
#
# These actions deliberately describe algebraic contributions without knowing
# whether the consuming backend is CSC, CSR, matrix-free, or a future device
# kernel. The concrete backend provided here is SparseMatrixCSC row/entry
# mutation for small CPU tutorials and diagnostics, not a hot-loop GPU path.

abstract type AbstractBoundaryAction end

struct SetEquationRow{I,T} <: AbstractBoundaryAction
    row::I
    diagonal::T
    rhs::T
end

struct SetNewtonRow{I,T} <: AbstractBoundaryAction
    row::I
    jacobian::T
    residual::T
end

struct AddJacobianEntry{I,T} <: AbstractBoundaryAction
    row::I
    col::I
    value::T
end

struct AddDiagonalEntry{I,T} <: AbstractBoundaryAction
    row::I
    value::T
end

struct AddRHSEntry{I,T} <: AbstractBoundaryAction
    row::I
    value::T
end

abstract type AbstractResidualBC end

struct LocalScalarResidualBC{I,R,J} <: AbstractResidualBC
    row::I
    residual::R
    jacobian::J
end

LocalScalarResidualBC(row; residual, jacobian) =
    LocalScalarResidualBC(row, residual, jacobian)

@inline residual_value(bc::LocalScalarResidualBC, values::AbstractVector) =
    bc.residual(values[bc.row])

@inline function boundary_actions(bc::LocalScalarResidualBC, values::AbstractVector)
    u = values[bc.row]
    return (SetNewtonRow(bc.row, bc.jacobian(u), bc.residual(u)),)
end

boundary_actions(bc::AbstractResidualBC, values::AbstractVector) =
    error("No boundary_actions method defined for $(typeof(bc)).")

function _zero_sparse_row!(A::SparseMatrixCSC, row::Integer)
    A[row, :] .= zero(eltype(A))
    return nothing
end

function apply_boundary_action!(A::SparseMatrixCSC, b::AbstractVector, action::SetEquationRow)
    _zero_sparse_row!(A, action.row)
    A[action.row, action.row] = action.diagonal
    b[action.row] = action.rhs
    return nothing
end

function apply_boundary_action!(A::SparseMatrixCSC, b::AbstractVector, action::SetNewtonRow)
    _zero_sparse_row!(A, action.row)
    A[action.row, action.row] = action.jacobian
    b[action.row] = -action.residual
    return nothing
end

function apply_boundary_action!(A::SparseMatrixCSC, b::AbstractVector, action::AddJacobianEntry)
    A[action.row, action.col] += action.value
    return nothing
end

function apply_boundary_action!(A::SparseMatrixCSC, b::AbstractVector, action::AddDiagonalEntry)
    A[action.row, action.row] += action.value
    return nothing
end

function apply_boundary_action!(A::SparseMatrixCSC, b::AbstractVector, action::AddRHSEntry)
    b[action.row] += action.value
    return nothing
end

function apply_boundary_actions!(A, b, actions)
    for action in actions
        apply_boundary_action!(A, b, action)
    end
    return nothing
end
