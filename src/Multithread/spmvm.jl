export SparseXCSR 
export activate_multithread

struct SparseXCSR{Bi,Tv,Ti,N} <: AbstractMatrix{Tv}
    parent::SparseMatrixCSR{Bi,Tv,Ti}
end

SparseXCSR(A::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti} = SparseXCSR{Bi,Tv,Ti,2}(A)

# Now add methods for the wrapper type SparseXCSR
Base.parent(A::SparseXCSR) = A.parent
Base.size(A::SparseXCSR) = size(parent(A))
KernelAbstractions.get_backend(A::SparseXCSR) = get_backend(A.parent.nzval)
Base.show(io::IO, A::SparseXCSR) = begin
    print(io, "CSR Matrix with $(length(A.parent.nzval)) entries")
end

function xmul!(
    y::AbstractVector, Ax::SparseXCSR, x::AbstractVector, alpha::Number, beta::Number)
    
    A = parent(Ax)
    A.n == size(x, 1) || throw(DimensionMismatch())
    A.m == size(y, 1) || throw(DimensionMismatch())

    o = getoffset(A)
    m = size(y, 1)

    # Simple loop for CPU to avoid any threading issues during debugging
    for row in 1:m
        accu = zero(eltype(y))
        for nz in nzrange(A, row)
            col = A.colval[nz] + o
            accu += A.nzval[nz] * x[col]
        end
        y[row] = alpha * accu + beta * y[row]
    end

    return y
end

"""
    activate_multithread(backend::CPU; nthreads=1) = BLAS.set_num_threads(nthreads)

Convenience function to set number of BLAS threads. 
    
# Input arguments

- `backend` is the only required input which must be `CPU()` from `KernelAbstractions.jl`
- `nthreads` can be used to set the number of BLAS cores (default `nthreads=1`)
"""
activate_multithread(backend::CPU; nthreads=1) = BLAS.set_num_threads(nthreads)


# Extend multiplications methods in LinearAlgebra and Base

function LinearAlgebra.mul!(y::AbstractVector, A::SparseXCSR, x::AbstractVector, alpha::Number, beta::Number)
    return xmul!(y, A, x, alpha, beta)
end

function LinearAlgebra.mul!(y::AbstractVector, A::SparseXCSR, x::AbstractVector)
    return xmul!(y, A, x, true, false)
end

function Base.:*(A::SparseXCSR, x::AbstractVector)
    y = similar(x)
    mul!(y, A, x)
    return y
end
