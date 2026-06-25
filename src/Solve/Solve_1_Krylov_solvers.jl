export _workspace
export Cg, Cgs, Bicgstab, Gmres
export krylov_solve!

abstract type AbstractLinearSolver end

struct Cg <: AbstractLinearSolver end
struct Cgs <: AbstractLinearSolver end
struct Bicgstab <: AbstractLinearSolver end
struct Gmres <: AbstractLinearSolver end

# Krylov.jl workspace constructors
_workspace(::Cg, b) = Krylov.CgWorkspace(Krylov.KrylovConstructor(b))
_workspace(::Cgs, b) = Krylov.CgsWorkspace(Krylov.KrylovConstructor(b))
_workspace(::Bicgstab, b) = Krylov.BicgstabWorkspace(Krylov.KrylovConstructor(b))
_workspace(::Gmres, b) = Krylov.GmresWorkspace(Krylov.KrylovConstructor(b))

using LinearAlgebra: I

"""
    krylov_solve!(solver, A, b, x; M=I, itmax=1000, atol=1e-12, rtol=1e-10, ldiv=false, history=false)

Generic wrapper for Krylov.jl solve! methods.
"""
function krylov_solve!(
    solver, A, b, x=nothing;
    M=I, itmax=1000, atol=1e-12, rtol=1e-10, ldiv=false, history=false
)
    # Unwrap SparseXCSR if needed
    real_A = A isa SparseXCSR ? parent(A) : A

    # Map workspace to method
    if typeof(solver) <: Krylov.CgWorkspace
        if x !== nothing
            Krylov.cg!(solver, real_A, b, x; M=M, ldiv=ldiv, itmax=itmax, atol=atol, rtol=rtol, history=history)
        else
            Krylov.cg!(solver, real_A, b; M=M, ldiv=ldiv, itmax=itmax, atol=atol, rtol=rtol, history=history)
        end
    elseif typeof(solver) <: Krylov.CgsWorkspace
        if x !== nothing
            Krylov.cgs!(solver, real_A, b, x; M=M, ldiv=ldiv, itmax=itmax, atol=atol, rtol=rtol, history=history)
        else
            Krylov.cgs!(solver, real_A, b; M=M, ldiv=ldiv, itmax=itmax, atol=atol, rtol=rtol, history=history)
        end
    elseif typeof(solver) <: Krylov.BicgstabWorkspace
        if x !== nothing
            Krylov.bicgstab!(solver, real_A, b, x; M=M, ldiv=ldiv, itmax=itmax, atol=atol, rtol=rtol, history=history)
        else
            Krylov.bicgstab!(solver, real_A, b; M=M, ldiv=ldiv, itmax=itmax, atol=atol, rtol=rtol, history=history)
        end
    elseif typeof(solver) <: Krylov.GmresWorkspace
        if x !== nothing
            Krylov.gmres!(solver, real_A, b, x; M=M, ldiv=ldiv, itmax=itmax, atol=atol, rtol=rtol, history=history)
        else
            Krylov.gmres!(solver, real_A, b; M=M, ldiv=ldiv, itmax=itmax, atol=atol, rtol=rtol, history=history)
        end
    else
        # Fallback to generic
        if x !== nothing
            Krylov.krylov_solve!(solver, real_A, b, x; M=M, ldiv=ldiv, itmax=itmax, atol=atol, rtol=rtol, history=history)
        else
            Krylov.krylov_solve!(solver, real_A, b; M=M, ldiv=ldiv, itmax=itmax, atol=atol, rtol=rtol, history=history)
        end
    end
    return solver
end
