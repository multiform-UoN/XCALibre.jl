export SolverSetup, Runtime, Schemes
export explicit_relaxation!, implicit_relaxation!, implicit_relaxation_diagdom!, setReference!
export solve_system!
export solve_equation!, solve_preassembled!
export residual, residual!, residual_norm, solve_residual, jvp!
export AdaptiveTimeStepping

import XCALibre.ModelFramework: →, PDEOperator

struct SolverSetup{
    F<:AbstractFloat,
    I<:Integer,
    S1<:AbstractLinearSolver,
    S2<:Union{Nothing, AbstractSmoother},
    PT<:PreconditionerType
    }
    solver::S1
    smoother::S2
    preconditioner::PT
    convergence::F
    relax::F
    limit::Union{Nothing, Tuple{F,F}}
    itmax::I
    atol::F
    rtol::F
end

const BoundaryCollection = Union{Tuple{Vararg{<:AbstractBoundary}}, AbstractVector{<:AbstractBoundary}}

(→)(L::PDEOperator, BCs::BoundaryCollection) = PDEOperator(L.templates, L.sources, BCs, L.setup)
(→)(L::PDEOperator, setup::SolverSetup) = PDEOperator(L.templates, L.sources, L.BCs, setup)
(→)(L::PDEOperator, x) = throw(ArgumentError(
    "PDEOperator can only be chained with boundary conditions or SolverSetup; got $(typeof(x))."
))

"""
    SolverSetup(; 
            # required keyword arguments 

            solver::S, 
            preconditioner::PT, 
            convergence, 
            relax,

            # optional keyword arguments

            float_type=Float64,
            smoother=nothing,
            limit=nothing,
            itmax::Integer=1000, 
            atol=(eps(_get_float(region)))^0.9,
            rtol=_get_float(region)(1e-1)

        ) where {S,PT<:PreconditionerType} = begin

            return SolverSetup(kwargs...)  
    end

This function is used to provide solver settings that will be used internally in XCALibre.jl. It returns a `SolverSetup` object with solver settings that are used internally by the flow solvers. 

# Input arguments

- `solver`: solver object from Krylov.jl and it could be one of `Bicgstab()`, `Cg()`, `Gmres()` which are re-exported in XCALibre.jl
- `preconditioner`: instance of preconditioner to be used e.g. Jacobi()
- `convergence` sets the stopping criteria of this field
- `relax`: specifies the relaxation factor to be used e.g. set to 1 for no relaxation
- `smoother`: specifies smoothing method to be applied before discretisation. `JacobiSmoother`: is currently the only choice (defaults to `nothing`)
- `limit`: used in some solvers to bound the solution within these limits e.g. (min, max). It defaults to `nothing`
- `itmax`: maximum number of iterations in a single solver pass (defaults to 1000) 
- `atol`: absolute tolerance for the solver (default to eps(FloatType)^0.9)
- `rtol`: set relative tolerance for the solver (defaults to 1e-1)
- `float_type`: specifies the floating point type to be used by the solver. It is also used to estimate the absolute tolerance for the solver (defaults to `Float64`)
"""
SolverSetup(;
        float_type=Float64,
        solver::S1, 
        smoother::S2=nothing,
        preconditioner::PT, 
        convergence, 
        relax, 
        limit=nothing,
        itmax::I=1000, 
        atol=(eps(float_type))^0.9,
        rtol=1e-1 |> float_type
        ) where{S1,S2,PT,I} = 
        SolverSetup{float_type,I,S1,S2,PT}(
            solver, smoother,preconditioner, 
            float_type(convergence), 
            float_type(relax), 
            limit,
            itmax, 
            float_type(atol),
            float_type(rtol))

struct AdaptiveTimeStepping{F<:AbstractFloat}
    maxCo::F
    minShrink::F
    maxGrow::F
end
Adapt.@adapt_structure AdaptiveTimeStepping

"""
    AdaptiveTimeStepping(; 
        # keyword arguments

        maxCo=0.75,
        minShrink=0.1,
        maxGrow=1.2
    )

Constructs an `AdaptiveTimeStepping` object used to control automatic time-step adjustment
based on the Courant number.

This struct is passed optionally to `Runtime` and enables adaptive time stepping in transient
simulations. If not provided, a fixed time step is used.

# Input arguments

- `maxCo::AbstractFloat`: target maximum Courant number. The time step will be adjusted
  such that the computed Courant number approaches this value.
- `minShrink::AbstractFloat`: lower bound on the multiplicative factor applied to the
  current time step. Prevents excessively large reductions in a single update.
- `maxGrow::AbstractFloat`: upper bound on the multiplicative factor applied to the
  current time step. Prevents excessive time-step growth.
"""
AdaptiveTimeStepping(;
    maxCo=0.75,
    minShrink=0.1,
    maxGrow=1.2
) = AdaptiveTimeStepping(float(maxCo), float(minShrink), float(maxGrow))

struct Runtime{I<:Integer,F<:AbstractFloat, V<:AbstractVector{F}, A<:Union{Nothing, AdaptiveTimeStepping}}
    iterations::I
    dt::V
    write_interval::I
    adaptive::A
end
Adapt.@adapt_structure Runtime

"""
    Runtime(; 
            # keyword arguments

            iterations::I, 
            write_interval::I, 
            time_step::N,
            adaptive::A
        ) where {I<:Integer,N<:Number} = begin
        
        # returned Runtime struct
        Runtime{I<:Integer,F<:AbstractFloat}
            (
                iterations=iterations, 
                dt=time_step, 
                write_interval=write_interval,
                adaptive=adaptive
            )
    end

This is a convenience function to set the top-level runtime information. The inputs are all keyword arguments and provide basic information to flow solvers just before running a simulation.

# Input arguments

- `iterations::Integer`: specifies the number of iterations in a simulation run.
- `write_interval::Integer`: defines how often simulation results are written to file (on the current working directory). The interval is currently based on number of iterations. Set to `-1` to run without writing results to file.
- `time_step::AbstractFloat`: the time step to use in the simulation. Notice that for steady solvers this is simply a counter and it is recommended to simply use `1`.
- `adaptive::Union{Nothing, AdaptiveTimeStepping}`: optionally enables adaptive time stepping. Pass an `AdaptiveTimeStepping` object to automatically adjust `dt` based on the Courant number during transient simulations. Defaults to `nothing`, meaning a fixed time step is used.

# Example

```julia
runtime = Runtime(iterations=2000, time_step=1, write_interval=2000)
```
"""
Runtime(; iterations::I,
          write_interval::I,
          time_step::N,
          adaptive=nothing) where {I<:Integer,N<:Number} = begin

    val = float(time_step)
    Runtime(iterations, [val], write_interval, adaptive)
end

# Set schemes function definition with default set variables
"""
    Schemes(;
        # keyword arguments and their default values
        time=SteadyState,
        divergence=Linear, 
        laplacian=Linear, 
        gradient=Gauss,
        limiter=nothing) = begin

        # Returns Schemes struct used to configure discretisation
        
        Schemes(
            time=time,
            divergence=divergence,
            laplacian=laplacian,
            gradient=gradient,
            limiter=limiter
        )   
    end

The `Schemes` struct is used at the top-level API to help users define discretisation schemes for every field solved.

# Inputs

- `time`: is used to set the time schemes (default is `SteadyState`)
- `divergence`: is used to set the divergence scheme (default is `Linear`) 
- `laplacian`: is used to set the laplacian scheme (default is `Linear`)
- `gradient`:  is used to set the gradient scheme (default is `Gauss`)
- `limiter`: is used to specify if gradient limiters should be used, currently supported limiters include `FaceBased` and `MFaceBased` (default is `nothing`)
"""
@kwdef struct Schemes
    time=SteadyState
    divergence=Linear
    laplacian=Linear
    gradient=Gauss
    limiter=nothing
end


function solve_equation!(
    eqn::ModelEquation{T,M,E,S,P}, config; rho_prev=_get_flux(eqn.model.terms[1]), time=nothing, ref=nothing, irelax=nothing
    ) where {T<:ScalarModel,M,E,S,P}

    phi = get_phi(eqn)
    setup = eqn.setup
    discretise!(eqn, phi, config; rho_prev=rho_prev)
    apply_boundary_conditions!(eqn, config; time=time)
    setReference!(eqn, ref, 1, config)
    if !isnothing(irelax)
        implicit_relaxation!(eqn, phi.values, irelax, nothing, config)
        # implicit_relaxation_diagdom!(eqn, phi.values, irelax, nothing, config)
    end
    if !isnothing(eqn.preconditioner)
        update_preconditioner!(eqn.preconditioner, phi.mesh, config)
    end
    res = solve_system!(eqn, setup, phi, nothing, config)
    return res
end

"""
    solve_preassembled!(eqn, config; time=nothing)

Apply boundary conditions and solve a scalar equation whose matrix and RHS have already
been assembled (e.g. via `assemble_matrix!` / `assemble_rhs!`).  Unlike `solve_equation!`
this does **not** call `discretise!`, so any manual modifications to `_b(eqn)` made after
assembly are preserved.  Used for Crank–Nicolson and other split-step schemes.
"""
function solve_preassembled!(
    eqn::ModelEquation{T,M,E,S,P}, config; time=nothing
    ) where {T<:ScalarModel,M,E,S,P}
    phi   = get_phi(eqn)
    setup = eqn.setup
    apply_boundary_conditions!(eqn, config; time=time)
    if !isnothing(eqn.preconditioner)
        update_preconditioner!(eqn.preconditioner, phi.mesh, config)
    end
    return solve_system!(eqn, setup, phi, nothing, config)
end

function solve_equation!(
    eqn::ModelEquation{T,M,E,S,P}, phi, phiBCs, solversetup, config; rho_prev=_get_flux(eqn.model.terms[1]), time=nothing, ref=nothing, irelax=nothing
    ) where {T<:ScalarModel,M,E,S,P}

    discretise!(eqn, phi, config; rho_prev=rho_prev)
    apply_boundary_conditions!(eqn, phiBCs, nothing, time, config)
    if length(eqn.model.terms) == 1 && typeof(eqn.model.terms[1]) <: Laplacian
        make_symmetric!(eqn, config) # added this to test stability of periodic boundaries
    end
    setReference!(eqn, ref, 1, config)
    if !isnothing(irelax)
        implicit_relaxation!(eqn, phi.values, irelax, nothing, config)
        # implicit_relaxation_diagdom!(eqn, phi.values, irelax, nothing, config)
    end
    if !isnothing(eqn.preconditioner)
        update_preconditioner!(eqn.preconditioner, phi.mesh, config)
    end
    res = solve_system!(eqn, solversetup, phi, nothing, config)
    return res
end

function solve_equation!(
    psiEqn::ModelEquation{T,M,E,S,P}, config; rho_prev=_get_flux(psiEqn.model.terms[1]), time=nothing
    ) where {T<:VectorModel,M,E,S,P}

    psi = get_phi(psiEqn)
    mesh = psi.mesh
    solversetup = psiEqn.setup

    discretise!(psiEqn, psi, config; rho_prev=rho_prev)
    update_equation!(psiEqn, config)

    apply_boundary_conditions!(psiEqn, config; time=time, component=XDir())
    implicit_relaxation_diagdom!(psiEqn, psi.x.values, solversetup.relax, XDir(), config)
    update_preconditioner!(psiEqn.preconditioner, mesh, config)
    resx = solve_system!(psiEqn, solversetup, psi.x, XDir(), config)

    update_equation!(psiEqn, config)

    apply_boundary_conditions!(psiEqn, config; time=time, component=YDir())
    implicit_relaxation_diagdom!(psiEqn, psi.y.values, solversetup.relax, YDir(), config)
    update_preconditioner!(psiEqn.preconditioner, mesh, config)
    resy = solve_system!(psiEqn, solversetup, psi.y, YDir(), config)

    # Z velocity calculations (3D Mesh only)
    resz = zero(_get_float(mesh))
    if typeof(mesh) <: Mesh3
        update_equation!(psiEqn, config)
        apply_boundary_conditions!(psiEqn, config; time=time, component=ZDir())
        implicit_relaxation_diagdom!(psiEqn, psi.z.values, solversetup.relax, ZDir(), config)
        update_preconditioner!(psiEqn.preconditioner, mesh, config)
        resz = solve_system!(psiEqn, solversetup, psi.z, ZDir(), config)
    end

    return resx, resy, resz
end

function solve_equation!(
    psiEqn::ModelEquation{T,M,E,S,P}, psi, psiBCs, solversetup, xdir, ydir, zdir, config; rho_prev=_get_flux(psiEqn.model.terms[1]), time=nothing
    ) where {T<:VectorModel,M,E,S,P}

    mesh = psi.mesh

    discretise!(psiEqn, psi, config; rho_prev=rho_prev)
    update_equation!(psiEqn, config)
    
    apply_boundary_conditions!(psiEqn, psiBCs, xdir, time, config)
    # implicit_relaxation!(psiEqn, psi.x.values, solversetup.relax, xdir, config)
    implicit_relaxation_diagdom!(psiEqn, psi.x.values, solversetup.relax, xdir, config)
    update_preconditioner!(psiEqn.preconditioner, mesh, config)
    resx = solve_system!(psiEqn, solversetup, psi.x, xdir, config)
    
    update_equation!(psiEqn, config)
    apply_boundary_conditions!(psiEqn, psiBCs, ydir, time, config)
    # implicit_relaxation!(psiEqn, psi.y.values, solversetup.relax, ydir, config)
    implicit_relaxation_diagdom!(psiEqn, psi.y.values, solversetup.relax, ydir, config)
    # update_preconditioner!(psiEqn.preconditioner, mesh, config)
    resy = solve_system!(psiEqn, solversetup, psi.y, ydir, config)
    
    # Z velocity calculations (3D Mesh only)
    resz = zero(_get_float(mesh))
    if typeof(mesh) <: Mesh3
        update_equation!(psiEqn, config)
        apply_boundary_conditions!(psiEqn, psiBCs, zdir, time, config)
        # implicit_relaxation!(psiEqn, psi.z.values, solversetup.relax, zdir, config)
        implicit_relaxation_diagdom!(psiEqn, psi.z.values, solversetup.relax, zdir, config)
        # update_preconditioner!(psiEqn.preconditioner, mesh, config)
        resz = solve_system!(psiEqn, solversetup, psi.z, zdir, config)
    end
    return resx, resy, resz
end

function solve_system!(phiEqn::ModelEquation, setup, result, component, config)

    # Auto-initialise solver/preconditioner if missing
    if isnothing(phiEqn.solver) || isnothing(phiEqn.preconditioner)
        # We need to update the ModelEquation in-place if possible, or use the setup
        # Since phiEqn is immutable, we use a temporary version or better, 
        # warn the user and use a default.
        # Actually, it's better to force initialization in the examples.
        # But for reliability, let's do it here.
        A = _A(phiEqn)
        b = _b(phiEqn, component)
        P = set_preconditioner(setup.preconditioner, phiEqn)
        update_preconditioner!(P, get_phi(phiEqn).mesh, config)
        S = _workspace(setup.solver, b)
        
        # Call the actual solver with these temporary objects
        krylov_solve!(
            S, A, b, result.values; 
            M=P.P, itmax=setup.itmax, atol=setup.atol, rtol=setup.rtol, ldiv=is_ldiv(P)
        )
        
        Krylov.iteration_count(S) == setup.itmax && @warn "Maximum number of iterations reached!"
        ndrange = length(result.values)
        kernel! = _copy!(_setup(config.hardware.backend, config.hardware.workgroup, ndrange)...)
        kernel!(result.values, S.x)
        KernelAbstractions.synchronize(config.hardware.backend)
        
        return solve_residual(phiEqn, component, config)
    end

    (; itmax, atol, rtol) = setup
    precon = phiEqn.preconditioner
    (; P) = precon 
    solver = phiEqn.solver
    (; x) = solver
    
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware
    (; values, mesh) = result
    
    A = _A(phiEqn)
    opA = parent(A)
    b = _b(phiEqn, component)

    apply_smoother!(setup.smoother, values, opA, b, hardware)

    krylov_solve!(
        solver, opA, b, values; 
        M=P, itmax=itmax, atol=atol, rtol=rtol, ldiv=is_ldiv(precon)
        )

    # Perform explicit step for Crank-Nicholson. Otherwise simply update field with solution
    if hasproperty(phiEqn.model.terms[1], :type) && typeof(phiEqn.model.terms[1].type) <: Time{CrankNicolson}
        xcal_foreach(x, config) do i 
            x[i] = 2*x[i] - values[i]
        end
    end

    ndrange = length(values)
    kernel! = _copy!(_setup(backend, workgroup, ndrange)...)
    kernel!(values, x)
    KernelAbstractions.synchronize(backend)

    Krylov.iteration_count(solver) == itmax && @warn "Maximum number of iterations reached!"

    # println(statistics(solver).niter)
    res = solve_residual(phiEqn, component, config)
    return res
end

@kernel function _copy!(a, b)
    i = @index(Global)

    @inbounds begin
        a[i] = b[i]  
    end
end

function explicit_relaxation!(phi, phi0, alpha, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(phi)
    kernel! = explicit_relaxation_kernel!(_setup(backend, workgroup, ndrange)...)
    kernel!(phi, phi0, alpha)
    # KernelAbstractions.synchronize(backend)
end

@kernel function explicit_relaxation_kernel!(phi, phi0, alpha)
    i = @index(Global)

    @inbounds begin
        phi[i] = phi0[i] + alpha*(phi[i] - phi0[i])
    end
end

## IMPLICIT RELAXATION KERNEL 

# Prepare variables for kernel and call
function implicit_relaxation!(
    phiEqn::E, field, alpha, component, config) where E<:ModelEquation
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Extract sparse matrix properties and values
    A = _A(phiEqn)
    b = _b(phiEqn, component)
    colval = _colval(A)
    rowptr = _rowptr(A)
    nzval = _nzval(A)

    ndrange = length(b)
    kernel! = implicit_relaxation_kernel!(_setup(backend, workgroup, ndrange)...)
    kernel!(colval, rowptr, nzval, b, field, alpha)
    # KernelAbstractions.synchronize(backend)
end

@kernel function implicit_relaxation_kernel!(colval, rowptr, nzval, b, field, alpha)
    i = @index(Global)
    
    @inbounds begin
        nIndex = spindex(rowptr, colval, i, i)
        nzval[nIndex] /= alpha
        b[i] += (1.0 - alpha)*nzval[nIndex]*field[i]
    end
end


## IMPLICIT RELAXATION KERNEL with DIAGONAL DOMINANCE

# Prepare variables for kernel and call
function implicit_relaxation_diagdom!(
    phiEqn::E, field, alpha, component, config) where E<:ModelEquation
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Extract sparse matrix properties and values
    A = _A(phiEqn)
    b = _b(phiEqn, component)
    colval = _colval(A)
    rowptr = _rowptr(A)
    nzval = _nzval(A)

    ndrange = length(b)
    kernel! = _implicit_relaxation_diagdom!(_setup(backend, workgroup, ndrange)...)
    kernel!(colval, rowptr, nzval, b, field, alpha)
    # KernelAbstractions.synchronize(backend)
end

@kernel function _implicit_relaxation_diagdom!(colval, rowptr, nzval, b, field, alpha)
    i = @index(Global)
    
    sumv = zero(eltype(b))

    @inbounds begin

        # Find nzval index relating to A[i,i]
        cIndex = spindex(rowptr, colval, i, i)

        start_index = rowptr[i]
        end_index = rowptr[i+1] -1
        for nzi ∈ start_index:end_index
            sumv += abs(nzval[nzi])
        end
        sumv -= abs(nzval[cIndex]) # remove diagonal contribution

        # Run implicit relaxation calculations
        D0 = nzval[cIndex]
        D_max = max(abs(D0), sumv)/alpha
        nzval[cIndex] = D_max
        b[i] += (D_max - D0)*field[i]
    end
end


function setReference!(pEqn::E, pRef, cellID, config) where E<:ModelEquation
    if pRef === nothing
        return nothing
    else
        (; hardware) = config
        (; backend, workgroup) = hardware
        (; b, A) = pEqn.equation
        nzval = _nzval(A)
        colval = _colval(A)
        rowptr = _rowptr(A)

        ndrange = 1
        kernel! = _setReference!(_setup(backend, workgroup, ndrange)...)
        kernel!(nzval, colval, rowptr, b, pRef, cellID)
    end
end

@kernel function _setReference!(nzval, colval, rowptr, b, pRef, cellID)
    i = @index(Global)

    @inbounds begin
        cIndex = spindex(rowptr, colval, cellID, cellID)
        b[cellID] = nzval[cIndex]*pRef
        nzval[cIndex] += nzval[cIndex]
    end
end

function _residual_equation(eqn; susp=false, ad_backend=:forwarddiff)
    has_nonlinear = any(eqn.model.terms) do t
        t isa NonlinearOperator ||
        (hasproperty(t, :type) && t.type isa NonLinearSi)
    end
    has_nonlinear || return eqn
    _, lin_eqn = linearize_physics(get_bcs(eqn), eqn; susp=susp, ad_backend=ad_backend)
    return lin_eqn
end

# residual!(r, eqn, config)  — FULL residual (interior + BC faces).
#
# Two computation paths — both discrete, both correct for linear problems:
#
#   explicit=false (default): assembled path — r = A·φ − b.
#     BCs are already in A and b from the assembly step (fvm:: style).
#     Pass assemble=false to skip re-assembly when A is already up to date.
#
#   explicit=true: explicit operator path — evaluates fluxes directly on φ
#     without going through the assembled matrix (fvc:: style in OpenFOAM).
#     Uses explicit_residual! (interior faces) + apply_bc_residuals! (BC faces).
#     Required for JFNK/Newton inner loops where φ changes between evaluations.
#
# See also: explicit_residual! for interior-only explicit evaluation (no BC).
function residual!(
    r, eqn::ModelEquation{T,M,E,S,P}, config; component=nothing, time=nothing,
    assemble=true, explicit=false, susp=false, ad_backend=:forwarddiff
    ) where {T<:ScalarModel,M,E,S,P}
    eqn = _residual_equation(eqn; susp=susp, ad_backend=ad_backend)
    phi = get_phi(eqn)
    if explicit
        fill!(r, zero(eltype(r)))
        explicit_residual!(r, eqn, phi, config)
        apply_bc_residuals!(r, eqn, config; component=component, time=time)
    else
        if assemble
            discretise!(eqn, phi, config)
            apply_boundary_conditions!(eqn, config; time=time, component=component)
        end
        A = _A(eqn)
        b = _b(eqn, component)
        values = get_values(phi, component)
        r .= A * values
        r .-= b
    end
    return r
end

function jvp!(Jv::AbstractVector, v::AbstractVector, eqn::ModelEquation{T,M,E,S,P}, config; component=nothing, time=nothing, ε=nothing) where {T<:ScalarModel,M,E,S,P}
    phi_vals = get_values(get_phi(eqn), component)
    F  = eltype(phi_vals)
    ε0 = ε === nothing ? sqrt(eps(F)) : F(ε)
    r0 = similar(phi_vals)
    r1 = similar(phi_vals)
    residual!(r0, eqn, config; component=component, time=time, explicit=true)
    @. phi_vals += ε0 * v
    residual!(r1, eqn, config; component=component, time=time, explicit=true)
    @. phi_vals -= ε0 * v   # restore
    @. Jv = (r1 - r0) / ε0
    return Jv
end

function residual!(r, eqn::ModelEquation{T,M,E,S,P}, config; kwargs...) where {T<:VectorModel,M,E,S,P}
    error("Mathematical residual vectors for VectorModel equations require an explicit component implementation. Use scalar components or solve_residual for solve-monitor norms for now.")
end

function residual(
    eqn::ModelEquation{T,M,E,S,P}, config; component=nothing, time=nothing,
    assemble=true, susp=false, ad_backend=:forwarddiff
) where {T<:ScalarModel,M,E,S,P}
    r = similar(_b(eqn, component))
    residual!(
        r, eqn, config;
        component=component, time=time, assemble=assemble, susp=susp, ad_backend=ad_backend,
    )
end

function residual(L::PDEOperator, phi::ScalarField, config; kwargs...)
    residual(L(phi), config; kwargs...)
end

residual_norm(r::AbstractArray) = norm(r)
residual_norm(eqn::ModelEquation, config; kwargs...) = residual_norm(residual(eqn, config; kwargs...))
residual_norm(L::PDEOperator, phi::ScalarField, config; kwargs...) =
    residual_norm(residual(L, phi, config; kwargs...))

function solve_residual(eqn, component, config)
    (; A, R, Fx) = eqn.equation
    b = _b(eqn, component)
    values = get_values(get_phi(eqn), component)

    # # Openfoam's residual definition (not optimised)
    # Fx .= A*values
    # R .= mean(values)
    # Fx_mean = A*R 
    # T1 = mean(norm.(b .- Fx))
    # T2 = mean(norm.(Fx .- Fx_mean))
    # T3 = mean(norm.(b .- Fx_mean))
    # Residual = T1/(T2 + T3)

    # Previous definition
    Fx .= A * values
    xcal_foreach(R, config) do i 
            @inbounds R[i] = (b[i] - Fx[i])^2
    end
    normb = norm(b)
    denominator = ifelse(normb > eps(normb), normb, one(normb))
    Residual = sqrt(sum(R)) / denominator
    return Residual
end

function make_symmetric!(eqn, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    (; b, A) = eqn.equation
    mesh = get_phi(eqn).mesh
    (; faces) = mesh
    nzval = _nzval(A)
    colval = _colval(A)
    rowptr = _rowptr(A)

    nbfaces = mesh.boundary_cellsID |> length
    ndrange = length(faces) - nbfaces
    kernel! = _make_symmetric!(_setup(backend, workgroup, ndrange)...)
    kernel!(colval, rowptr, nzval, faces, nbfaces)
end

@kernel function _make_symmetric!(colval, rowptr, nzval, faces, nbfaces)
    i = @index(Global)
    fID = i + nbfaces

    face = faces[fID]
    (; ownerCells) = face 
    cID1 = ownerCells[1]
    cID2 = ownerCells[2]

    cIndex1 = spindex(rowptr, colval, cID1, cID2)
    cIndex2 = spindex(rowptr, colval, cID2, cID1)

    Apn = nzval[cIndex1]
    nzval[cIndex2] = Apn

end
