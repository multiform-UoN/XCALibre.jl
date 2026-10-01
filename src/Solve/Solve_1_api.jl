export SolverSetup, Runtime, Schemes
export explicit_relaxation!, implicit_relaxation!, implicit_relaxation_diagdom!, setReference!
export solve_system!
export sync!
export wrap_eqn
export unwrap_eqn
export is_distributed_mesh
export is_report_rank
export solve_equation!, solve_preassembled!
export residual, residual!, residual_norm, jvp!
export AdaptiveTimeStepping

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
- `convergence`: residual target for this field that stops the outer (e.g. SIMPLE) iteration; it does not control the linear solver, which stops on `atol`, `rtol` and `itmax`.
- `relax`: specifies the relaxation factor to be used e.g. set to 1 for no relaxation
- `smoother`: specifies smoothing method to be applied before discretisation. `JacobiSmoother`: is currently the only choice (defaults to `nothing`)
- `limit`: used in some solvers to bound the solution within these limits e.g. (min, max). It defaults to `nothing`
- `itmax`: maximum number of iterations in a single solver pass (defaults to 1000, or 200 for `AMG`). Also applies to PETSc solves.
- `atol`: absolute tolerance for the solver (default to eps(FloatType)^0.9). Also applies to PETSc solves.
- `rtol`: set relative tolerance for the solver (defaults to 1e-1). Also applies to PETSc solves.
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
        itmax::I=(solver isa AMG ? 200 : 1000),
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
    maxAlphaCo::F
    minShrink::F
    maxGrow::F
end
Adapt.@adapt_structure AdaptiveTimeStepping

"""
    AdaptiveTimeStepping(; 
        # keyword arguments

        maxCo=0.75,
        maxAlphaCo=0.5,
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
- `maxAlphaCo::AbstractFloat`: target maximum Alpha Courant number. The time step will be adjusted
  such that the computed Courant number approaches this value.
- `minShrink::AbstractFloat`: lower bound on the multiplicative factor applied to the
  current time step. Prevents excessively large reductions in a single update.
- `maxGrow::AbstractFloat`: upper bound on the multiplicative factor applied to the
  current time step. Prevents excessive time-step growth.
"""
AdaptiveTimeStepping(;
    maxCo=0.75,
    maxAlphaCo=0.5,
    minShrink=0.1,
    maxGrow=1.2
) = AdaptiveTimeStepping(float(maxCo), float(maxAlphaCo), float(minShrink), float(maxGrow))

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


# eqn.model.terms[1].flux implies that the time term must always be defined first when constructing an equation.
function solve_equation!(
    eqn::ModelEquation{T,M,E,S,P}, phi, phiBCs, solversetup, config; rho_prev=_get_flux(eqn.model.terms[1]), time=nothing, ref=nothing, irelax=nothing
    ) where {T<:ScalarModel,M,E,S,P}

    discretise!(eqn, phi, config, rho_prev=rho_prev)  
    apply_boundary_conditions!(eqn, phiBCs, nothing, time, config)
    if length(eqn.model.terms) == 1 && typeof(eqn.model.terms[1]) <: Laplacian
        make_symmetric!(eqn, config) # added this to test stability of periodic boundaries
    end
    setReference!(eqn, ref, 1, config)
    if !isnothing(irelax)
        implicit_relaxation!(eqn, phi.values, irelax, nothing, config)
        # implicit_relaxation_diagdom!(eqn, phi.values, irelax, nothing, config)
    end
    update_preconditioner!(eqn.preconditioner, phi.mesh, config)
    res = solve_system!(eqn, solversetup, phi, nothing, config)
    return res
end

# psiEqn.model.terms[1].flux implies that the time term must always be defined first when constructing an equation.
function solve_equation!(
    psiEqn::ModelEquation{T,M,E,S,P}, psi, psiBCs, solversetup, xdir, ydir, zdir, config; rho_prev=_get_flux(psiEqn.model.terms[1]), time=nothing
    ) where {T<:VectorModel,M,E,S,P}

    mesh = psi.mesh

    discretise!(psiEqn, psi, config, rho_prev=rho_prev)
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

    (; itmax, atol, rtol) = setup
    precon = phiEqn.preconditioner
    (; P) = precon 
    solver = phiEqn.solver
    (; x) = solver
    
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware
    (; values, mesh) = result
    
    A = _A(phiEqn)
    opA = A
    b = _b(phiEqn, component)

    apply_smoother!(setup.smoother, values, A, b, hardware)

    krylov_solve!(
        solver, opA, _like_workspace(x, b), _like_workspace(x, values); 
        M=P, itmax=itmax, atol=atol, rtol=rtol, ldiv=is_ldiv(precon), history=false
        )

    # Perform explicit step for Crank-Nicholson. Otherwise simply update field with solution
    if typeof(phiEqn.model.terms[1].type) <: Time{CrankNicolson}
        xcal_foreach(x, config) do i 
            x[i] = 2*x[i] - values[i]
        end
    end

    ndrange = length(values)
    kernel! = _sized(_copy!, backend, workgroup, ndrange)
    kernel!(values, x)

    iterations = Krylov.iteration_count(solver)
    iterations == itmax && @warn "Maximum number of iterations reached!"

    res = residual(phiEqn, component, config)
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
    kernel! = _sized(explicit_relaxation_kernel!, backend, workgroup, ndrange)
    kernel!(phi, phi0, alpha)
    # KernelAbstractions.synchronize(backend)
    sync!(phi, phi.mesh, config) # self-syncing seam (no-op serial)
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
    diag_nz = phiEqn.equation.diag_nz

    ndrange = length(b)
    kernel! = _sized(implicit_relaxation_kernel!, backend, workgroup, ndrange)
    kernel!(colval, rowptr, nzval, diag_nz, b, field, alpha)
    # KernelAbstractions.synchronize(backend)
end

@kernel function implicit_relaxation_kernel!(colval, rowptr, nzval, diag_nz, b, field, alpha)
    i = @index(Global)
    
    @inbounds begin
        nIndex = diag_nz[i]
        nzval[nIndex] /= alpha
        b[i] += (one(alpha) - alpha)*nzval[nIndex]*field[i]
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
    diag_nz = phiEqn.equation.diag_nz

    ndrange = length(b)
    kernel! = _sized(_implicit_relaxation_diagdom!, backend, workgroup, ndrange)
    kernel!(colval, rowptr, nzval, diag_nz, b, field, alpha)
    # KernelAbstractions.synchronize(backend)
end

@kernel function _implicit_relaxation_diagdom!(colval, rowptr, nzval, diag_nz, b, field, alpha)
    i = @index(Global)
    
    sumv = zero(eltype(b))

    @inbounds begin

        cIndex = diag_nz[i]

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
        kernel! = _sized(_setReference!, backend, workgroup, ndrange)
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

function residual(eqn, component, config)
    (; A, R, Fx) = eqn.equation
    b = _b(eqn, component)
    values = get_values(get_phi(eqn), component)
    (; backend, workgroup) = config.hardware

    rowptr = _rowptr(A)
    colval = _colval(A)
    nzval = _nzval(A)
    ndrange = length(values)
    kernel! = _sized(_scaled_residual!, backend, workgroup, ndrange)
    kernel!(R, Fx, rowptr, colval, nzval, values, b)

    denominator = sum(Fx)
    denominator = ifelse(denominator > eps(denominator), denominator, one(denominator))
    Residual = sum(R) / denominator

    # Alternative: OpenFOAM normalised residual T1/(T2 + T3) (not optimised)
    return Residual
end

@kernel function _scaled_residual!(R, Fx, @Const(rowptr), @Const(colval), @Const(nzval), @Const(values), @Const(b))
    i = @index(Global)
    Ax = zero(eltype(R))
    Dx = zero(eltype(R))
    xi = values[i]

    @inbounds for nzi ∈ rowptr[i]:(rowptr[i + 1] - 1)
        Aij = nzval[nzi]
        j = colval[nzi]
        Ax += Aij * values[j]
        if j == i
            Dx = Aij * xi
        end
    end

    @inbounds begin
        R[i] = abs(b[i] - Ax)
        Fx[i] = abs(Dx)
    end
end

# halo-exchange seam: DistributedMesh method lives in Distribute; serial is a free no-op
@inline sync!(x, mesh::Union{Mesh2,Mesh3}, config) = nothing

# linear-solve seam: setup wraps each eqn so the body calls generic solve_equation!/
# solve_system!. Serial = identity; Distribute overrides for DistributedMesh (DistributedEqn +
# PETScSolver). Extra kwargs (petsc_options) are ignored serially.
wrap_eqn(eqn, mesh, setup, config; kwargs...) = eqn

# raw ModelEquation behind a (possibly wrapped) eqn: solver bodies assemble/discretise on the
# raw eqn but solve through the wrapper. Serial identity; Distribute unwraps DistributedEqn.
@inline unwrap_eqn(eqn) = eqn


# mesh-kind predicate: Distribute overrides for DistributedMesh. mesh is concrete in bodies so
# calls constant-fold — used to skip Krylov precond/workspace setup and rank-0-only reporting.
@inline is_distributed_mesh(mesh) = false

# true where solver progress/@info should print: always serial, only rank 0 when distributed
@inline is_report_rank(mesh) = true

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
    kernel! = _sized(_make_symmetric!, backend, workgroup, ndrange)
    kernel!(colval, rowptr, nzval, faces, nbfaces)
end

@kernel function _make_symmetric!(colval, rowptr, nzval, faces, nbfaces)
    i = @index(Global)
    fID = i + nbfaces

    face = faces[fID]
    (; ownerCells) = face
    # canonical row = min owner: on partitioned meshes owner1 may be a ghost whose CSR
    # row is garbage; coeff is symmetric so serial value is unchanged
    cID1 = min(ownerCells[1], ownerCells[2])
    cID2 = max(ownerCells[1], ownerCells[2])

    cIndex1 = spindex(rowptr, colval, cID1, cID2)
    cIndex2 = spindex(rowptr, colval, cID2, cID1)

    Apn = nzval[cIndex1]
    nzval[cIndex2] = Apn

end

# PDE scripts store the solver on the operator and call solve_equation!(eqn, config).
# The field, boundary and setup arguments used by SIMPLE/PISO stay on the methods above.
import XCALibre.ModelFramework: →
(→)(L::PDEOperator, setup::SolverSetup) =
    PDEOperator(L.templates, L.sources, L.BCs, setup)

function solve_equation!(
    eqn::ModelEquation{T,M,E,S,P}, config;
    rho_prev=_get_flux(eqn.model.terms[1]), time=nothing, ref=nothing, irelax=nothing
    ) where {T<:ScalarModel,M,E,S,P}
    phi = get_phi(eqn)
    setup = eqn.setup
    discretise!(eqn, phi, config; rho_prev=rho_prev)
    apply_boundary_conditions!(eqn, config; time=time)
    if length(eqn.model.terms) == 1 && typeof(eqn.model.terms[1]) <: Laplacian
        make_symmetric!(eqn, config)
    end
    setReference!(eqn, ref, 1, config)
    if !isnothing(irelax)
        implicit_relaxation!(eqn, phi.values, irelax, nothing, config)
    end
    if !isnothing(eqn.preconditioner)
        update_preconditioner!(eqn.preconditioner, phi.mesh, config)
    end
    return solve_system!(eqn, setup, phi, nothing, config)
end

function solve_preassembled!(
    eqn::ModelEquation{T,M,E,S,P}, config; time=nothing
    ) where {T<:ScalarModel,M,E,S,P}
    phi = get_phi(eqn)
    apply_boundary_conditions!(eqn, config; time=time)
    if !isnothing(eqn.preconditioner)
        update_preconditioner!(eqn.preconditioner, phi.mesh, config)
    end
    return solve_system!(eqn, eqn.setup, phi, nothing, config)
end

function solve_equation!(
    psiEqn::ModelEquation{T,M,E,S,P}, config;
    rho_prev=_get_flux(psiEqn.model.terms[1]), time=nothing
    ) where {T<:VectorModel,M,E,S,P}
    psi = get_phi(psiEqn)
    mesh = psi.mesh
    solversetup = psiEqn.setup
    discretise!(psiEqn, psi, config; rho_prev=rho_prev)
    update_equation!(psiEqn, config)
    apply_boundary_conditions!(psiEqn, config; time=time, component=XDir())
    implicit_relaxation_diagdom!(psiEqn, psi.x.values, solversetup.relax, XDir(), config)
    if !isnothing(psiEqn.preconditioner)
        update_preconditioner!(psiEqn.preconditioner, mesh, config)
    end
    resx = solve_system!(psiEqn, solversetup, psi.x, XDir(), config)
    update_equation!(psiEqn, config)
    apply_boundary_conditions!(psiEqn, config; time=time, component=YDir())
    implicit_relaxation_diagdom!(psiEqn, psi.y.values, solversetup.relax, YDir(), config)
    resy = solve_system!(psiEqn, solversetup, psi.y, YDir(), config)
    resz = zero(_get_float(mesh))
    if typeof(mesh) <: Mesh3
        update_equation!(psiEqn, config)
        apply_boundary_conditions!(psiEqn, config; time=time, component=ZDir())
        implicit_relaxation_diagdom!(psiEqn, psi.z.values, solversetup.relax, ZDir(), config)
        resz = solve_system!(psiEqn, solversetup, psi.z, ZDir(), config)
    end
    return resx, resy, resz
end

function _residual_equation(eqn; susp=false, ad_backend=:forwarddiff)
    has_nonlinear = any(eqn.model.terms) do t
        t isa NonlinearOperator || (hasproperty(t, :type) && t.type isa NonLinearSi)
    end
    has_nonlinear || return eqn
    new_bcs, lin_eqn, _ = linearize_physics(get_bcs(eqn), eqn; susp=susp, ad_backend=ad_backend)
    return _with_bcs(lin_eqn, new_bcs)
end

function solve_residual(eqn, component, config)
    scalar = component isa Integer || component === nothing
    b = scalar ? _b(eqn, nothing) : _b(eqn, component)
    values = scalar ? get_values(get_phi(eqn), nothing) : get_values(get_phi(eqn), component)
    Fx = similar(b)
    mul!(Fx, _A(eqn), values)
    normb = norm(b)
    denominator = ifelse(normb > eps(typeof(normb)), normb, one(normb))
    return sqrt(sum(abs2, b .- Fx)) / denominator
end

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
        values = get_values(phi, component)
        mul!(r, _A(eqn), values)
        r .-= _b(eqn, component)
    end
    return r
end

function residual!(r, eqn::ModelEquation{T,M,E,S,P}, config; kwargs...) where {T<:VectorModel,M,E,S,P}
    error("Vector residual! needs a component. Decompose the equation or use residual(eqn, component, config) for the solver norm.")
end

function jvp!(
    Jv::AbstractVector, v::AbstractVector, eqn::ModelEquation{T,M,E,S,P}, config;
    component=nothing, time=nothing, ε=nothing
    ) where {T<:ScalarModel,M,E,S,P}
    phi_vals = get_values(get_phi(eqn), component)
    F = eltype(phi_vals)
    vnorm = norm(v)
    if iszero(vnorm)
        fill!(Jv, zero(eltype(Jv)))
        return Jv
    end
    ε0 = ε === nothing ? sqrt(eps(F)) * (one(F) + norm(phi_vals)) / vnorm : F(ε)
    r0 = similar(phi_vals)
    r1 = similar(phi_vals)
    residual!(r0, eqn, config; component=component, time=time, explicit=true)
    @. phi_vals += ε0 * v
    try
        residual!(r1, eqn, config; component=component, time=time, explicit=true)
    finally
        @. phi_vals -= ε0 * v
    end
    @. Jv = (r1 - r0) / ε0
    return Jv
end

function residual(
    eqn::ModelEquation{T,M,E,S,P}, config; component=nothing, time=nothing,
    assemble=true, susp=false, ad_backend=:forwarddiff
) where {T<:ScalarModel,M,E,S,P}
    r = similar(_b(eqn, component))
    residual!(r, eqn, config; component=component, time=time, assemble=assemble,
        susp=susp, ad_backend=ad_backend)
end

residual(L::PDEOperator, phi::ScalarField, config; kwargs...) = residual(L(phi), config; kwargs...)

residual_norm(r::AbstractArray) = norm(r)
residual_norm(eqn::ModelEquation, config; kwargs...) = residual_norm(residual(eqn, config; kwargs...))
residual_norm(L::PDEOperator, phi::ScalarField, config; kwargs...) =
    residual_norm(residual(L, phi, config; kwargs...))
