using ForwardDiff
using Enzyme
using Accessors

export linearize_physics, linearize_bcs, homogeneous, newton_solve!

# DSL constructor overloads: three-argument form returns NonlinearOperator wrapper
# (extending constructors defined in ModelFramework)
ModelFramework.Laplacian{T}(flux, func::Function, phi) where T =
    NonlinearOperator(Operator(flux, phi, 1, Laplacian{T}()), NonlinearMap(func))

ModelFramework.Divergence{T}(flux, func::Function, phi) where T =
    NonlinearOperator(Operator(flux, phi, 1, Divergence{T}()), NonlinearMap(func))

ModelFramework.Laplacian{T}(flux, func::Function) where T =
    NonlinearOperatorTemplate(OperatorTemplate(flux, 1, Laplacian{T}()), NonlinearMap(func))

ModelFramework.Divergence{T}(flux, func::Function) where T =
    NonlinearOperatorTemplate(OperatorTemplate(flux, 1, Divergence{T}()), NonlinearMap(func))

ModelFramework.Laplacian{T}(flux, map::NonlinearMap, phi) where T =
    NonlinearOperator(Operator(flux, phi, 1, Laplacian{T}()), map)

ModelFramework.Divergence{T}(flux, map::NonlinearMap, phi) where T =
    NonlinearOperator(Operator(flux, phi, 1, Divergence{T}()), map)

ModelFramework.Laplacian{T}(flux, map::NonlinearMap) where T =
    NonlinearOperatorTemplate(OperatorTemplate(flux, 1, Laplacian{T}()), map)

ModelFramework.Divergence{T}(flux, map::NonlinearMap) where T =
    NonlinearOperatorTemplate(OperatorTemplate(flux, 1, Divergence{T}()), map)

# ── New PDEOperator DSL: NonLinearSi(func) → OperatorTemplate ─────────────────
# Field is deferred — bound later when L(phi) is called via OperatorTemplate.__call__.
# Enables:
#   L = (... + NonLinearSi(f) == ...) → BCs → solver
#   newton_solve!(L, C, config)
#
# Implementation note: the type tag NonLinearSi{T} is created with the inner
# parametric constructor NonLinearSi{T}(map) to avoid dispatching back through
# these outer constructor methods.
ModelFramework.NonLinearSi(func::Function) = begin
    map = NonlinearMap(func)
    OperatorTemplate(nothing, 1, NonLinearSi{typeof(map)}(map))
end

ModelFramework.NonLinearSi(func::Function, deriv::Function) = begin
    map = NonlinearMap(func, deriv)
    OperatorTemplate(nothing, 1, NonLinearSi{typeof(map)}(map))
end

# ── Old DSL (backward compatible): NonLinearSi(func, phi) → bound Operator ────
ModelFramework.NonLinearSi(func::Function, phi) = begin
    map = NonlinearMap(func)
    Operator(nothing, phi, 1, NonLinearSi{typeof(map)}(map))
end

ModelFramework.NonLinearSi(map::NonlinearMap, phi) =
    Operator(nothing, phi, 1, NonLinearSi{typeof(map)}(map))

@inline _zero_bc_value(value::Number) = zero(value)
@inline _zero_bc_value(value::StaticArray) = zero(value)
@inline _zero_bc_value(value::AbstractVector) = zero.(value)
@inline _zero_bc_value(value) =
    error("Cannot construct a homogeneous value for boundary value type $(typeof(value)).")

@inline homogeneous(bc::Dirichlet) = Dirichlet(bc.ID, _zero_bc_value(bc.value), bc.IDs_range)
@inline homogeneous(bc::Wall{I,V,R}) where {I,V<:Number,R} =
    Wall(bc.ID, _zero_bc_value(bc.value), bc.IDs_range)
@inline homogeneous(bc::Wall{I,V,R}) where {I,V<:StaticArray,R} =
    Wall(bc.ID, _zero_bc_value(bc.value), bc.IDs_range)
@inline homogeneous(bc::Robin) = Robin(
    bc.ID,
    RobinValue(a=bc.value.a, b=bc.value.b, value=_zero_bc_value(bc.value.value)),
    bc.IDs_range,
)
@inline homogeneous(bc::Zerogradient) = bc
@inline homogeneous(bc::Extrapolated) = bc
@inline homogeneous(bc::Symmetry) = bc
@inline homogeneous(bc::Outlet) = bc
# Periodic BCs impose a connectivity constraint, not a prescribed value.
# The Newton correction δφ must satisfy the same periodicity as φ itself,
# so the homogeneous periodic BC is identical to the original.
@inline homogeneous(bc::Periodic) = bc
@inline homogeneous(bc::PeriodicParent) = bc
@inline homogeneous(bc::Tuple{}) = ()
@inline homogeneous(BCs::Tuple) = map(homogeneous, BCs)
@inline homogeneous(BCs::AbstractVector{<:AbstractBoundary}) = map(homogeneous, BCs)
@inline homogeneous(L::PDEOperator) = PDEOperator(L.templates, (), homogeneous(L.BCs), L.setup)
@inline homogeneous(bc::AbstractBoundary) =
    error("Boundary $(typeof(bc)) does not have a homogeneous transformation yet.")

# ---------------------------------------------------------------------------
# linearize_bcs: replace NonLinearRobin BCs with their Newton-linearised Robin
# ---------------------------------------------------------------------------
function linearize_bcs(BCs, phi::ScalarField)
    return Discretise.update_nonlinear_robin(BCs, phi)
end

function linearize_bcs(BCs::NamedTuple, phi::ScalarField)
    names = propertynames(BCs)
    updated = map(values(BCs)) do field_bcs
        field_bcs isa Tuple ? Discretise.update_nonlinear_robin(field_bcs, phi) : field_bcs
    end
    return NamedTuple{names}(updated)
end

@inline _map_value(map::NonlinearMap, value) = map.func(value)

function _map_derivative(map::NonlinearMap, value, ad_backend)
    if map.derivative !== nothing
        return map.derivative(value)
    elseif ad_backend === :forwarddiff
        return ForwardDiff.derivative(map.func, value)
    elseif ad_backend === :enzyme
        return Enzyme.autodiff(Enzyme.Reverse, map.func, Enzyme.Active, Enzyme.Active(value))[1][1]
    else
        error("Unsupported AD backend: $ad_backend. Use :forwarddiff, :enzyme, or provide NonlinearMap(f, df).")
    end
end

function _assert_cpu_linearization(mesh)
    backend = _get_backend(mesh)
    backend isa CPU && return nothing
    error("linearize_physics currently supports CPU-resident nonlinear prepasses only. Provide a CPU field/template or implement a device AD path for backend $(typeof(backend)).")
end

# ---------------------------------------------------------------------------
# linearize_physics: convert all NonlinearOperator terms into standard Operators
# using automatic differentiation (ForwardDiff default, Enzyme optional)
# ---------------------------------------------------------------------------
"""
    linearize_physics(BCs, model_eqn; susp=false, ad_backend=:forwarddiff)

Pre-pass Newton linearisation. Detects `NonlinearOperator` and `NonLinearSi` terms,
differentiates the stored functions at current field values, and returns a fresh
linearised `ModelEquation`. The input equation is kept as the nonlinear template.
"""
function linearize_physics(BCs, model_eqn::ModelEquation; susp=false, ad_backend=:forwarddiff)
    phi = get_phi(model_eqn)
    # 1. Linearise BCs (NonLinearRobin → Robin)
    new_bcs = linearize_bcs(BCs, phi)

    # 2. Linearise model terms
    extra_sources = []

    new_terms = map(model_eqn.model.terms) do term
        # Case A: Non-linear Differential Operators (Wrapped in NonlinearOperator)
        # Must check this BEFORE NonLinearSi since NonlinearOperator has no .type field
        if term isa NonlinearOperator
            map = term.map
            inner_op = term.op
            term_phi = inner_op.phi
            vals = term_phi.values
            mesh = term_phi.mesh
            _assert_cpu_linearization(mesh)

            jacobian = ScalarField(mesh)
            offset = ScalarField(mesh)
            reference = ScalarField(mesh)
            for i in eachindex(vals)
                v0 = vals[i]
                r0 = _map_value(map, v0)
                dv = _map_derivative(map, v0, ad_backend)
                reference.values[i] = r0
                jacobian.values[i] = dv
                offset.values[i] = r0 - dv * v0
            end

            return AffineOperator(inner_op, jacobian, offset, reference, map)

        # Case B: Standard Non-linear Implicit Source (NonLinearSi)
        elseif hasproperty(term, :type) && typeof(term.type) <: NonLinearSi
            map = term.type.func
            term_phi = term.phi
            vals = term_phi.values
            mesh = term_phi.mesh
            _assert_cpu_linearization(mesh)
            k_imp = ScalarField(mesh)
            s_exp = ScalarField(mesh)

            for i in eachindex(vals)
                v0 = vals[i]
                r0 = _map_value(map, v0)
                dv = _map_derivative(map, v0, ad_backend)
                term_sign = term.sign

                # Equation: ... + R(phi) = ...
                # Newton: R(phi) ≈ dv * phi + (r0 - dv * v0)
                # SuSp logic is opt-in; otherwise keep the exact local Newton term.
                if !susp || term_sign * dv > 0
                    k_imp.values[i] = dv
                    s_exp.values[i] = -term_sign * (r0 - dv * v0)
                else
                    k_imp.values[i] = 0.0
                    s_exp.values[i] = -term_sign * r0
                end
            end
            push!(extra_sources, Source(s_exp))
            return Operator(k_imp, term_phi, term.sign, Si())

        else
            return term
        end
    end

    # 3. Rebuild Model and Equation
    new_terms_tuple = Tuple(new_terms)
    new_sources = (model_eqn.model.sources..., extra_sources...)
    new_model = Model{length(new_terms_tuple), length(new_sources)}(new_terms_tuple, new_sources)

    return new_bcs, @set model_eqn.model = new_model
end

function _newton_tolerance(model_eqn, tol)
    tol !== nothing && return tol
    setup = model_eqn.setup
    setup === nothing && return sqrt(eps(_get_float(get_phi(model_eqn).mesh)))
    return setup.convergence
end

function _with_bcs(model_eqn::ModelEquation, BCs)
    @set model_eqn.equation.BCs = BCs
end

function newton_solve!(
    model_eqn::ModelEquation{T,M,E,S,P}, config;
    tol=nothing, maxiter=config.runtime.iterations, damping=1.0,
    susp=false, ad_backend=:forwarddiff, verbose=false,
    exact_residual=true
) where {T<:ScalarModel,M,E,S,P}
    phi = get_phi(model_eqn)
    tolerance = _newton_tolerance(model_eqn, tol)
    TF = _get_float(phi.mesh)
    history = TF[]
    converged = false

    for iter in 1:maxiter
        updated_bcs, linear_eqn = linearize_physics(get_bcs(model_eqn), model_eqn; susp=susp, ad_backend=ad_backend)
        linear_eqn = _with_bcs(linear_eqn, updated_bcs)

        # exact_residual=true  → r = A_lin*u - b_lin = F(u_k) exactly (Newton linearisation
        #                          ensures A_lin*u - b_lin = F_nonlinear(u_k) at current u_k)
        # exact_residual=false → cheap Krylov monitor norm ||b - A*x||/||b|| from last solve
        if exact_residual
            r = residual(linear_eqn, config)
            rnorm = residual_norm(r)
        else
            rnorm = solve_residual(linear_eqn, 1, config)
        end
        push!(history, TF(rnorm))
        verbose && @info "Newton iteration $iter" residual=rnorm
        if rnorm <= tolerance
            converged = true
            break
        end

        previous = copy(phi.values)
        solve_equation!(linear_eqn, config)
        if damping != 1
            @. phi.values = previous + damping * (phi.values - previous)
        end
    end

    return (converged=converged, iterations=length(history), residuals=history)
end

function newton_solve!(L::PDEOperator, phi::ScalarField, config; kwargs...)
    newton_solve!(L(phi), config; kwargs...)
end
