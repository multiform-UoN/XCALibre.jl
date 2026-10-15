export linearize_physics

# NEWTON LINEARISATION OF NONLINEAR OPERATORS
# f(φ) ≈ f(φ₀) + f'(φ₀)(φ - φ₀) cell by cell, in one kernel pass (CPU threads or GPU). Ghost cells of
# a distributed mesh are evaluated from the ghost values of φ, so no extra halo exchange is needed.

@kernel function _linearise_operator!(jacobian, offset, reference, phi, map)
    i = @index(Global)
    @inbounds begin
        v0 = phi[i]
        r0 = map.func(v0)
        dv = map.derivative(v0)
        reference[i] = r0
        jacobian[i] = dv
        offset[i] = r0 - dv * v0
    end
end

_linearise_term(term, config) = term
function _linearise_term(term::NonlinearOperator, config)
    (; backend, workgroup) = config.hardware
    phi = term.op.phi
    jacobian, offset, reference = ScalarField(phi.mesh), ScalarField(phi.mesh), ScalarField(phi.mesh)
    kernel! = _sized(_linearise_operator!, backend, workgroup, length(phi.values))
    kernel!(jacobian.values, offset.values, reference.values, phi.values, term.map)
    KernelAbstractions.synchronize(backend)
    AffineOperator(term.op, jacobian, offset, reference, term.map)
end

"""
    linearize_physics(BCs, eqn, config)

Newton linearisation of `eqn` about the current values of its field: every `NonlinearOperator`
becomes an `AffineOperator`, other terms are unchanged. Returns `(BCs, linearised_eqn)`; boundary
values of an affine term are mapped by its `NonlinearMap`, so `BCs` are given for φ.
"""
function linearize_physics(BCs, eqn::ModelEquation, config)
    terms = map(t -> _linearise_term(t, config), eqn.model.terms)
    sources = eqn.model.sources
    model = Model{length(terms),length(sources)}(terms, sources)
    return BCs, (@set eqn.model = model)
end
