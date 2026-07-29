export decompose

"""
    decompose(phi::ScalarField) -> [phi]
    decompose(psi::VectorField) -> [psi.x, psi.y, psi.z] (or just x, y for 2D)

Decomposes a field into a flat vector of its scalar components.
"""
function decompose(phi::ScalarField)
    return [phi]
end

function decompose(psi::VectorField)
    if typeof(psi.mesh) <: Mesh2 || typeof(psi.mesh) <: Mesh2D
        return [ScalarField(psi.x.values, psi.mesh), ScalarField(psi.y.values, psi.mesh)]
    else
        return [ScalarField(psi.x.values, psi.mesh), ScalarField(psi.y.values, psi.mesh), ScalarField(psi.z.values, psi.mesh)]
    end
end

"""
    decompose_bcs(BCs, comp::Integer)

Decomposes a list of boundary conditions for a given component index.
If a boundary condition specifies a vector value, it extracts the scalar value for that component.
"""
function decompose_bcs(BCs, comp::Integer)
    map(BCs) do bc
        if hasproperty(bc, :value) && bc.value isa AbstractArray
            # Standard BC with vector value (Dirichlet, Wall, etc)
            T = typeof(bc).name.wrapper
            return T(bc.ID, bc.value[comp], bc.IDs_range)
        elseif hasproperty(bc, :value) && hasproperty(bc.value, :value) && bc.value.value isa AbstractArray
            # Robin BC with vector RobinValue
            T = typeof(bc).name.wrapper
            new_val = @set bc.value.value = bc.value.value[comp]
            return T(bc.ID, new_val, bc.IDs_range)
        end
        return bc
    end
end

"""
    decompose(eqn::ModelEquation{VectorModel}) -> Vector{ModelEquation{ScalarModel}}

Decomposes a `VectorEquation` into an array of `ScalarEquation`s, one for each spatial dimension.
Vector-valued sources and boundary conditions are automatically extracted for the corresponding component.
Operators inside the equation are re-bound to the scalar component fields (`psi.x`, `psi.y`, etc.).
"""
function decompose(eqn::ModelEquation{VectorModel, M, E, S, P, ST}) where {M,E,S,P,ST}
    psi = get_phi(eqn)
    comps = decompose(psi)
    bcs = get_bcs(eqn)

    scalar_eqns = []
    for (i, phi_comp) in enumerate(comps)
        # 1. Decompose Boundary Conditions
        comp_bcs = decompose_bcs(bcs, i)

        # 2. Decompose Sources
        comp_sources = map(eqn.model.sources) do src
            if src.field isa VectorField
                fld = i == 1 ? src.field.x : (i == 2 ? src.field.y : src.field.z)
                return Src(fld, src.sign)
            else
                # Scalar sources apply equally (e.g. 0.0) or must be handled by user
                return src
            end
        end

        # 3. Re-bind Operators to the scalar component field
        # We must attach the mesh to phi_comp since VectorField components drop it
        phi_with_mesh = ScalarField(phi_comp.values, psi.mesh)
        comp_terms = Tuple(Operator(t.flux, phi_with_mesh, t.sign, t.type) for t in eqn.model.terms)

        comp_model = Model{length(comp_terms), length(comp_sources)}(
            comp_terms,
            comp_sources
        )

        comp_eqn = ModelEquation(
            ScalarModel(), comp_model, ScalarEquation(phi_with_mesh, comp_bcs),
            eqn.solver, eqn.preconditioner, eqn.setup
        )
        push!(scalar_eqns, comp_eqn)
    end
    return scalar_eqns
end

# Allow single ScalarEquation to pass through decompose safely
function decompose(eqn::ModelEquation{ScalarModel, M, E, S, P, ST}) where {M,E,S,P,ST}
    return [eqn]
end

# Decompose for PDEOperator (used when building monolithic systems from DSL)
function decompose(L::PDEOperator)
    # If the templates are already bound Operators, we can treat it as a ScalarEquation
    if length(L.templates) > 0 && !(L.templates[1] isa OperatorTemplate)
        phi = L.templates[1].phi
        model = Model{length(L.templates), length(L.sources)}(L.templates, L.sources)
        return [ModelEquation(ScalarModel(), model, ScalarEquation(phi, L.BCs), nothing, nothing, L.setup)]
    end
    error("PDEOperator must be bound to a field (e.g. L(phi)) before it can be decomposed for a monolithic system.")
end

# Decompose for Model
function decompose(model::Model)
    if length(model.terms) > 0
        phi = model.terms[1].phi
        return [ModelEquation(ScalarModel(), model, ScalarEquation(phi, ()), nothing, nothing, nothing)]
    end
    error("Model must have at least one term with a bound field to be decomposed.")
end
