# ---------------------------------------------------------------------------
# Outer constructor for MonolithicSystem.
# Builds field→block-index map from the first term of each equation (self-field).
# ---------------------------------------------------------------------------
"""
    MonolithicSystem(eqns, phi_list)

Construct a monolithic block-coupled system.

`phi_list[i]` must be the field that equation `i` solves for.
Any `VectorEquation` and `VectorField` passed will be automatically decomposed 
into their constituent scalar components.

# Example
    sys = MonolithicSystem([p_eqn, U_eqn], [p, U])
"""
function MonolithicSystem(eqns, phi_list)
    flat_eqns = []
    flat_phis = []
    
    for (eqn, phi) in zip(eqns, phi_list)
        append!(flat_eqns, decompose(eqn))
        append!(flat_phis, decompose(phi))
    end
    
    n_cells = length(flat_phis[1].mesh.cells)
    # Key: objectid of the mutable values array — stable across immutable struct copies
    field_to_idx = Dict{UInt, Int}()
    for (i, phi) in enumerate(flat_phis)
        field_to_idx[objectid(phi.values)] = i
    end
    MonolithicSystem(Vector{ModelEquation}(flat_eqns), flat_phis, length(flat_eqns), n_cells, field_to_idx)
end
