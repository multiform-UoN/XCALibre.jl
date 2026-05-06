# ---------------------------------------------------------------------------
# Outer constructor for MonolithicSystem.
# Builds field→block-index map from the first term of each equation (self-field).
# ---------------------------------------------------------------------------
"""
    MonolithicSystem(eqns, phi_list)

Construct a monolithic block-coupled system.

`phi_list[i]` must be the ScalarField that equation `i` solves for (the
"self field" of that equation).  This cannot be inferred automatically
because the first term of an equation may be a cross-field coupling term.

# Example
    sys = MonolithicSystem([C1_eqn, C2_eqn], [C1, C2])
"""
function MonolithicSystem(eqns::Vector{<:ModelEquation}, phi_list)
    n_cells = length(phi_list[1].mesh.cells)
    # Key: objectid of the mutable values array — stable across immutable struct copies
    field_to_idx = Dict{UInt, Int}()
    for (i, phi) in enumerate(phi_list)
        field_to_idx[objectid(phi.values)] = i
    end
    MonolithicSystem(eqns, phi_list, length(eqns), n_cells, field_to_idx)
end
