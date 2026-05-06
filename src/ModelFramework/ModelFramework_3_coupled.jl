export MonolithicSystem, solve_monolithic!

"""
    MonolithicSystem(equations::Vector{ModelEquation})

A container for multiple equations that will be assembled into a single 
large sparse matrix and solved monolithically.
"""
struct MonolithicSystem{E<:Vector{<:ModelEquation}}
    equations::E
    n_vars::Int
    n_cells::Int
    field_to_idx::Dict{Any, Int}
end

function MonolithicSystem(eqns::Vector{<:ModelEquation})
    phi1 = get_phi(eqns[1])
    n_cells = length(phi1.mesh.cells)
    
    field_to_idx = Dict{Any, Int}()
    for (i, eqn) in enumerate(eqns)
        field_to_idx[get_phi(eqn)] = i
    end
    
    return MonolithicSystem(eqns, length(eqns), n_cells, field_to_idx)
end
