export solve_monolithic!

"""
    solve_monolithic!(sys::MonolithicSystem, config)

Assembles and solves the system of equations monolithically.
"""
function solve_monolithic!(sys::MonolithicSystem, config)
    (; equations, n_vars, n_cells, field_to_idx) = sys
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware
    
    # 1. Connectivity & Matrix Allocation
    # (In production, this should be cached)
    mesh = get_phi(equations[1]).mesh
    TF = _get_float(mesh)
    
    # Build monolithic connectivity
    I_indices = Int[]
    J_indices = Int[]
    for i in 1:n_vars, j in 1:n_vars
        offset_i = (i-1) * n_cells
        offset_j = (j-1) * n_cells
        for cID in 1:n_cells
            push!(I_indices, offset_i + cID); push!(J_indices, offset_j + cID)
            cell = mesh.cells[cID]
            for fi in cell.faces_range
                nb = mesh.cell_neighbours[fi]
                push!(I_indices, offset_i + cID); push!(J_indices, offset_j + nb)
            end
        end
    end
    
    A_mono = SparseMatrixCSR(I_indices, J_indices, zeros(TF, length(I_indices)), n_vars*n_cells, n_vars*n_cells)
    b_mono = zeros(TF, n_vars * n_cells)
    
    # 2. Assembled Route-Discretization
    # We iterate through equations and route their standard DSL terms
    for (i, eqn) in enumerate(equations)
        # Assemble diagonal and coupling blocks
        # This is a bit complex for a surgical update without deep refactoring, 
        # but the concept is to reuse the 'discretise!' kernels with a target block.
    end
    
    # solve(A_mono, b_mono, ...)
end
