"""
    monolithic_discretise!(sys::CoupledSystem, monolithic_A, monolithic_b, config)

Assembles all equations in the CoupledSystem into a single monolithic CSR matrix
and right-hand side vector.
"""
function monolithic_discretise!(sys::CoupledSystem, A_mono, b_mono, config)
    (; equations, n_vars, n_cells, field_to_idx) = sys
    (; hardware, runtime) = config
    (; backend) = hardware
    
    # 1. Reset Matrix and RHS
    A_mono.nzval .= 0.0
    b_mono .= 0.0
    
    rowptr = A_mono.rowptr
    colval = A_mono.colval
    nzval = A_mono.nzval

    # 2. Iterate through each equation i
    for i in 1:n_vars
        eqn = equations[i]
        phi_target = get_phi(eqn)
        mesh = phi_target.mesh
        cells = mesh.cells
        faces = mesh.faces
        cell_neighbours = mesh.cell_neighbours
        
        # 2.1 Loop through each TERM in equation i
        # Each term acts on some field 'phi_source' (j)
        for term in eqn.model.terms
            phi_source = term.phi
            j = field_to_idx[phi_source] # Index of the field being acted on
            
            # Use existing discretization logic!
            # We wrap the assembly to shift indices from (cID) to (i, j) blocks.
            
            # CPU implementation for now (prototyping monolithic routing)
            for cID in 1:n_cells
                cell = cells[cID]
                global_row = (i - 1) * n_cells + cID
                
                for fi in cell.faces_range
                    face = faces[fi]
                    nb = cell_neighbours[fi]
                    global_col = (j - 1) * n_cells + nb
                    
                    # Call standard scheme! (unmodified)
                    # It returns (ac, an) for the current term
                    ac, an = scheme!(term, nothing, cell, face, cells[nb], 0, 0, 0, fi, nothing, runtime)
                    
                    # Route to monolithic matrix
                    # Diagonal contribution (for current cell)
                    idx_diag = spindex(rowptr, colval, global_row, (j - 1) * n_cells + cID)
                    Atomix.@atomic nzval[idx_diag] += ac
                    
                    # Off-diagonal contribution (for neighbour cell)
                    idx_nb = spindex(rowptr, colval, global_row, global_col)
                    Atomix.@atomic nzval[idx_nb] += an
                end
                
                # Source contribution
                # RHS logic...
            end
        end
        
        # 2.2 Boundary Conditions for Equation i
        # Same routing logic applied to BC dispatch
    end
end

"""
    solve_monolithic!(sys::CoupledSystem, config)

Top-level API for solving a coupled system monolithically.
"""
function solve_monolithic!(sys::CoupledSystem, config)
    # Assemble A and b
    # (Future: Cache connectivity to avoid Set operations every step)
    I, J, V = monolithic_sparse_connectivity(sys)
    A_mono = SparseMatrixCSR(I, J, V, sys.n_vars * sys.n_cells, sys.n_vars * sys.n_cells)
    b_mono = zeros(sys.n_vars * sys.n_cells)
    
    monolithic_discretise!(sys, A_mono, b_mono, config)
    
    # Solve system using standard Krylov solvers
    # results = solve_system!(A_mono, b_mono, ...)
    
    # Map global solution vector back to individual fields
end
