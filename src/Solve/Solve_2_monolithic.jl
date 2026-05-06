export solve_monolithic!

"""
    solve_monolithic!(sys::MonolithicSystem, bcs_list, config)

Assembles all equations in `sys` into a single block-CSR matrix and solves
with BiCGSTAB.  Iterates `runtime.iterations` times, re-assembling each
outer iteration so that semi-implicit BCs (e.g. Extrapolated) converge.

`bcs_list` must be an indexable collection where `bcs_list[j]` provides the
boundary conditions for the j-th field (in the same order as the equations
passed to `MonolithicSystem`).  Cross-field coupling terms automatically use
the BCs of their own field (`phi_j`), not the row equation's BCs.

# Example
    sys = MonolithicSystem([C1_eqn, C2_eqn], [C1, C2])
    res = solve_monolithic!(sys, (BCs.C1, BCs.C2), config)
"""
function solve_monolithic!(sys::MonolithicSystem, bcs_list, config)
    (; equations, phi_list, n_vars, n_cells, field_to_idx) = sys
    (; hardware, runtime) = config
    mesh = phi_list[1].mesh
    TF = _get_float(mesh)
    iterations = runtime.iterations

    # --- build monolithic sparsity pattern (all n_vars×n_vars blocks share mesh topology) ---
    I_idx = Int[]
    J_idx = Int[]
    for i in 1:n_vars, j in 1:n_vars
        row_off = (i - 1) * n_cells
        col_off = (j - 1) * n_cells
        for cID in 1:n_cells
            push!(I_idx, row_off + cID)
            push!(J_idx, col_off + cID)
            cell = mesh.cells[cID]
            for fi in cell.faces_range
                nb = mesh.cell_neighbours[fi]
                push!(I_idx, row_off + cID)
                push!(J_idx, col_off + nb)
            end
        end
    end

    N = n_vars * n_cells
    # Use standard sparse() to sum duplicates, then convert to CSR
    A_sparse = sparse(I_idx, J_idx, zeros(TF, length(I_idx)), N, N)
    A_csr = SparseMatrixCSR(A_sparse)
    b_mono = zeros(TF, N)

    # Pre-allocate Krylov workspace (reused across outer iterations)
    A_op = SparseXCSR(A_csr)
    ws = BicgstabWorkspace(KrylovConstructor(b_mono))

    # Block-Jacobi Preconditioner: 
    # Extract diagonal entries of A_mono to build a simple Jacobi preconditioner
    P_diag = zeros(TF, N)

    res = TF(NaN)
    for iter in 1:iterations

        # --- assemble interior face contributions ---
        monolithic_discretise!(sys, A_csr, b_mono, config)

        # --- apply boundary conditions ---
        monolithic_apply_bcs!(sys, A_csr, b_mono, bcs_list, config)

        # Update Jacobi preconditioner
        for k in 1:N
            diag_idx = spindex(A_csr.rowptr, A_csr.colval, k, k)
            d_val = A_csr.nzval[diag_idx]
            P_diag[k] = abs(d_val) > eps(TF) ? 1.0 / d_val : 1.0
        end

        # Solve with diagonal scaling
        b_scaled = b_mono .* P_diag
        # Matrix scaling: P_diag * A * P_diag? No, just left preconditioning.
        # krylov_solve! doesn't take a vector for M, it takes an operator.
        M_op = opDiagonal(P_diag)

        krylov_solve!(ws, A_op, b_mono; M=M_op, atol=1e-12, rtol=1e-10, itmax=5000)

        if !ws.stats.solved
            @warn "Monolithic BiCGSTAB iter=$iter: did not converge (niter=$(Krylov.iteration_count(ws)))"
        end

        x = ws.x

        # --- write results back to individual fields ---
        # Use sys.phi_list (not get_phi(eqn)) since get_phi returns the first *term*'s phi,
        # which may be a cross-field coupling term rather than the equation's self-field.
        for (i, phi) in enumerate(phi_list)
            row_off = (i - 1) * n_cells
            phi.values .= x[row_off+1:row_off+n_cells]
        end

        # Estimate residual
        res = isempty(ws.stats.residuals) ? TF(NaN) : TF(ws.stats.residuals[end])
    end

    return res
end
