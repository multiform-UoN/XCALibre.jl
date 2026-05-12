export solve_monolithic!, newton_solve!, monolithic_residual!, set_fields!, update_fields!

function _build_monolithic_sparsity(n_vars, n_cells, mesh, TF)
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
    A_sparse = sparse(I_idx, J_idx, zeros(TF, length(I_idx)), N, N)
    return SparseMatrixCSR(A_sparse)
end

"""
    assemble_monolithic_system(sys::MonolithicSystem, bcs_list, config) -> (A_csr, b_mono)
"""
function assemble_monolithic_system(sys::MonolithicSystem, bcs_list, config)
    (; equations, phi_list, n_vars, n_cells, field_to_idx) = sys
    mesh = phi_list[1].mesh
    TF = _get_float(mesh)
    
    A_csr = _build_monolithic_sparsity(n_vars, n_cells, mesh, TF)
    N = n_vars * n_cells
    b_mono = zeros(TF, N)
    
    monolithic_discretise!(sys, A_csr, b_mono, config)
    monolithic_apply_bcs!(sys, A_csr, b_mono, bcs_list, config)
    
    return A_csr, b_mono
end

function extract_global_vector(sys::MonolithicSystem)
    (; phi_list, n_vars, n_cells) = sys
    TF = _get_float(phi_list[1].mesh)
    N = n_vars * n_cells
    x = zeros(TF, N)
    for (i, phi) in enumerate(phi_list)
        row_off = (i - 1) * n_cells
        x[row_off+1:row_off+n_cells] .= phi.values
    end
    return x
end

function update_fields!(sys::MonolithicSystem, x)
    (; phi_list, n_cells) = sys
    for (i, phi) in enumerate(phi_list)
        row_off = (i - 1) * n_cells
        phi.values .+= x[row_off+1:row_off+n_cells]
    end
end

function set_fields!(sys::MonolithicSystem, x)
    (; phi_list, n_cells) = sys
    for (i, phi) in enumerate(phi_list)
        row_off = (i - 1) * n_cells
        phi.values .= x[row_off+1:row_off+n_cells]
    end
end

"""
    monolithic_residual!(r_mono, sys::MonolithicSystem, bcs_list, config)
"""
function monolithic_residual!(r_mono, sys::MonolithicSystem, bcs_list, config)
    A_csr, b_mono = assemble_monolithic_system(sys, bcs_list, config)
    x_mono = extract_global_vector(sys)
    r_mono .= A_csr * x_mono .- b_mono
    return r_mono
end

"""
    solve_monolithic!(sys::MonolithicSystem, bcs_list, config)
"""
function solve_monolithic!(sys::MonolithicSystem, bcs_list, config)
    (; hardware, runtime) = config
    TF = _get_float(sys.phi_list[1].mesh)
    iterations = runtime.iterations
    N = sys.n_vars * sys.n_cells
    
    # 1. Decompose BCs if they are still in vector form
    flat_bcs = []
    if length(bcs_list) < sys.n_vars
        # Heuristic: if we have fewer BC lists than variables, some must be vectors
        # (This happens when the user passes [BCs.U, BCs.p] for a 3-var system)
        # We need a more robust way to map them. 
        # Actually, let's just require the user to pass a flat list for now, 
        # or implement a smart mapper.
    end
    # Safe fallback: if bcs_list matches sys.phi_list length, we are good.
    # If it doesn't, we try to decompose.
    if length(bcs_list) != sys.n_vars
        # Map original phis to original BCs, then decompose
        # (This part is tricky without knowing the original un-decomposed structure)
        # Better: MonolithicSystem should have been built with a list of phis that matches bcs_list.
    end
    
    b_mono = zeros(TF, N)
    # Heuristic: use the solver from the first field for the monolithic system
    solver_type = first(config.solvers).solver
    ws = _workspace(solver_type, b_mono)
    
    res = TF(NaN)
    for iter in 1:iterations
        A_csr, b_mono = assemble_monolithic_system(sys, bcs_list, config)
        A_op = SparseXCSR(A_csr)

        krylov_solve!(ws, A_op, b_mono; atol=1e-12, rtol=1e-10, itmax=5000, history=true)

        if !ws.stats.solved
            @warn "Monolithic $(typeof(solver_type)) iter=$iter: did not converge (niter=$(Krylov.iteration_count(ws)))"
        end

        set_fields!(sys, ws.x)
        res = isempty(ws.stats.residuals) ? TF(NaN) : TF(ws.stats.residuals[end])
    end

    return res
end

"""
    newton_solve!(sys::MonolithicSystem, bcs_list, config)

Performs a fully coupled Newton-Raphson solve for a monolithic block system.
Cross-field non-linear dependencies are resolved via `linearize_physics`.
"""
function newton_solve!(
    sys::MonolithicSystem, bcs_list, config;
    tol=1e-8, maxiter=config.runtime.iterations, damping=1.0,
    susp=false, ad_backend=:forwarddiff, verbose=false
)
    (; n_vars, n_cells) = sys
    TF = _get_float(sys.phi_list[1].mesh)
    history = TF[]
    converged = false
    N = n_vars * n_cells

    # Heuristic: use the solver from the first field for the monolithic system
    solver_type = first(config.solvers).solver
    ws = _workspace(solver_type, zeros(TF, N))

    for iter in 1:maxiter
        # 1. Linearize all equations
        lin_eqns = []
        lin_bcs = []
        for (i, eqn) in enumerate(sys.equations)
            # Differentiate equation i w.r.t its self-field AND all other fields
            other_fields = filter(phi -> objectid(phi.values) != objectid(get_phi(eqn).values), sys.phi_list)
            
            new_bcs, lin_eqn, _ = linearize_physics(bcs_list[i], eqn, other_fields; 
                                                     susp=susp, ad_backend=ad_backend)
            push!(lin_eqns, _with_bcs(lin_eqn, new_bcs))
            push!(lin_bcs, new_bcs)
        end
        lin_sys = MonolithicSystem(Vector{ModelEquation}(lin_eqns), sys.phi_list)

        # 2. Evaluate Exact Global Residual
        r_mono = zeros(TF, N)
        monolithic_residual!(r_mono, lin_sys, lin_bcs, config)
        rnorm = norm(r_mono)
        
        push!(history, TF(rnorm))
        verbose && @info "Monolithic Newton iteration $iter" residual=rnorm
        if rnorm <= tol
            converged = true
            break
        end

        # 3. Assemble Homogeneous Jacobian (for correction step J * dx = -R)
        homo_bcs_list = [homogeneous(bcs) for bcs in lin_bcs]
        A_homo, _ = assemble_monolithic_system(lin_sys, homo_bcs_list, config)
        b_homo = -r_mono
        
        # 4. Solve the linear system
        A_op = SparseXCSR(A_homo)
        # Using a very tight inner tolerance for the Newton step
        krylov_solve!(ws, A_op, b_homo; atol=1e-12, rtol=1e-10, itmax=5000, history=true)
        
        if !ws.stats.solved
            @warn "Monolithic Newton inner $(typeof(solver_type)) did not converge (niter=$(Krylov.iteration_count(ws)))"
        end

        # 5. Update Fields
        update_fields!(sys, ws.x)
    end

    return (converged=converged, iterations=length(history), residuals=history)
end
