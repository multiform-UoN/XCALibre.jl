using XCALibre
using LinearAlgebra
using Accessors
using Printf
using Statistics
using Optim

# ==============================================================================
# Example: Constrained Optimization of Dispersivity via Homogenization
# ==============================================================================
# Objective: Maximize the effective longitudinal dispersivity (α_xx) 
#            with respect to porosity (ϕ) for a fixed Darcian velocity (U).
#
# Physics:
# 1. Fixed Darcian velocity U_macro.
# 2. Pore velocity v = U_macro / ϕ.
# 3. Microscopic Dispersion D = D_mol.
# 4. Homogenization solves for corrector X: v.grad(X) - D.lapl(X) = -v
# 5. Effective Dispersion D* = D_mol + <vX>
# 6. Dispersivity α = (D* - D_mol) / |v|
# ==============================================================================

# 1. Setup Static Mesh (Unit Cell)
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "laplace_unit_5by5.unv")) # Simple RVE

backend = CPU(); workgroup=1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# Fixed Darcian Velocity (m/s)
U_macro = 1e-3 
D_mol = 1e-6

# 2. Define Objective Function
function objective_dispersivity(params)
    phi_val = params[1] # Porosity
    
    # Constraints check (though Optim handles it, good for safety)
    if phi_val <= 0 || phi_val >= 1.0 return 1e10 end
    
    # Calculate pore velocity
    v_pore = U_macro / phi_val
    
    # Setup BCs for Corrector X (Periodic-like or Extrapolated for this simple RVE)
    BCs = assign(
        (
            X = [
                Extrapolated(:left_wall),
                Extrapolated(:right_wall),
                Zerogradient(:bottom_wall),
                Zerogradient(:upper_wall)
            ],
        ),
        region=mesh_dev
    )

    schemes = (X = Schemes(divergence=Upwind, laplacian=Linear),)
    solvers = (X = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),)
    config = Configuration(solvers=solvers, schemes=schemes, runtime=Runtime(iterations=1, time_step=1.0, write_interval=-1), hardware=hardware, boundaries=BCs)

    # Setup Fields
    X = ScalarField(mesh_dev); initialise!(X, 0.0)
    
    # Flux based on uniform pore velocity in X
    mdotf = FaceScalarField(mesh_dev)
    for fID in eachindex(mdotf.values)
        mdotf.values[fID] = v_pore * mesh_dev.faces[fID].area * mesh_dev.faces[fID].normal[1]
    end
    
    # RHS for corrector: -v
    sourceX = ScalarField(mesh_dev); initialise!(sourceX, -v_pore)

    # Corrector Equation: v.grad(X) - D.lapl(X) = -v
    X_eqn = (
          Divergence{schemes.X.divergence}(mdotf, X)
        - Laplacian{schemes.X.laplacian}(ConstantScalar(D_mol), X)
        ==
        Source(sourceX)
    ) → ScalarEquation(X, BCs.X)

    # Solve
    @reset X_eqn.preconditioner = set_preconditioner(solvers.X.preconditioner, X_eqn)
    @reset X_eqn.solver = XCALibre._workspace(solvers.X.solver, XCALibre._b(X_eqn))
    solve_equation!(X_eqn, X, BCs.X, solvers.X, config)
    
    # Calculate Effective Dispersion: D* = D_mol + <v*X>
    vols = [cell.volume for cell in mesh.cells]
    vol_tot = sum(vols)
    vX_avg = sum(v_pore .* X.values .* vols) / vol_tot
    D_eff = D_mol + vX_avg
    
    # Dispersivity α = (D* - D_mol) / |v|
    alpha = abs(vX_avg / v_pore)
    
    # Return negative for maximization
    return -alpha
end

# 3. Optimization using Optim.jl
@info "Starting Optimization of Dispersivity vs Porosity..."
# Initial guess [porosity]
p0 = [0.5]
# Constraints: 0.1 < phi < 0.9
lower = [0.1]
upper = [0.9]

results = optimize(objective_dispersivity, lower, upper, p0, Fminbox(GradientDescent()))

# 4. Results
best_phi = Optim.minimizer(results)[1]
max_alpha = -Optim.minimum(results)

@printf("\nOptimization Results:\n")
@printf("Optimal Porosity ϕ:     %.4f\n", best_phi)
@printf("Maximized Dispersivity α: %.6f m\n", max_alpha)
@printf("Fixed Darcian Velocity U: %.4e m/s\n", U_macro)

@info "Optimization Finished!"
