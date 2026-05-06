using XCALibre
using LinearAlgebra
using Accessors
using Printf
using Statistics

# ==============================================================================
# Example: Non-linear ADR with Analytical Flow Field
# ==============================================================================
# This script demonstrates:
# 1. Bypassing the SIMPLE flow solver by using an analytical function for U.
# 2. Automated Newton linearization of a non-linear reaction source.
# 3. Use of the initialise!(field, func) API.

# 1. Setup Mesh
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "flatplate_2D_laminar.unv"), scale=0.001)

backend = CPU(); workgroup = 1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# 2. Define Analytical Flow Field
# instead of solving Navier-Stokes, we define U as a function of coordinates
# Let's use a simple parabolic or linear profile for demonstration.
analytical_U(x, y, z) = [1.0 * (1.0 - (y/0.1)^2), 0.0, 0.0] 

# Create a minimal model shell
model = Physics(
    time = Steady(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
)

# 3. Initialise Velocity Directly from Function (BYPASS FLOW SOLVER)
@info "Initialising analytical flow field..."
initialise!(model.momentum.U, analytical_U)

# 4. Define Physics for C
k_react = 0.5
reaction_func(c) = k_react * c^2

BCs = assign(
    (
        C = [
            Dirichlet(:inlet, 1.0),
            Zerogradient(:wall),
            Symmetry(:top),
            Extrapolated(:outlet)
        ],
    ),
    region=mesh_dev
)

schemes = (C = Schemes(divergence=Upwind, laplacian=Linear),)
solvers = (C = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0),)
config = Configuration(solvers=solvers, schemes=schemes, runtime=Runtime(iterations=100, write_interval=-1, time_step=1.0), hardware=hardware, boundaries=BCs)

# 5. Setup ADR
C = ScalarField(mesh_dev); initialise!(C, 1.0)
mdotf = FaceScalarField(mesh_dev)
# Calculate flux once from the analytical velocity field
interpolate!(model.momentum.Uf, model.momentum.U, config)
flux!(mdotf, model.momentum.Uf, config)
gamma = ConstantScalar(1e-4)

# 6. Solve Non-linear ADR Loop
C_eqn = (
      Divergence{schemes.C.divergence}(mdotf, C)
    - Laplacian{schemes.C.laplacian}(gamma, C)
    + NonLinearSi(reaction_func, C)
    ==
    Source(ConstantScalar(0.0))
) → ScalarEquation(C, BCs.C)

# Initialise solver
@reset C_eqn.preconditioner = set_preconditioner(solvers.C.preconditioner, C_eqn)
@reset C_eqn.solver = XCALibre._workspace(solvers.C.solver, XCALibre._b(C_eqn))

@info "Solving Non-Linear ADR with analytical flow..."
for i in 1:50
    global C_eqn
    # Automated Newton Linearization
    updated_bcs, C_eqn = linearize_physics(BCs, C_eqn; susp=true)
    
    res = solve_equation!(C_eqn, C, updated_bcs.C, solvers.C, config)
    
    if i % 10 == 0
        @printf("Iteration %d, C Res: %.2e, Mean C: %.4f\n", i, res, mean(C.values))
    end
    if res < solvers.C.convergence
        @info "Converged at iter $i"
        break
    end
end

# 7. Save Results
@info "Saving results..."
writer = initialise_writer(VTK(), mesh_dev)
write_results(1, 1.0, mesh_dev, writer, BCs, ("C", C))
