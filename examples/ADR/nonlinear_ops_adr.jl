using XCALibre
using LinearAlgebra
using Accessors
using Printf
using Statistics

# ==============================================================================
# Example: Non-linear Operators ADR (Newton Linearization)
# ==============================================================================
# This script tests non-linear Divergence and Laplacian terms.
# PDE: div(U * f(C)) - div(Gamma * grad(g(C))) = 0
# with f(C) = C^2 and g(C) = C^1.5

# 1. Setup Mesh
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "flatplate_2D_laminar.unv"), scale=0.001)

backend = CPU(); workgroup = 1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# 2. Define Analytical Flow Field
analytical_U(x, y, z) = [1.0, 0.0, 0.0]
model = Physics(time=Steady(), energy=Energy{Isothermal}(), domain=mesh_dev)
initialise!(model.momentum.U, analytical_U)

# 3. Define Non-linear Physics functions
f_advect(c) = c^2
g_diffuse(c) = c^1.5

# 4. Define BCs
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
solvers = (C = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-7, relax=1.0),)
config = Configuration(solvers=solvers, schemes=schemes, runtime=Runtime(iterations=100, write_interval=-1, time_step=1.0), hardware=hardware, boundaries=BCs)

# 5. Setup ADR
C = ScalarField(mesh_dev); initialise!(C, 0.8)
mdotf = FaceScalarField(mesh_dev)
interpolate!(model.momentum.Uf, model.momentum.U, config)
flux!(mdotf, model.momentum.Uf, config)
gamma = ConstantScalar(1e-4)

# 6. Non-linear ADR Equation Definition
C_eqn_template = (
      Divergence{schemes.C.divergence}(mdotf, f_advect, C) # div(U * C^2)
    - Laplacian{schemes.C.laplacian}(gamma, g_diffuse, C)  # div(G * grad(C^1.5))
    ==
    Source(ConstantScalar(0.0))
) → ScalarEquation(C, BCs.C)

# Initialise solver
@reset C_eqn_template.preconditioner = set_preconditioner(solvers.C.preconditioner, C_eqn_template)
@reset C_eqn_template.solver = XCALibre._workspace(solvers.C.solver, XCALibre._b(C_eqn_template))

@info "Solving Non-Linear Operators ADR..."
# ad_backend = :enzyme
ad_backend = :forwarddiff

total_time = 0.0
for i in 1:20
    global total_time

    # Timing start
    iter_start = time_ns()

    # 6.1 Newton Linearization
    updated_bcs, C_eqn = linearize_physics(BCs, C_eqn_template; ad_backend=ad_backend)

    # 6.2 Linear Solve
    res = solve_equation!(C_eqn, C, updated_bcs.C, solvers.C, config)

    # Timing end
    iter_end = time_ns()
    iter_time = (iter_end - iter_start) / 1e9
    total_time += iter_time

    @printf("Iteration %d: Res = %.2e, Mean C = %.4f, Time = %.4fs\n",
            i, res, mean(C.values), iter_time)

    if res < solvers.C.convergence
        @info "Converged at iter $i"
        break
    end
end

@printf("\nTotal solution time: %.4fs (Avg: %.4fs/iter)\n", total_time, total_time/20)

# 7. Save Results
@info "Saving results..."
writer = initialise_writer(VTK(), mesh_dev)
write_results(1, 1.0, mesh_dev, writer, BCs, ("C", C))
