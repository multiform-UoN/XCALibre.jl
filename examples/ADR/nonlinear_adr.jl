using XCALibre
using LinearAlgebra
using Printf
using Statistics

# 1. Setup Mesh
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "flatplate_2D_laminar.unv"), scale=0.001)

backend = CPU(); workgroup = 1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# 2. Define Physics for Flow (Fixed velocity)
nu = 1e-4
velocity = [1.0, 0.0, 0.0]

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
)

# 3. Define BCs including NonLinearRobin
# Let's say the wall has a non-linear flux: grad(C).n = -k * C^2
k_surface = 10.0
flux_func(c) = -k_surface * c^2

BCs = assign(
    region=mesh_dev,
    (
        U = [Dirichlet(:inlet, velocity), Wall(:wall, [0.0, 0.0, 0.0]), Symmetry(:top), Extrapolated(:outlet)],
        p = [Extrapolated(:inlet), Wall(:wall), Symmetry(:top), Dirichlet(:outlet, 0.0)],
        C = [
            Dirichlet(:inlet, 1.0),
            NonLinearRobin(:wall, flux_func), # AUTOMATIC LINEARIZATION
            Symmetry(:top),
            Extrapolated(:outlet)
        ]
    )
)

schemes = (
    U = Schemes(divergence=Upwind),
    p = Schemes(divergence=Linear),
    C = Schemes(divergence=Upwind, laplacian=Linear)
)

solvers = (
    U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-6, relax=0.7),
    p = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-6, relax=0.3),
    C = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0)
)

runtime = Runtime(iterations=100, write_interval=-1, time_step=1.0)
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

# 4. Solve Flow Field (One pass)
initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
@info "Solving Flow Field..."
run!(model, config)

# 5. Setup Scalar Transport
C = ScalarField(mesh_dev)
initialise!(C, 1.0)

mdotf = FaceScalarField(mesh_dev)
interpolate!(model.momentum.Uf, model.momentum.U, config)
flux!(mdotf, model.momentum.Uf, config)
gamma = ConstantScalar(1e-4)

L_C = ((
      Divergence{schemes.C.divergence}(mdotf)
    - Laplacian{schemes.C.laplacian}(gamma)
    ==
    Source(ConstantScalar(0.0))
) → BCs.C) → solvers.C

C_eqn = L_C(C)

@info "Solving scalar transport with Newton-linearised NonLinearRobin BC..."
last_res = Ref(Inf)
for iter in 1:50
    # NonLinearRobin is linearised outside the boundary kernel. This keeps the
    # nonlinear BC visible without rebuilding solver/preconditioner plumbing by hand.
    C_bcs, linear_eqn, _ = linearize_physics(BCs.C, C_eqn)
    iter_res = solve_equation!(linear_eqn, C, C_bcs, solvers.C, config)
    last_res[] = iter_res

    if iter % 10 == 0 || iter_res < solvers.C.convergence
        @printf("Iteration %d, C residual %.2e\n", iter, iter_res)
    end
    iter_res < solvers.C.convergence && break
end

wall_mean = mean(C.values[mesh_dev.boundary_cellsID[BCs.C[2].IDs_range]])
@printf("NonLinearRobin ADR: final residual=%.2e, wall mean C=%.4f\n", last_res[], wall_mean)

@info "Non-linear ADR example completed!"
