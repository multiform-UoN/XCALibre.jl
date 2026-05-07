using XCALibre
using LinearAlgebra
using Accessors
using Printf

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

# Inject C into model for linearize_bcs to find it
# (XCALibre's Physics struct doesn't have custom scalar fields by default,
# so we create a simple wrapper or just pass C manually)
# For this example, I'll pass a dummy object that matches what linearize_bcs expects.
model_with_C = (U=model.momentum.U, p=model.momentum.p, C=C, mesh=mesh_dev)

mdotf = FaceScalarField(mesh_dev)
interpolate!(model.momentum.Uf, model.momentum.U, config)
flux!(mdotf, model.momentum.Uf, config)
gamma = ConstantScalar(1e-4)

@info "Solving Non-Linear Scalar Transport..."
for i in 1:50
    # 5.1 AUTOMATIC LINEARIZATION (linearize NonLinearRobin BCs for C field)
    new_C_bcs = linearize_bcs(BCs.C, C)

    # 5.2 Build/Update Equation with new BCs
    C_eqn = (
          Divergence{schemes.C.divergence}(mdotf, C)
        - Laplacian{schemes.C.laplacian}(gamma, C)
        ==
        Source(ConstantScalar(0.0))
    ) → ScalarEquation(C, new_C_bcs)

    # Initialise solver
    @reset C_eqn.preconditioner = set_preconditioner(solvers.C.preconditioner, C_eqn)
    @reset C_eqn.solver = XCALibre._workspace(solvers.C.solver, XCALibre._b(C_eqn))

    # 5.3 Solve
    res = solve_system!(C_eqn, solvers.C, C, nothing, config)
    
    if i % 10 == 0
        @printf("Iteration %d, C Res: %.2e, Avg C on Wall: %.4f\n", 
                i, res, mean(C.values[mesh.boundary_cellsID[BCs.C[2].IDs_range]]))
    end
    if res < solvers.C.convergence
        @info "Converged at iter $i"
        break
    end
end

@info "Non-linear ADR example completed!"
