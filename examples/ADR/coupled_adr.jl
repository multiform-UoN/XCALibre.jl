using XCALibre
using LinearAlgebra
using Accessors

# 1. Setup Mesh
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "flatplate_2D_laminar.unv"), scale=0.001)

backend = CPU(); workgroup = 1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# 2. Define Physics for Flow
nu = 1e-4
velocity = [1.0, 0.0, 0.0]

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
)

BCs = assign(
    region=mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Extrapolated(:outlet),
            Wall(:wall, [0.0, 0.0, 0.0]),
            Symmetry(:top)
        ],
        p = [
            Extrapolated(:inlet),
            Dirichlet(:outlet, 0.0),
            Wall(:wall),
            Symmetry(:top)
        ],
        C1 = [
            Dirichlet(:inlet, 1.0),
            Extrapolated(:outlet),
            Neumann(:wall, 0.0),
            Symmetry(:top)
        ],
        C2 = [
            Dirichlet(:inlet, 0.0),
            Extrapolated(:outlet),
            Neumann(:wall, 0.0),
            Symmetry(:top)
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
    C = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=0.8)
)

runtime = Runtime(iterations=100, write_interval=100, time_step=1.0)
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

# 3. Solve Flow
initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
@info "Solving Flow Field..."
run!(model, config)

# 4. Setup Coupled Scalars (C1 + C2 -> P)
C1 = ScalarField(mesh_dev)
C2 = ScalarField(mesh_dev)
initialise!(C1, 1.0)
initialise!(C2, 0.0)

# Calculate mass flux from velocity field
mdotf = FaceScalarField(mesh_dev)
interpolate!(model.momentum.Uf, model.momentum.U, config)
flux!(mdotf, model.momentum.Uf, config)

gamma = ConstantScalar(1e-4)
k_rate = 10.0 # Reaction rate constant

# Pre-allocate implicit source fields for coupling
# S_imp1 = k * C2  (consumption of C1)
# S_imp2 = k * C1  (consumption of C2)
S_imp1 = ScalarField(mesh_dev)
S_imp2 = ScalarField(mesh_dev)

# Define ADR Equations
C1_eqn = (
      Divergence{schemes.C.divergence}(mdotf, C1)
    - Laplacian{schemes.C.laplacian}(gamma, C1)
    + Si(S_imp1, C1)
    ==
    Source(ConstantScalar(0.0))
) → ScalarEquation(C1, BCs.C1)

C2_eqn = (
      Divergence{schemes.C.divergence}(mdotf, C2)
    - Laplacian{schemes.C.laplacian}(gamma, C2)
    + Si(S_imp2, C2)
    ==
    Source(ConstantScalar(0.0))
) → ScalarEquation(C2, BCs.C2)

# Initialise solvers
@reset C1_eqn.preconditioner = set_preconditioner(solvers.C.preconditioner, C1_eqn)
@reset C1_eqn.solver = XCALibre._workspace(solvers.C.solver, XCALibre._b(C1_eqn))
@reset C2_eqn.preconditioner = set_preconditioner(solvers.C.preconditioner, C2_eqn)
@reset C2_eqn.solver = XCALibre._workspace(solvers.C.solver, XCALibre._b(C2_eqn))

@info "Solving Coupled ADR..."
for i in 1:runtime.iterations
    # Update coupling terms
    S_imp1.values .= k_rate .* C2.values
    S_imp2.values .= k_rate .* C1.values

    res1 = solve_equation!(C1_eqn, C1, BCs.C1, solvers.C, config)
    res2 = solve_equation!(C2_eqn, C2, BCs.C2, solvers.C, config)

    if i % 10 == 0
        println("Iteration $i, C1 Res: $res1, C2 Res: $res2")
    end
end

# Save results
@info "Saving results..."
writer = initialise_writer(VTK(), mesh_dev)
write_results(1, 1.0, mesh_dev, writer, BCs,
    ("U", model.momentum.U),
    ("p", model.momentum.p),
    ("C1", C1),
    ("C2", C2)
)
