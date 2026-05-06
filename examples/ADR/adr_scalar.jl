using XCALibre
using LinearAlgebra
using Accessors

# 1. Setup Mesh
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "flatplate_2D_laminar.unv"), scale=0.001)

backend = CPU(); workgroup = 1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# 2. Define Physics for Flow (to get a velocity field)
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
        C = [
            Dirichlet(:inlet, 1.0),
            Extrapolated(:outlet),
            Robin(:wall, a=1.0, b=1.0, value=0.0), # Surface reaction: C + dC/dn = 0
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
    C = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0)
)

runtime = Runtime(iterations=200, write_interval=100, time_step=1.0)
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

# 3. Solve Flow Field
initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
@info "Solving Flow Field..."
run!(model, config)

# 4. Setup Scalar Transport
C = ScalarField(mesh_dev)
initialise!(C, 0.0)

# Calculate mass flux from velocity field
mdotf = FaceScalarField(mesh_dev)
interpolate!(model.momentum.Uf, model.momentum.U, config)
flux!(mdotf, model.momentum.Uf, config)

gamma = ConstantScalar(1e-4) # Diffusion coefficient
k_vol = ConstantScalar(0.1)  # Volumetric reaction rate (consumption)

# Define ADR Equation
# We'll use the ModelFramework DSL
C_eqn = (
      Divergence{schemes.C.divergence}(mdotf, C)
    - Laplacian{schemes.C.laplacian}(gamma, C)
    + Si(k_vol, C) # Implicit reaction term k_vol * C
    ==
    Source(ConstantScalar(0.0))
) → ScalarEquation(C, BCs.C)

# Initialise solver for C
@reset C_eqn.preconditioner = set_preconditioner(solvers.C.preconditioner, C_eqn)
@reset C_eqn.solver = XCALibre._workspace(solvers.C.solver, XCALibre._b(C_eqn))

@info "Solving Scalar Transport..."
for i in 1:100
    res = solve_equation!(C_eqn, C, BCs.C, solvers.C, config)
    if i % 10 == 0
        println("Iteration $i, C Residual: $res")
    end
    if res < solvers.C.convergence
        println("C converged at iteration $i")
        break
    end
end

# Save results
@info "Saving results..."
writer = initialise_writer(VTK(), mesh_dev)
write_results(1, 1.0, mesh_dev, writer, BCs,
    ("U", model.momentum.U),
    ("p", model.momentum.p),
    ("C", C)
)
