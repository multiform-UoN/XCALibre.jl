using XCALibre
using LinearAlgebra
using Accessors
using Printf
using Statistics

# ==============================================================================
# Example: Scalar Transport with Non-linear Implicit Source Term
# ==============================================================================
# This script demonstrates the use of NonLinearSi, a custom operator I added 
# to XCALibre to allow automatic linearization of arbitrary reaction terms F(u).
# It uses ForwardDiff.jl to compute the Jacobian (diagonal contribution) 
# and automatically updates the system matrix.

# 1. Setup Mesh
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "flatplate_2D_laminar.unv"), scale=0.001)

backend = CPU(); workgroup = 1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# 2. Define Physics for Flow (Fixed velocity field)
nu = 1e-4
velocity = [1.0, 0.0, 0.0]

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
)

# 3. Define Non-linear Source Term F(C)
# Let's use a standard Michaelis-Menten or simple polynomial reaction
# R(C) = k * C / (K + C)  or  R(C) = k * C^2
k_react = 0.5
reaction_func(c) = k_react * c^2

# 4. Define BCs
BCs = assign(
    region=mesh_dev,
    (
        U = [Dirichlet(:inlet, velocity), Wall(:wall, [0.0, 0.0, 0.0]), Symmetry(:top), Extrapolated(:outlet)],
        p = [Extrapolated(:inlet), Wall(:wall), Symmetry(:top), Dirichlet(:outlet, 0.0)],
        C = [
            Dirichlet(:inlet, 1.0),   # Constant concentration at inlet
            Zerogradient(:wall),      # No flux on wall
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

config = Configuration(solvers=solvers, schemes=schemes, runtime=Runtime(iterations=100, write_interval=-1, time_step=1.0), hardware=hardware, boundaries=BCs)

# 5. Solve Flow Field
initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
@info "Solving Flow Field (SIMPLE)..."
run!(model, config)

# 6. Setup Non-linear ADR Equation
C = ScalarField(mesh_dev)
initialise!(C, 1.0)

# Calculate mass flux for divergence term
mdotf = FaceScalarField(mesh_dev)
interpolate!(model.momentum.Uf, model.momentum.U, config)
flux!(mdotf, model.momentum.Uf, config)
gamma = ConstantScalar(1e-4)

# Build Equation using the new NonLinearSi operator
# This stores the function and is linearized inside the loop.
C_eqn_template = (
      Divergence{schemes.C.divergence}(mdotf, C)
    - Laplacian{schemes.C.laplacian}(gamma, C)
    + NonLinearSi(reaction_func, C)  # <--- NEW OPERATOR
    ==
    Source(ConstantScalar(0.0))
) → ScalarEquation(C, BCs.C)

# Initialise solver
@reset C_eqn_template.preconditioner = set_preconditioner(solvers.C.preconditioner, C_eqn_template)
@reset C_eqn_template.solver = XCALibre._workspace(solvers.C.solver, XCALibre._b(C_eqn_template))

@info "Solving Non-Linear Implicit Source ADR..."
# Choose AD backend: :forwarddiff or :enzyme
ad_backend = :enzyme 

total_time = 0.0
for i in 1:100
    global total_time
    # 6.1 AUTOMATIC LINEARIZATION
    iter_start = time_ns()

    updated_bcs, C_eqn = linearize_physics(BCs, C_eqn_template; susp=true, ad_backend=ad_backend)

    # 6.2 Solve
    res = solve_equation!(C_eqn, C, updated_bcs.C, solvers.C, config)

    iter_end = time_ns()
    iter_time = (iter_end - iter_start) / 1e9
    total_time += iter_time

    if i % 10 == 0
        @printf("Iteration %d, C Res: %.2e, Mean C: %.4f, Time: %.4fs\n", 
                i, res, mean(C.values), iter_time)
    end

    if res < solvers.C.convergence
        @info "Converged at iteration $i"
        break
    end
end

@info "Non-linear Source ADR example completed!"

# 7. Save Results
@info "Saving Results to VTK..."
writer = initialise_writer(VTK(), mesh_dev)
write_results(1, 1.0, mesh_dev, writer, BCs,
    ("U", model.momentum.U),
    ("p", model.momentum.p),
    ("C", C)
)
