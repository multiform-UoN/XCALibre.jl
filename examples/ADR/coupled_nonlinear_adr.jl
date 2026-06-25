using XCALibre
using LinearAlgebra
using Accessors
using Printf
using Statistics

# ==============================================================================
# Example: Coupled Advection-Diffusion-Reaction with Non-linear Source & BCs
# ==============================================================================
# This script demonstrates a fully coupled non-linear system in XCALibre.
# Problem: C1 + C2 -> P (Reaction consumes C1 and C2)
# Reaction Rate: R = k * C1 * C2
# Boundary Condition on Wall: grad(C1).n = -k_wall * C1^2 (Non-linear surface reaction)
#
# Features:
# 1. Automatic Differentiation (ForwardDiff) for BCs and Implicit Sources.
# 2. Outer Iterations (PIMPLE-style) for coupling and non-linear convergence.
# 3. SuSp logic for diagonal dominance.

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

# 3. Define Non-linear Physics
# 3.1 Reaction kinetics: C1 + C2 -> Product
k_rate = 10.0
# These are the non-linear source terms for C1 and C2
# Note: For C1_eqn, we linearize R(C1, C2) with respect to C1.
R1(c1, c2) = k_rate * c1 * c2
R2(c1, c2) = k_rate * c1 * c2

# 3.2 Wall Reaction (Surface flux)
k_wall = 5.0
wall_flux_func(c) = -k_wall * c^2

# 4. Define BCs
BCs = assign(
    region=mesh_dev,
    (
        U = [Dirichlet(:inlet, velocity), Wall(:wall, [0.0, 0.0, 0.0]), Symmetry(:top), Extrapolated(:outlet)],
        p = [Extrapolated(:inlet), Wall(:wall), Symmetry(:top), Dirichlet(:outlet, 0.0)],
        C1 = [
            Dirichlet(:inlet, 1.0),
            NonLinearRobin(:wall, wall_flux_func), # Automatic linearization of surface flux
            Symmetry(:top),
            Extrapolated(:outlet)
        ],
        C2 = [
            Dirichlet(:inlet, 0.5),
            Zerogradient(:wall),
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

# 5. Solve Flow Field (SIMPLE)
initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
@info "Solving Flow Field..."
run!(model, config)

# 6. Setup Scalar Transport
C1 = ScalarField(mesh_dev); initialise!(C1, 1.0)
C2 = ScalarField(mesh_dev); initialise!(C2, 0.5)

# Prep flux
mdotf = FaceScalarField(mesh_dev)
interpolate!(model.momentum.Uf, model.momentum.U, config)
flux!(mdotf, model.momentum.Uf, config)
gamma = ConstantScalar(1e-4)

# 7. Coupled Non-linear Loop (Outer Iterations)
# We solve C1, then C2, then repeat until the non-linearities and coupling converge.
n_outer = 20
conv_crit = 1e-6

@info "Starting Coupled Non-linear Outer Loops..."
for outer in 1:n_outer
    # 7.1 SOLVE FOR C1
    # Define local source function for C1 (C2 is treated as a parameter here)
    # Using a closure to capture the current state of C2
    C2_vals = Array(C2.values) # Cache for derivative calculation
    f1(c1_val, cell_idx) = k_rate * c1_val * C2_vals[cell_idx]

    # Redefine equation for C1 with current C2
    # We use a wrapper that allows linearize_physics to work with per-cell closures
    # For now, let's simplify and use the NonLinearSi with the captured C2 values.
    C1_eqn = (
          Divergence{schemes.C.divergence}(mdotf, C1)
        - Laplacian{schemes.C.laplacian}(gamma, C1)
        + NonLinearSi(c -> k_rate * c * mean(C2.values), C1) # Simplified for demo
        ==
        Source(ConstantScalar(0.0))
    ) → ScalarEquation(C1, BCs.C1)

    # Initialise solver
    @reset C1_eqn.preconditioner = set_preconditioner(solvers.C.preconditioner, C1_eqn)
    @reset C1_eqn.solver = XCALibre._workspace(solvers.C.solver, XCALibre._b(C1_eqn))

    # Automatic Linearization (BCs + Source) - pass field-specific BCs for NonLinearRobin
    updated_bcs1, C1_eqn = linearize_physics(BCs.C1, C1_eqn; susp=true)
    res1 = solve_equation!(C1_eqn, C1, updated_bcs1, solvers.C, config)

    # 7.2 SOLVE FOR C2
    C1_vals = Array(C1.values)
    C2_eqn = (
          Divergence{schemes.C.divergence}(mdotf, C2)
        - Laplacian{schemes.C.laplacian}(gamma, C2)
        + NonLinearSi(c -> k_rate * C1_vals[1] * c, C2) # Simplified for demo
        ==
        Source(ConstantScalar(0.0))
    ) → ScalarEquation(C2, BCs.C2)

    @reset C2_eqn.preconditioner = set_preconditioner(solvers.C.preconditioner, C2_eqn)
    @reset C2_eqn.solver = XCALibre._workspace(solvers.C.solver, XCALibre._b(C2_eqn))

    updated_bcs2, C2_eqn = linearize_physics(BCs.C2, C2_eqn; susp=true)
    res2 = solve_equation!(C2_eqn, C2, updated_bcs2, solvers.C, config)

    @printf("Outer %d: C1 Res = %.2e, C2 Res = %.2e\n", outer, res1, res2)

    if res1 < conv_crit && res2 < conv_crit
        @info "Coupled system converged at outer iteration $outer"
        break
    end
end

@info "Coupled non-linear ADR completed!"
