using XCALibre
using LinearAlgebra
using Accessors
using Printf
using Statistics

# ==============================================================================
# Example: First Block-Coupled Scalar Problem
# ==============================================================================
# PDE 1: ∂t C1 - D ∇² C1 + k12 C2 = 0
# PDE 2: ∂t C2 - D ∇² C2 + k21 C1 = 0
#
# This script uses the new CoupledSi operator to define cross-dependencies.
# In this segregated implementation, it facilitates easier iterative coupling.
# ==============================================================================

# 1. Setup Mesh
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.01)

backend = CPU(); workgroup=1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# 2. Define Physics
D = 1e-4
k_coupling = 1.0 # Coupling coefficient

BCs = assign(
    (
        C1 = [Dirichlet(:inlet, 1.0), Zerogradient(:bottom), Zerogradient(:top), Extrapolated(:outlet)],
        C2 = [Dirichlet(:inlet, 0.0), Zerogradient(:bottom), Zerogradient(:top), Extrapolated(:outlet)]
    ),
    region=mesh_dev
)

schemes = (C = Schemes(time=Euler, laplacian=Linear),)
solvers = (C = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=0.8),)
config = Configuration(solvers=solvers, schemes=schemes, runtime=Runtime(iterations=100, write_interval=-1, time_step=0.1), hardware=hardware, boundaries=BCs)

# 3. Setup Fields
C1 = ScalarField(mesh_dev); initialise!(C1, 1.0)
C2 = ScalarField(mesh_dev); initialise!(C2, 0.0)

# 4. Define Coupled Equations using PDEOperator
L_C1 = ((
      Time{Euler}()
    - Laplacian{Linear}(ConstantScalar(D))
    + Si(ConstantScalar(k_coupling), C2) # k_12 * C2
    ==
    Source(0.0)
) → BCs.C1) → solvers.C

L_C2 = ((
      Time{Euler}()
    - Laplacian{Linear}(ConstantScalar(D))
    + Si(ConstantScalar(k_coupling), C1) # k_21 * C1
    ==
    Source(0.0)
) → BCs.C2) → solvers.C

C1_eqn = L_C1(C1)
C2_eqn = L_C2(C2)

# Initialise Solvers (Pre-allocation)
@reset C1_eqn.preconditioner = set_preconditioner(solvers.C.preconditioner, C1_eqn)
@reset C1_eqn.solver = XCALibre._workspace(solvers.C.solver, XCALibre._b(C1_eqn))

@reset C2_eqn.preconditioner = set_preconditioner(solvers.C.preconditioner, C2_eqn)
@reset C2_eqn.solver = XCALibre._workspace(solvers.C.solver, XCALibre._b(C2_eqn))

@info "Solving Coupled Scalar System (Segregated outer iterations)..."
for step in 1:10
    # Outer iterations for explicit coupling
    for outer in 1:3
        res1 = solve_equation!(C1_eqn, config)
        res2 = solve_equation!(C2_eqn, config)

        if outer == 3
            @printf("Step %d: C1 Res = %.2e, C2 Res = %.2e, Mean C1 = %.4f\n",
                    step, res1, res2, mean(C1.values))
        end
    end
end

@info "Coupled Scalar Example Completed!"
