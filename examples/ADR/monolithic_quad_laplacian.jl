using XCALibre
using LinearAlgebra
using Accessors
using Printf
using Statistics

# ==============================================================================
# Example: quadLaplacian Monolithic Problem
# ==============================================================================
# System:
# Eq 1: Laplacian(a11, C1) + Laplacian(a12, C2) = 0
# Eq 2: Laplacian(a21, C1) + Laplacian(a22, C2) = 0
#
# This solves the system monolithically in a single sparse matrix:
# [ L(a11)  L(a12) ] [ C1 ]   [ 0 ]
# [ L(a21)  L(a22) ] [ C2 ] = [ 0 ]
# ==============================================================================

# 1. Setup Mesh
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.01)

backend = CPU(); workgroup=1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# 2. Physics & Coefficients
a11, a12 = 1.0, 0.2
a21, a22 = 0.3, 0.8

BCs = assign(
    (
        C1 = [Dirichlet(:inlet, 1.0), Zerogradient(:bottom), Zerogradient(:top), Extrapolated(:outlet)],
        C2 = [Dirichlet(:inlet, 0.0), Zerogradient(:bottom), Zerogradient(:top), Extrapolated(:outlet)]
    ),
    region=mesh_dev
)

schemes = (C = Schemes(laplacian=Linear),)
runtime = Runtime(iterations=50, write_interval=50, time_step=1)

# 3. Setup Fields
C1 = ScalarField(mesh_dev); initialise!(C1, 0.0)
C2 = ScalarField(mesh_dev); initialise!(C2, 0.0)

solvers = (C1 = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
               convergence=1e-8, relax=1.0),)
config = Configuration(schemes=schemes, solvers=solvers, runtime=runtime,
                       hardware=hardware, boundaries=BCs)

# 4. Define Equations using standard operators on cross-fields
C1_eqn = (
      - Laplacian{schemes.C.laplacian}(ConstantScalar(a11), C1)
      - Laplacian{schemes.C.laplacian}(ConstantScalar(a12), C2) # <--- Coupling
      ==
      Source(ConstantScalar(0.0))
) → ScalarEquation(C1, BCs.C1)

C2_eqn = (
      - Laplacian{schemes.C.laplacian}(ConstantScalar(a21), C1) # <--- Coupling
      - Laplacian{schemes.C.laplacian}(ConstantScalar(a22), C2)
      ==
      Source(ConstantScalar(0.0))
) → ScalarEquation(C2, BCs.C2)

# 5. Monolithic Solve
@info "Initialising Monolithic quadLaplacian System..."
sys = MonolithicSystem([C1_eqn, C2_eqn], [C1, C2])

@info "Solving Monolithic System..."
res = solve_monolithic!(sys, (BCs.C1, BCs.C2), config)

@printf("\nMonolithic Solve Results:\n")
@printf("Residual: %.2e\n", res)
@printf("Mean C1:  %.6f\n", mean(C1.values))
@printf("Mean C2:  %.6f\n", mean(C2.values))

# Volume-averaged results using built-in utility
C1_avg = volume_average(C1)
C2_avg = volume_average(C2)
@printf("Vol-avg C1: %.6f\n", C1_avg)
@printf("Vol-avg C2: %.6f\n", C2_avg)

# Save VTK output
writer = initialise_writer(VTK(), mesh_dev)
write_results(1, 1.0, mesh_dev, writer, (BCs.C1, BCs.C2), ("C1", C1), ("C2", C2))

@info "quadLaplacian Monolithic Example Completed!"
