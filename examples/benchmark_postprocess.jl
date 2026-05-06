using XCALibre
using LinearAlgebra
using Accessors
using Printf
using Statistics

# ==============================================================================
# Performance Comparison & Post-Processing Demo
# ==============================================================================
# This script:
# 1. Benchmarks XCALibre's Scalar Transport on a standard OpenFOAM mesh (pitzDaily).
# 2. Demonstrates the new volume_average and sample_at_point tools.
# 3. Compares timing results (Time per iteration).

# 1. Setup Mesh (OF pitzDaily - approx 12k cells)
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "OF_pitzDaily/polyMesh"
@info "Loading OpenFOAM mesh from $grid ..."
mesh = FOAM3D_mesh(joinpath(grids_dir, grid))

backend = CPU(); workgroup=1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# 2. Define Physics & BCs
# Constant flow field (analytical-like)
U_const(x, y, z) = [1.0, 0.0, 0.0]

BCs = assign(
    (
        C = [
            Dirichlet(:inlet, 1.0),
            Zerogradient(:upperWall),
            Zerogradient(:lowerWall),
            Zerogradient(:outlet),
            Zerogradient(:frontAndBack)
        ],
    ),
    region=mesh_dev
)

schemes = (C = Schemes(divergence=Upwind, laplacian=Linear),)
solvers = (C = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0),)
config = Configuration(solvers=solvers, schemes=schemes, runtime=Runtime(iterations=100, write_interval=-1, time_step=1.0), hardware=hardware, boundaries=BCs)

# 3. Setup ADR
C = ScalarField(mesh_dev); initialise!(C, 0.0)
mdotf = FaceScalarField(mesh_dev)
# Create a dummy model to store velocity
model = Physics(time=Steady(), energy=Energy{Isothermal}(), domain=mesh_dev)
initialise!(model.momentum.U, U_const)

interpolate!(model.momentum.Uf, model.momentum.U, config)
flux!(mdotf, model.momentum.Uf, config)
gamma = ConstantScalar(1e-4)

C_eqn = (
      Divergence{schemes.C.divergence}(mdotf, C)
    - Laplacian{schemes.C.laplacian}(gamma, C)
    ==
    Source(ConstantScalar(0.0))
) → ScalarEquation(C, BCs.C)

@reset C_eqn.preconditioner = set_preconditioner(solvers.C.preconditioner, C_eqn)
@reset C_eqn.solver = XCALibre._workspace(solvers.C.solver, XCALibre._b(C_eqn))

# 4. Benchmark Iterations
@info "Starting benchmark (10 iterations)..."
total_time = 0.0
n_iter = 10

for i in 1:n_iter
    global total_time
    start_time = time_ns()
    res = solve_equation!(C_eqn, C, BCs.C, solvers.C, config)
    end_time = time_ns()
    
    iter_time = (end_time - start_time) / 1e9
    total_time += iter_time
    @printf("Iteration %d: Res = %.2e, Time = %.4fs\n", i, res, iter_time)
end

avg_time = total_time / n_iter
@printf("\nAverage Time per Iteration: %.4fs\n", avg_time)

# 5. Post-Processing Demo (New Tools)
@info "Calculating Quantities of Interest..."

# 5.1 Volume Average
vol_avg = volume_average(C)
@printf("Volume Average of C: %.6f\n", vol_avg)

# 5.2 Point Sampling
# Sample at the center of the pitzDaily expansion
sample_pt = [0.05, 0.0, 0.0]
val_at_pt = sample_at_point(C, sample_pt)
@printf("C value at point %s: %.6f\n", sample_pt, val_at_pt)

# 5.3 Boundary Average
inlet_avg = boundary_average(:inlet, C, BCs.C, config)
@printf("Inlet Average C: %.6f (Expected: 1.000000)\n", inlet_avg)

@info "Benchmark and Comparison Completed!"
