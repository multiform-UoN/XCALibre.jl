using XCALibre
using LinearAlgebra
using Accessors
using Printf
using Statistics

# ==============================================================================
# Example: Biharmonic Problem (Δ²ϕ = S) via operator splitting
# ==============================================================================
# PDE: Δ²ϕ = S  (S = 10)
# Simply-supported plate BCs: ϕ = 0 and Δϕ = 0 on all boundaries.
#
# Solved as a decoupled two-step system:
#   ψ := -Δϕ  (intermediate variable, ψ=0 on ∂Ω)
#   Step 1: -Δψ = S        (with ψ=0 on ∂Ω)
#   Step 2: -Δϕ = -ψ       (with ϕ=0 on ∂Ω)
#
# The two steps are decoupled (no outer iteration needed) because ψ and ϕ
# share the same zero Dirichlet BC on a simply-supported boundary.
#
# The Biharmonic{Linear} operator uses the same face-flux structure as
# Laplacian{Linear}. See phaseField/ folder for Cahn-Hilliard examples
# that use the full coupled formulation.
# ==============================================================================

# 1. Setup Mesh
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "laplace_unit_5by5.unv"))

backend = CPU(); workgroup = 1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# Extended stencil info (for reporting / future high-order assembly)
i_ext, j_ext, _ = extended_sparse_matrix_connectivity(mesh)
@info "Stencil: 1st-degree=$(length(mesh.cell_neighbours)), 2nd-degree=$(length(j_ext)) entries."

# 2. Define BCs (simply-supported plate: ϕ=0 and Δϕ=0 on ∂Ω)
BCs = assign(
    region=mesh_dev,
    (
        phi = [Dirichlet(b.name, 0.0) for b in mesh.boundaries],
        psi = [Dirichlet(b.name, 0.0) for b in mesh.boundaries],
    )
)

solvers = (
    phi = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    psi = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
)
schemes = (
    phi = Schemes(laplacian=Linear),
    psi = Schemes(laplacian=Linear),
)
runtime = Runtime(iterations=1, write_interval=-1, time_step=1.0)
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

# 3. Fields
phi = ScalarField(mesh_dev); initialise!(phi, 0.0)
psi = ScalarField(mesh_dev); initialise!(psi, 0.0)
S_val = 10.0
gamma = ConstantScalar(1.0)

# 4. Step 1: Solve -Δψ = S  (ψ = 0 on ∂Ω)
@info "Solving Step 1: -Δψ = S..."
psi_eqn = (
    - Biharmonic{schemes.psi.laplacian}(gamma, psi)
    ==
    Source(ConstantScalar(S_val))
) → ScalarEquation(psi, BCs.psi)
@reset psi_eqn.preconditioner = set_preconditioner(solvers.psi.preconditioner, psi_eqn)
@reset psi_eqn.solver = XCALibre._workspace(solvers.psi.solver, XCALibre._b(psi_eqn))
res_psi = solve_equation!(psi_eqn, psi, BCs.psi, solvers.psi, config)
@printf("Step 1 Residual: %.2e, Mean ψ: %.6f\n", res_psi, mean(psi.values))

# 5. Step 2: Solve -Δϕ = -ψ  (ϕ = 0 on ∂Ω)
@info "Solving Step 2: -Δϕ = -ψ..."
psi_as_source = ScalarField(mesh_dev)
psi_as_source.values .= psi.values  # ψ = -Δϕ → -Δϕ = ψ
phi_eqn = (
    - Biharmonic{schemes.phi.laplacian}(gamma, phi)
    ==
    Source(psi_as_source)
) → ScalarEquation(phi, BCs.phi)
@reset phi_eqn.preconditioner = set_preconditioner(solvers.phi.preconditioner, phi_eqn)
@reset phi_eqn.solver = XCALibre._workspace(solvers.phi.solver, XCALibre._b(phi_eqn))
res_phi = solve_equation!(phi_eqn, phi, BCs.phi, solvers.phi, config)
@printf("Step 2 Residual: %.2e, Mean ϕ: %.6f\n", res_phi, mean(phi.values))

# 6. Postprocessing
@printf("\nBiharmonic Results (Δ²ϕ = %g):\n", S_val)
@printf("  Mean ψ = %.6f  (analytical: S*L²/8 ≈ %.4f for unit square)\n",
        mean(psi.values), S_val / 8.0)
@printf("  Mean ϕ = %.6f  (analytical: S*L⁴/64 ≈ %.4f for unit square)\n",
        mean(phi.values), S_val / 64.0)
@printf("  Volume-averaged ϕ = %.6f\n", volume_average(phi))

# 7. VTK output
writer = initialise_writer(VTK(), mesh_dev)
write_results(1, 1.0, mesh_dev, writer, BCs, ("phi", phi), ("psi", psi))

@info "Biharmonic example completed!"
