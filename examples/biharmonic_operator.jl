using XCALibre
using LinearAlgebra
using Accessors
using Printf
using Statistics

# ==============================================================================
# Example: High-Order Biharmonic Problem with Extended Stencil
# ==============================================================================
# This script demonstrates the use of a 4th-order Biharmonic operator.
# PDE: Δ²ϕ = S
#
# Features:
# 1. Extended connectivity (Second-degree neighbours).
# 2. Implicit 4th order derivative implementation in XCALibre.

# 1. Setup Mesh
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "laplace_unit_5by5.unv"))

backend = CPU(); workgroup = 1024
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# 2. Setup Connectivity
# We use the new extended connectivity tool to build a wider stencil
i, j, v = extended_sparse_matrix_connectivity(mesh)
@info "Stencil size expanded from $(length(mesh.cell_neighbours)) to $(length(j)) entries."

# 3. Define Physics & BCs
# (Using Zerogradient for simplicity, but Biharmonic usually requires 2 BCs)
BCs = assign(
    region=mesh_dev,
    (
        phi = [Zerogradient(b.name) for b in mesh.boundaries],
    )
)

solvers = (
    phi = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
)

schemes = (
    phi = Schemes(laplacian=Linear), # Scheme for nested operators
)

config = Configuration(solvers=solvers, schemes=schemes, runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0), hardware=hardware, boundaries=BCs)

# 4. Build and Solve Biharmonic Equation
phi = ScalarField(mesh_dev); initialise!(phi, 1.0)
gamma = ConstantScalar(1.0) # Diffusion coefficient for biharmonic

# Construct the ScalarEquation manually to use the extended connectivity
# Note: ScalarEquation constructor now supports passing custom connectivity (A)
# but for this example, we'll just demonstrate the DSL capability.

phi_eqn = (
    Biharmonic{schemes.phi.laplacian}(gamma, phi) # <--- 4TH ORDER OPERATOR
    ==
    Source(ConstantScalar(10.0)) # Constant source
) → ScalarEquation(phi, BCs.phi)

# Initialise solver and preconditioner
@reset phi_eqn.preconditioner = set_preconditioner(solvers.phi.preconditioner, phi_eqn)
@reset phi_eqn.solver = XCALibre._workspace(solvers.phi.solver, XCALibre._b(phi_eqn))

@info "Solving Biharmonic Problem..."
res = solve_equation!(phi_eqn, phi, BCs.phi, solvers.phi, config)

@printf("Biharmonic Residual: %.2e, Mean phi: %.4f\n", res, mean(phi.values))

@info "Biharmonic example completed!"
