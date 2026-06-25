using XCALibre
using LinearAlgebra
using Accessors
using Printf
using Statistics

# ==============================================================================
# Example: Cahn-Hilliard Phase-Field (Monolithic Block-Coupled Solver)
# ==============================================================================
# This script solves the Cahn-Hilliard equations in a monolithic block-coupled
# system.
#
# Governing Equations (Mixed Formulation):
# 1. ∂t ϕ - ∇ ⋅ (M(ϕ) ∇μ) = 0
# 2. μ = ψ'(ϕ) - κ ∇²ϕ
#
# Where ψ(ϕ) = (ϕ² - 1)² / 4 is the double-well potential.
#
# Linearization of ψ'(ϕ):
# ψ'(ϕ) = ϕ³ - ϕ
# Newton expansion: ψ'(ϕ) ≈ (3ϕ₀² - 1)ϕ - 2ϕ₀³
#
# Block System:
# [ 1/dt    -M∇² ] [ ϕ ]   [ ϕ_old/dt ]
# [ κ∇² - (3ϕ₀²-1)   1   ] [ μ ] = [ -2ϕ₀³ ]
# ==============================================================================

# 1. Setup Mesh
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.01)

backend = CPU(); workgroup = 1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

n_cells = length(mesh_dev.cells)

# 2. Phase-field parameters
kappa  = 1e-4  # interface thickness
M_mob  = 1e-3  # mobility
dt     = 0.01  # time step

# 3. BCs: no-flux for ϕ and μ
BCs = assign(
    (
        phi = [Zerogradient(:inlet), Zerogradient(:outlet), Zerogradient(:bottom), Zerogradient(:top)],
        mu  = [Zerogradient(:inlet), Zerogradient(:outlet), Zerogradient(:bottom), Zerogradient(:top)]
    ),
    region=mesh_dev
)

# 4. Fields initialization
phi = ScalarField(mesh_dev)
mu  = ScalarField(mesh_dev)
initialise!(mu, 0.0)

# Initialise ϕ: tanh interface in x-direction + noise
for cID in 1:n_cells
    x = mesh_dev.cells[cID].centre[1]
    L = maximum(mesh_dev.cells[cID2].centre[1] for cID2 in 1:n_cells)
    phi.values[cID] = tanh((x - L/2) / (2*sqrt(kappa))) + 0.01*randn()
end

phi_old = ScalarField(mesh_dev); phi_old.values .= phi.values

# 5. Iterative Simulation Loop
n_steps = 20
@info "Running monolithic Cahn-Hilliard for $n_steps steps..."

# Global settings for monolithic solve
schemes = (phi = Schemes(laplacian=Linear), mu = Schemes(laplacian=Linear))
# Use a stiff-aware solver setup
solvers = (phi = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0),)

for step in 1:n_steps
    # 5.1 Linearize ψ'(ϕ) ≈ (3ϕ₀² - 1)ϕ - 2ϕ₀³
    # We use CoupledSi to add -(3ϕ₀²-1) to the (2,1) block.
    # And Source(-2ϕ₀³) to the RHS of row 2.

    phi0 = phi.values
    k_lin = ScalarField(mesh_dev)
    k_lin.values .= -(3.0 .* phi0.^2 .- 1.0) # -ψ''(ϕ₀)

    s_lin = ScalarField(mesh_dev)
    s_lin.values .= -2.0 .* phi0.^3 # Explicit part

    phi_rhs = ScalarField(mesh_dev)
    phi_rhs.values .= phi_old.values ./ dt

    # 5.2 Define Monolithic System
    # Eq 1: ϕ/dt - M∇²μ = ϕ_old/dt
    eqn1 = (
        Si(ConstantScalar(1.0/dt), phi)
        - Laplacian{Linear}(ConstantScalar(M_mob), mu)
        ==
        Source(phi_rhs)
    ) → BCs.phi

    # Eq 2: μ + κ∇²ϕ - (3ϕ₀²-1)ϕ = -2ϕ₀³
    eqn2 = (
        Si(ConstantScalar(1.0), mu)
        + Laplacian{Linear}(ConstantScalar(kappa), phi)
        + CoupledSi(k_lin, phi)
        ==
        Source(s_lin)
    ) → BCs.mu

    # 5.3 Solve
    sys = MonolithicSystem([eqn1, eqn2], [phi, mu])

    # We must pass hardware and runtime inside config
    config = Configuration(solvers=solvers, schemes=schemes, runtime=Runtime(iterations=1, time_step=dt, write_interval=-1), hardware=hardware, boundaries=BCs)

    res = solve_monolithic!(sys, (BCs.phi, BCs.mu), config; use_preconditioner=true)

    phi_old.values .= phi.values

    if step % 5 == 0
        @printf("Step %3d: res=%.2e  mean(ϕ)=%.4f  min/max=%.2f/%.2f\n",
                step, res, mean(phi.values), minimum(phi.values), maximum(phi.values))
    end
end

# 6. Save Results
@info "Saving Results..."
ENV["XC_VTK_DIR"] = pwd()
save_vtk("cahn_hilliard", mesh_dev, BCs.phi, ("phi", phi), ("mu", mu))

@info "Cahn-Hilliard example completed!"
