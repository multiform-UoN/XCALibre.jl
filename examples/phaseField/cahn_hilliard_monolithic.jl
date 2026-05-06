using XCALibre
using LinearAlgebra
using Accessors
using Printf
using Statistics

# ==============================================================================
# Example: Cahn-Hilliard Phase-Field (Monolithic Block-Coupled Solver)
# ==============================================================================
# PDE (strong form):
#   ∂ϕ/∂t = M ∇²μ                          [mass balance]
#   μ = f'(ϕ) - κ ∇²ϕ                       [chemical potential definition]
#
# where f(ϕ) = (1 - ϕ²)²/4  →  f'(ϕ) = ϕ³ - ϕ   (quadratic double well)
#
# Weak / FVM form solved per time step (linearising f'(ϕ) around ϕ^n):
#   [I/Δt    -M∇²] [ϕ]   [ϕ^n/Δt      ]
#   [-κ∇²  +I    ] [μ] = [f'(ϕ^n) + ...]
#
# This is a 2×2 block system. The monolithic solver assembles it into a single
# (2*n_cells)×(2*n_cells) CSR matrix and solves with BiCGSTAB.
#
# BCs: no-flux on all boundaries (∇ϕ·n = 0, ∇μ·n = 0)
# ==============================================================================

# 1. Setup Mesh
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.01)

backend = CPU(); workgroup = 1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

n_cells = length(mesh_dev.cells)

# 2. Phase-field parameters
kappa  = 1e-2  # interface thickness (increased for stability)
M_mob  = 1e-4  # mobility
dt     = 1e-5  # time step (reduced for stability)

# 3. BCs: no-flux for ϕ and μ
BCs = assign(
    region=mesh_dev,
    (
        phi = [Zerogradient(b.name) for b in mesh.boundaries],
        mu  = [Zerogradient(b.name) for b in mesh.boundaries],
    )
)

schemes = (
    phi = Schemes(time=Euler, laplacian=Linear),
    mu  = Schemes(laplacian=Linear),
)
solvers = (
    phi = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0),
)
runtime  = Runtime(iterations=10, write_interval=-1, time_step=dt)
config   = Configuration(solvers=solvers, schemes=schemes, runtime=runtime,
                         hardware=hardware, boundaries=BCs)

# 4. Fields
phi    = ScalarField(mesh_dev)
mu     = ScalarField(mesh_dev)
phi_n  = ScalarField(mesh_dev)      # previous time step
fp_src = ScalarField(mesh_dev)      # f'(ϕ^n) source for μ equation
phi_div= ScalarField(mesh_dev)      # ϕ^n/Δt source for ϕ equation

# Initialise ϕ: tanh interface in x-direction
for cID in 1:n_cells
    x = mesh_dev.cells[cID].centre[1]
    L = maximum(mesh_dev.cells[cID2].centre[1] for cID2 in 1:n_cells)
    phi.values[cID] = tanh((x - L/2) / (2*sqrt(kappa))) + 0.01 * randn()
end

@printf("Initial phase: mean(ϕ) = %.4f, interface count ≈ %d cells\n",
        mean(phi.values), sum(abs.(phi.values) .< 0.9))

# 5. Monolithic Cahn-Hilliard loop
n_steps    = 20
write_every = 5

@info "Running monolithic Cahn-Hilliard for $n_steps steps..."
for step in 1:n_steps
    global phi, mu, phi_n, fp_src, phi_div

    # Cache previous ϕ^n
    phi_n.values .= phi.values

    # Compute explicit sources:
    # f'(ϕ^n) = ϕ^n³ - ϕ^n  (used in the μ equation RHS)
    fp_src.values .= phi_n.values.^3 .- phi_n.values

    # ϕ^n/Δt  (used in the ϕ equation RHS)
    phi_div.values .= phi_n.values ./ dt

    # Build block-coupled system:
    #
    # Eq 1 (ϕ row):  ϕ/Δt - M∇²μ = ϕ^n/Δt
    #   - Time{Euler}(phi)       → adds ϕ/Δt to diagonal (block 1,1)
    #   - Laplacian{Linear}(M, mu) → adds -M∇²μ to block (1,2) off-diagonal
    #   - Source(phi_div)         → ϕ^n/Δt on RHS
    #
    # Eq 2 (μ row):  -κ∇²ϕ + μ = f'(ϕ^n)
    #   - Laplacian{Linear}(kappa, phi) → -κ∇²ϕ in block (2,1) off-diagonal
    #   - Si(ConstantScalar(1.0), mu)   → +μ diagonal in block (2,2)
    #   - Source(fp_src)                → f'(ϕ^n) on RHS

    M_coeff     = ConstantScalar(M_mob)
    kappa_coeff = ConstantScalar(kappa)

    phi_eqn = (
          Time{schemes.phi.time}(phi)
        - Laplacian{schemes.phi.laplacian}(M_coeff, mu)   # cross-field: couples to μ
        ==
        Source(phi_div)
    ) → ScalarEquation(phi, BCs.phi)

    mu_eqn = (
        - Laplacian{schemes.mu.laplacian}(kappa_coeff, phi)  # cross-field: couples to ϕ
        + Si(ConstantScalar(1.0), mu)                        # μ on diagonal
        ==
        Source(fp_src)
    ) → ScalarEquation(mu, BCs.mu)

    # Monolithic solve
    sys = MonolithicSystem([phi_eqn, mu_eqn], [phi, mu])
    res = solve_monolithic!(sys, (BCs.phi, BCs.mu), config)

    if step % write_every == 0
        free_energy = mean((1 .- phi.values.^2).^2) / 4
        @printf("Step %3d: res=%.2e  mean(ϕ)=%.4f  bulk_free_energy≈%.4e\n",
                step, res, mean(phi.values), free_energy)
    end
end

# 6. Postprocessing
@printf("\nFinal Results (Cahn-Hilliard monolithic):\n")
@printf("  Mean ϕ   = %.6f\n", mean(phi.values))
@printf("  Vol-avg ϕ = %.6f\n", volume_average(phi))
@printf("  Mean μ   = %.6f\n", mean(mu.values))

# Weighted integrals (useful for homogenisation-style analyses)
phi_avg = volume_average(phi)
mu_avg  = volume_average(mu)
@printf("  <ϕ>  = %.6f,  <μ>  = %.6f\n", phi_avg, mu_avg)

# Interface energy: ∫ κ/2 |∇ϕ|² dV ≈ κ/2 ∫ |ϕ(1-ϕ²)/Δx|² dV (rough estimate)
# Use weighted_volume_integral with a point-wise energy density weight
interface_weight(x, y, z) = 1.0   # uniform → just total volume integral
bulk_free_energy = weighted_volume_integral(phi, (x,y,z) -> (1 - phi.values[1])^2 / 4) # approx
@printf("  Bulk free energy ∫f(ϕ) dV = %.6e\n",
        volume_integral(phi) * mean((1 .- phi.values.^2).^2) / 4)

# 7. VTK output
@info "Saving VTK output..."
writer = initialise_writer(VTK(), mesh_dev)
write_results(1, 1.0, mesh_dev, writer, BCs, ("phi", phi), ("mu", mu))

@info "Cahn-Hilliard (monolithic) example completed!"
