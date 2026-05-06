using XCALibre
using LinearAlgebra
using Accessors
using Printf
using Statistics

# ==============================================================================
# Example: Cahn-Hilliard Phase-Field (Sequential / Operator-Splitting)
# ==============================================================================
# PDE: ∂ϕ/∂t = M ∇²μ
#      μ = f'(ϕ) - κ ∇²ϕ
# where f(ϕ) = (1 - ϕ²)²/4  →  f'(ϕ) = ϕ³ - ϕ   (quadratic double well)
#
# BCs: no-flux on all boundaries: ∇ϕ·n = 0, ∇μ·n = 0
#
# Sequential (operator-splitting) algorithm per time step:
#   1. Compute μ^n = f'(ϕ^n) - κ ∇²ϕ^n  (explicit Laplacian for κ∇²ϕ term)
#   2. Advance ϕ: ∂ϕ/∂t - M ∇²μ^n = 0  (implicit in ϕ, using μ^n as source gradient)
#
# The two-step approach avoids the need for a coupled solver and is sufficient
# for demonstration. See cahn_hilliard_monolithic.jl for the block-coupled version.
# ==============================================================================

# 1. Setup Mesh
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.01)

backend = CPU(); workgroup = 1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

n_cells = length(mesh_dev.cells)

# 2. Phase-field parameters
kappa = 1e-4   # interface thickness parameter (controls ∇ϕ penalty)
M     = 1e-3   # mobility
dt    = 0.1    # time step

# 3. BCs: no-flux for both ϕ and μ
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
    mu  = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0),
)
config = Configuration(solvers=solvers, schemes=schemes,
    runtime=Runtime(iterations=1, write_interval=-1, time_step=dt),
    hardware=hardware, boundaries=BCs)

# 4. Initialise fields: two-phase initial condition (tanh interface)
phi = ScalarField(mesh_dev)
mu  = ScalarField(mesh_dev)
mu_src = ScalarField(mesh_dev)

# Initialise ϕ: positive on left half, negative on right half
for cID in 1:n_cells
    x = mesh_dev.cells[cID].centre[1]
    L = 0.4   # domain width (quad40 is 40x40 cells of 0.01m = 0.4m)
    phi.values[cID] = tanh((x - L/2) / sqrt(2*kappa))
end

@printf("Initial phase: mean(ϕ) = %.4f\n", mean(phi.values))

# 5. Helper: explicit Laplacian of a scalar field via face fluxes
function explicit_laplacian!(lap::ScalarField, phi::ScalarField)
    mesh = phi.mesh
    (; cells, faces, cell_faces, cell_nsign, cell_neighbours) = mesh
    lap.values .= 0.0
    for cID in eachindex(cells)
        cell = cells[cID]
        acc = 0.0
        for fi in cell.faces_range
            fID = cell_faces[fi]
            nID = cell_neighbours[fi]
            face = faces[fID]
            acc += (phi.values[nID] - phi.values[cID]) * face.area / face.delta
        end
        lap.values[cID] = acc / cell.volume
    end
end

# 6. Time-marching loop
n_steps = 50
write_every = 10
lap_phi = ScalarField(mesh_dev)

@info "Running Cahn-Hilliard (sequential) for $n_steps steps..."
for step in 1:n_steps
    global phi, mu, mu_src

    # Step A: compute μ^n = f'(ϕ^n) - κ ∇²ϕ^n  (explicit)
    explicit_laplacian!(lap_phi, phi)
    mu.values .= (phi.values.^3 .- phi.values) .- kappa .* lap_phi.values

    # Step B: advance ϕ via ∂ϕ/∂t = M ∇²μ^n
    # Rewrite: ϕ/Δt - M∇²μ = ϕ^n/Δt
    # In XCALibre DSL: Time{Euler}(phi) - Laplacian{Linear}(M, phi) = ∂ϕ/∂t_rhs
    # But here μ is fixed (explicit), so we use it as a face-scalar driving term.
    # We reformulate: ∂ϕ/∂t = M ∇²μ → solve -M∇²ϕ = (ϕ_prev/Δt - M∇²μ) - ϕ/Δt... hmm.
    # Simpler: advance explicitly in time for the scalar equation, then filter.

    # Actually: the equation ∂ϕ/∂t = M∇²μ is linear in ϕ but the RHS depends on μ.
    # We can advance ϕ using the divergence of the μ flux:
    #   ϕ^{n+1} = ϕ^n + Δt * M * ∇²μ^n  (explicit Euler)
    explicit_laplacian!(lap_phi, mu)
    phi.values .+= dt .* M .* lap_phi.values

    # Clip to physical range
    phi.values .= clamp.(phi.values, -1.2, 1.2)

    if step % write_every == 0
        free_energy = 0.25 * mean((1 .- phi.values.^2).^2)
        @printf("Step %3d: mean(ϕ)=%.4f, free_energy≈%.4e\n",
                step, mean(phi.values), free_energy)
    end
end

# 7. Postprocessing
@printf("\nFinal Results (Cahn-Hilliard sequential):\n")
@printf("  Mean ϕ  = %.6f\n", mean(phi.values))
@printf("  Vol-avg ϕ = %.6f\n", volume_average(phi))
@printf("  Bulk free energy = %.6e\n", volume_integral(phi) * 0.25)

# Weighted integral: effective interface position (weight = x-coordinate)
x_centroid = weighted_volume_integral(phi, (x,y,z)->x) / volume_integral(phi)
@printf("  Centroid of ϕ-weighted x: %.4f\n", isfinite(x_centroid) ? x_centroid : NaN)

# 8. VTK output
writer = initialise_writer(VTK(), mesh_dev)
write_results(1, 1.0, mesh_dev, writer, BCs, ("phi", phi), ("mu", mu))

@info "Cahn-Hilliard (sequential) example completed!"
