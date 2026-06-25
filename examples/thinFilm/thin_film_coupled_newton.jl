using XCALibre
using LinearAlgebra
using Accessors
using Printf
using Statistics

# ==============================================================================
# Advanced Example: Coupled Monolithic Newton Thin-Film Solver
# ==============================================================================
# This script solves a coupled (h, p) system for thin-film flow:
#   1. Curvature: p + γ ∇²h = 0
#   2. Continuity: ∂h/∂t - ∇ ⋅ (M ∇p) + αh² = 0
#
# We solve this as a 2-variable Monolithic block system using Newton-Raphson.
# ==============================================================================

# 1. Setup Mesh
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.01)

backend = CPU(); workgroup=1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# 2. Parameters
gamma = 0.01
mu = 1.0
dt = 0.1

# 3. Setup Fields
h = ScalarField(mesh_dev); initialise!(h, 0.1)
p = ScalarField(mesh_dev); initialise!(p, 0.0)
h.values .+= 0.01 .* rand(length(h.values))

# 4. Setup Solver Configuration
BCs = assign(
    region=mesh_dev,
    (
        h = [Zerogradient(b.name) for b in mesh.boundaries],
        p = [Dirichlet(:inlet, 1.0), Zerogradient(:outlet), Zerogradient(:top), Zerogradient(:bottom)],
    )
)

schemes = (
    h = Schemes(time=Euler, laplacian=Linear),
    p = Schemes(laplacian=Linear),
)

config = Configuration(
    solvers=nothing,
    schemes=schemes,
    runtime=Runtime(iterations=20, time_step=dt, write_interval=-1),
    hardware=hardware,
    boundaries=BCs
)

@info "Starting Monolithic Newton Time Loop..."

for step in 1:5
    global h, p

    # ── Define Monolithic Equations (abstract PDE -> BCs -> bind field) ──────
    # Same PDE definition style can be reused with different BCs/fields.

    # Row 1 (p): p + γ ∇²h = 0   (cross term on h)
    pde_p = (
          Si(ConstantScalar(1.0))
        + Laplacian{Linear}(ConstantScalar(gamma), h)
        == Source(0.0)
    )
    L_p = pde_p → BCs.p

    # Row 2 (h): ∂h/∂t - ∇ ⋅ (M ∇p) + αh² = 0
    pde_h = (
          Time{Euler}()
        - Laplacian{Linear}(ConstantScalar(0.01), p)
        + NonLinearSi(val -> 0.1 * val^2)             # Non-linear self-field term
        == Source(0.0)
    )
    L_h = pde_h → BCs.h

    # ── Solve via Newton ──────────────────────────────────────────────────────
    p_eqn = L_p(p)
    h_eqn = L_h(h)

    sys = MonolithicSystem([p_eqn, h_eqn], [p, h])
    bcs_list = [BCs.p, BCs.h]

    res = newton_solve!(sys, bcs_list, config; tol=1e-8, verbose=true)

    @printf("Step %d: Newton iterations = %d, Final Res = %.2e, Mean h = %.4f\n",
            step, res.iterations, res.residuals[end], mean(h.values))
end

@info "Coupled Newton Example Completed!"
