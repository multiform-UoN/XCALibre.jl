# =============================================================================
# Compressible (Penalty) Stokes Benchmark — XCALibre.jl
# =============================================================================
#
# This example mimics the 'compressibleSolid' regime in OpenFOAM benchmarks.
# It solves the Stokes equations without an explicit pressure variable,
# instead enforcing near-incompressibility using a large penalty parameter (bulk viscosity).
#
# Equations:
#   - Momentum: -∇·(2μ ε(u)) - ∇(λ ∇·u) = f
#   where λ is the penalty parameter.
#
# Geometry: 2D Straight Channel (quad40.unv, 40x40 mesh, 1m x 1m)
# Parameters: mu = 1.0, lambda = 1e7, f = (1.0, 0.0)
# BCs: 
#   - Walls (top/bottom): u = (0,0) [No-slip]
#   - Inlet/Outlet: zeroGradient (Neumann) for u.
# =============================================================================

using XCALibre
using Accessors
using LinearAlgebra
using Printf
using Statistics

# ── 1. Mesh and Hardware ──────────────────────────────────────────────────────
grids_dir = pkgdir(XCALibre, "examples", "0_GRIDS")
mesh_path = joinpath(grids_dir, "quad40.unv")
mesh      = UNV2D_mesh(mesh_path, scale=0.025)

backend   = CPU(); workgroup = 1024
hardware  = Hardware(backend=backend, workgroup=workgroup)
mesh_dev  = adapt(backend, mesh)

# ── 2. Material and Forcing Parameters ────────────────────────────────────────
mu_val      = 1.0       # Viscosity (Unitary)
penalty_val = 1e7       # Penalty parameter (λ) for near-incompressibility
force_x     = 1.0       # Unitary x-direction forcing
force_y     = 0.0       # Zero y-direction forcing

@info "Penalty Stokes Benchmark: mu=$mu_val, penalty=$penalty_val, force=($force_x, $force_y)"

# ── 3. Fields ─────────────────────────────────────────────────────────────────
u = ScalarField(mesh_dev); initialise!(u, 0.0)
v = ScalarField(mesh_dev); initialise!(v, 0.0)

# ── 4. Boundary Conditions ───────────────────────────────────────────────────
BCs = assign(
    region = mesh_dev,
    (
        u = [Dirichlet(:top,    0.0),
             Dirichlet(:bottom, 0.0),
             Zerogradient(:inlet),
             Zerogradient(:outlet)],
        v = [Dirichlet(:top,    0.0),
             Dirichlet(:bottom, 0.0),
             Zerogradient(:inlet),
             Zerogradient(:outlet)],
    )
)

# ── 5. Solver and Numerical Schemes ───────────────────────────────────────────
solvers = (
    u = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    v = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
)
schemes = (u = Schemes(laplacian=Linear),
           v = Schemes(laplacian=Linear))
runtime = Runtime(iterations=1, write_interval=-1, time_step=1.0)

config = Configuration(solvers=solvers, schemes=schemes,
                       runtime=runtime, hardware=hardware, boundaries=BCs)

# ── 6. Monolithic Equation Definitions ────────────────────────────────────────
mu_cst     = ConstantScalar(mu_val)
lam_mu_cst = ConstantScalar(penalty_val + mu_val) # Combined bulk term

# Penalty-based Momentum (2-field system)
# -μ∇²u - (μ+λ) ∂(∇·u)/∂x = f_x
L_u = ((
    - Laplacian{Linear}(mu_cst)
    - GradDiv{Linear,1,1}(lam_mu_cst)       # Self-coupling component (u to u)
    - GradDiv{Linear,1,2}(lam_mu_cst, v)    # Cross-coupling component (v to u)
    == Source(force_x)
) → BCs.u) → solvers.u

L_v = ((
    - Laplacian{Linear}(mu_cst)
    - GradDiv{Linear,2,1}(lam_mu_cst, u)    # Cross-coupling component (u to v)
    - GradDiv{Linear,2,2}(lam_mu_cst)       # Self-coupling component (v to v)
    == Source(force_y)
) → BCs.v) → solvers.v

u_eqn = L_u(u); v_eqn = L_v(v)
sys = MonolithicSystem([u_eqn, v_eqn], [u, v])

# ── 7. Solve ──────────────────────────────────────────────────────────────────
@info "Solving Penalty Stokes system (Pressure-Free)..."
res = solve_monolithic!(sys, (BCs.u, BCs.v), config)
@info "Solve complete. Residual: $res"

# ── 8. Post-process and Results ───────────────────────────────────────────────
max_u = maximum(abs.(u.values))
@info "Peak Velocity max|u| = $max_u"

# Divergence check (measure of incompressibility)
∇u = Grad{Gauss}(u); uf = FaceScalarField(mesh_dev)
∇v = Grad{Gauss}(v); vf = FaceScalarField(mesh_dev)
grad!(∇u, uf, u, BCs.u, nothing, config)
grad!(∇v, vf, v, BCs.v, nothing, config)
divU = ∇u.result.x.values .+ ∇v.result.y.values
@info "Divergence Stats: mean|div U| = $(mean(abs.(divU)))"

save_output(u, "stokes_penalty_u", 0.0, config)
save_output(v, "stokes_penalty_v", 0.0, config)
@info "Benchmark stokes_penalty finished."
