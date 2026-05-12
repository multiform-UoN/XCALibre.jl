# =============================================================================
# Penalty Stokes Solver — Monolithic 2-field (u, v)
# =============================================================================
#
# Governing equations:
#   - μ ∇²u - (μ+λ) ∂(∇·u)/∂x = f_x
#   - μ ∇²v - (μ+λ) ∂(∇·u)/∂y = f_y
#
# where λ is a large penalty parameter approximating incompressibility.
#
# =============================================================================

using XCALibre
using Accessors
using LinearAlgebra
using Printf
using Statistics

# ── Mesh ──────────────────────────────────────────────────────────────────────
# Using a simple 20x20 cavity (quad40.unv is 40x40)
grids_dir = pkgdir(XCALibre, "examples", "0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.025) # 1m x 1m cavity
backend   = CPU(); workgroup = 1024
hardware  = Hardware(backend=backend, workgroup=workgroup)
mesh_dev  = adapt(backend, mesh)
n_cells   = length(mesh_dev.cells)

# ── Material parameters ───────────────────────────────────────────────────────
mu_val  = 1.0
lambda_val = 1e7 * mu_val  # Penalty parameter
lam_mu_val = lambda_val + mu_val

@info "Penalty Stokes: mu=$mu_val, lambda=$lambda_val (Penalty)"

# ── Fields ────────────────────────────────────────────────────────────────────
u = ScalarField(mesh_dev); initialise!(u, 0.0)
v = ScalarField(mesh_dev); initialise!(v, 0.0)

# ── Solver / config ───────────────────────────────────────────────────────────
solvers = (
    u = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                    convergence=1e-10, relax=1.0),
    v = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                    convergence=1e-10, relax=1.0),
)
schemes = (u = Schemes(laplacian=Linear),
           v = Schemes(laplacian=Linear))
runtime = Runtime(iterations=1, write_interval=-1, time_step=1.0)

# BCs: Lid-driven cavity
# Top: u=1, v=0
# Others: u=0, v=0
BCs = assign(
    region = mesh_dev,
    (
        u = [Dirichlet(:top,    1.0),
             Dirichlet(:bottom, 0.0),
             Dirichlet(:inlet,  0.0), # inlet/outlet are left/right in quad40.unv
             Dirichlet(:outlet, 0.0)],
        v = [Dirichlet(:top,    0.0),
             Dirichlet(:bottom, 0.0),
             Dirichlet(:inlet,  0.0),
             Dirichlet(:outlet, 0.0)],
    )
)
config = Configuration(solvers=solvers, schemes=schemes,
                       runtime=runtime, hardware=hardware, boundaries=BCs)

# ── Constant flux scalars ─────────────────────────────────────────────────────
mu_cst     = ConstantScalar(mu_val)
lam_mu_cst = ConstantScalar(lam_mu_val)

# ── Monolithic equations ──────────────────────────────────────────────────────
L_u = ((
    - Laplacian{Linear}(mu_cst)
    - GradDiv{Linear,1,1}(lam_mu_cst)
    - GradDiv{Linear,1,2}(lam_mu_cst, v)
    == Source(0.0)
) → BCs.u) → solvers.u

L_v = ((
    - Laplacian{Linear}(mu_cst)
    - GradDiv{Linear,2,1}(lam_mu_cst, u)
    - GradDiv{Linear,2,2}(lam_mu_cst)
    == Source(0.0)
) → BCs.v) → solvers.v

u_eqn = L_u(u)
v_eqn = L_v(v)

sys = MonolithicSystem([u_eqn, v_eqn], [u, v])

# ── Solve ─────────────────────────────────────────────────────────────────────
@info "Solving Penalty Stokes system..."
res = solve_monolithic!(sys, (BCs.u, BCs.v), config)
@info "Solve complete. Residual: $res"

# ── Post-process: Divergence ──────────────────────────────────────────────────
∇u = Grad{Gauss}(u); uf = FaceScalarField(mesh_dev)
∇v = Grad{Gauss}(v); vf = FaceScalarField(mesh_dev)
grad!(∇u, uf, u, BCs.u, nothing, config)
grad!(∇v, vf, v, BCs.v, nothing, config)

divU = ∇u.result.x.values .+ ∇v.result.y.values
max_div = maximum(abs.(divU))
mean_div = mean(abs.(divU))

@info "Divergence Stats:"
@info "  Max |div U|: $max_div"
@info "  Mean |div U|: $mean_div"

# Save result
save_output(u, "u", 0.0, config)
save_output(v, "v", 0.0, config)

# Check for locking (velocity should not be zero everywhere)
max_u = maximum(abs.(u.values))
if max_u < 1e-3
    @warn "POTENTIAL LOCKING DETECTED: max|u| = $max_u"
else
    @info "Flow detected: max|u| = $max_u"
end
