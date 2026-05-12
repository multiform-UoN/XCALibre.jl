# =============================================================================
# Standard Incompressible Stokes Solver — Monolithic 3-field (u, v, p)
# =============================================================================
#
# Governing equations:
#   - Momentum: ∇·(2μ ε(u)) - ∇p + f = 0
#   - Continuity: ∇·u = 0
#
# This example uses Rhie-Chow stabilization for collocated Finite Volume.
# =============================================================================

using XCALibre
using Accessors
using LinearAlgebra
using Printf
using Statistics

# ── Mesh ──────────────────────────────────────────────────────────────────────
grids_dir = pkgdir(XCALibre, "examples", "0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.025)
backend   = CPU(); workgroup = 1024
hardware  = Hardware(backend=backend, workgroup=workgroup)
mesh_dev  = adapt(backend, mesh)

# ── Material parameters (All set to 1.0) ───────────────────────────────────────
mu_val  = 1.0
force_x = 1.0
force_y = 1.0

@info "Standard Stokes: mu=$mu_val, force=($force_x, $force_y)"

# ── Fields ────────────────────────────────────────────────────────────────────
u = ScalarField(mesh_dev); initialise!(u, 0.0)
v = ScalarField(mesh_dev); initialise!(v, 0.0)
p = ScalarField(mesh_dev); initialise!(p, 0.0)

# ── Solver / config ───────────────────────────────────────────────────────────
solvers = (
    u = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    v = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    p = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
)
schemes = (u = Schemes(laplacian=Linear),
           v = Schemes(laplacian=Linear),
           p = Schemes(laplacian=Linear))
runtime = Runtime(iterations=1, write_interval=-1, time_step=1.0)

BCs = assign(
    region = mesh_dev,
    (
        u = [Dirichlet(:top, 1.0), Dirichlet(:bottom, 0.0), Dirichlet(:inlet, 0.0), Dirichlet(:outlet, 0.0)],
        v = [Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0), Dirichlet(:inlet, 0.0), Dirichlet(:outlet, 0.0)],
        p = [Zerogradient(:top), Zerogradient(:bottom), Zerogradient(:inlet), Zerogradient(:outlet)],
    )
)
config = Configuration(solvers=solvers, schemes=schemes,
                       runtime=runtime, hardware=hardware, boundaries=BCs)

# ── Constant flux scalars ─────────────────────────────────────────────────────
mu_cst    = ConstantScalar(mu_val)
one_cst   = ConstantScalar(1.0)
tau_rc    = ConstantScalar(0.1) # Rhie-Chow stabilization

# ── Monolithic equations ──────────────────────────────────────────────────────
L_u = ((
    - Laplacian{Linear}(mu_cst)               # Viscous stress
    + ScalarGrad{Linear,1}(one_cst, p)       # Pressure gradient coupling
    == Source(force_x)                        # Body force
) → BCs.u) → solvers.u

L_v = ((
    - Laplacian{Linear}(mu_cst)
    + ScalarGrad{Linear,2}(one_cst, p)
    == Source(force_y)
) → BCs.v) → solvers.v

L_p = ((
    - Laplacian{Linear}(tau_rc)               # Rhie-Chow term (stabilization)
    + VectorDiv{Linear,1}(one_cst, u)        # Div U components
    + VectorDiv{Linear,2}(one_cst, v)
    == Source(0.0)
) → BCs.p) → solvers.p

u_eqn = L_u(u); v_eqn = L_v(v); p_eqn = L_p(p)
sys = MonolithicSystem([u_eqn, v_eqn, p_eqn], [u, v, p])

# ── Solve ─────────────────────────────────────────────────────────────────────
@info "Solving Incompressible Stokes system..."
setReference!(p_eqn, 0.0, 1, config)
res = solve_monolithic!(sys, (BCs.u, BCs.v, BCs.p), config)
@info "Solve complete. Residual: $res"

# ── Post-process ──────────────────────────────────────────────────────────────
save_output(u, "u", 0.0, config)
save_output(v, "v", 0.0, config)
save_output(p, "p", 0.0, config)
@info "Standard Stokes results saved."
