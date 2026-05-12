# =============================================================================
# Coupled Stokes Solver — Monolithic 3-field (u, v, p)
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

# ── Material parameters ───────────────────────────────────────────────────────
mu_val  = 1.0

# ── Fields ────────────────────────────────────────────────────────────────────
u = ScalarField(mesh_dev); initialise!(u, 0.0)
v = ScalarField(mesh_dev); initialise!(v, 0.0)
p = ScalarField(mesh_dev); initialise!(p, 0.0)

# ── Solver / config ───────────────────────────────────────────────────────────
solvers = (
    u = SolverSetup(solver=Gmres(), preconditioner=Jacobi(),
                    convergence=1e-10, relax=1.0),
    v = SolverSetup(solver=Gmres(), preconditioner=Jacobi(),
                    convergence=1e-10, relax=1.0),
    p = SolverSetup(solver=Gmres(), preconditioner=Jacobi(),
                    convergence=1e-10, relax=1.0),
)
schemes = (u = Schemes(laplacian=Linear),
           v = Schemes(laplacian=Linear),
           p = Schemes(laplacian=Linear))
runtime = Runtime(iterations=1, write_interval=-1, time_step=1.0)

BCs = assign(
    region = mesh_dev,
    (
        u = [Dirichlet(:top,    1.0),
             Dirichlet(:bottom, 0.0),
             Dirichlet(:inlet,  0.0),
             Dirichlet(:outlet, 0.0)],
        v = [Dirichlet(:top,    0.0),
             Dirichlet(:bottom, 0.0),
             Dirichlet(:inlet,  0.0),
             Dirichlet(:outlet, 0.0)],
        p = [Zerogradient(:top),
             Zerogradient(:bottom),
             Zerogradient(:inlet),
             Zerogradient(:outlet)],
    )
)
config = Configuration(solvers=solvers, schemes=schemes,
                       runtime=runtime, hardware=hardware, boundaries=BCs)

# ── Constant flux scalars ─────────────────────────────────────────────────────
mu_cst    = ConstantScalar(mu_val)
one_cst   = ConstantScalar(1.0)

# Rhie-Chow stabilization coefficient
tau_rc_val = 0.1
tau_rc_cst = ConstantScalar(tau_rc_val)

# ── Monolithic equations ──────────────────────────────────────────────────────
L_u = ((
    - Laplacian{Linear}(mu_cst)
    + ScalarGrad{Linear,1}(one_cst, p)
    == Source(0.0)
) → BCs.u) → solvers.u

L_v = ((
    - Laplacian{Linear}(mu_cst)
    + ScalarGrad{Linear,2}(one_cst, p)
    == Source(0.0)
) → BCs.v) → solvers.v

L_p = ((
    - Laplacian{Linear}(tau_rc_cst)      # Rhie-Chow stabilization (acts on p)
    + VectorDiv{Linear,1}(one_cst, u)
    + VectorDiv{Linear,2}(one_cst, v)
    == Source(0.0)
) → BCs.p) → solvers.p

u_eqn = L_u(u)
v_eqn = L_v(v)
p_eqn = L_p(p)

sys = MonolithicSystem([u_eqn, v_eqn, p_eqn], [u, v, p])

# ── Solve ─────────────────────────────────────────────────────────────────────
@info "Solving Coupled Stokes system (No Rhie-Chow)..."
# Setting a reference pressure since all BCs are Zerogradient
setReference!(p_eqn, 0.0, 1, config)
res = solve_monolithic!(sys, (BCs.u, BCs.v, BCs.p), config)
@info "Solve complete. Residual: $res"

# Check for checkerboarding (pressure oscillations)
p_vals = p.values
p_grad_norm = mean(abs.(p_vals[2:end] .- p_vals[1:end-1]))
@info "Pressure gradient norm (indicator of oscillations): $p_grad_norm"

max_u = maximum(abs.(u.values))
@info "Flow magnitude: max|u| = $max_u"
