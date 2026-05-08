# =============================================================================
# Biot Consolidation — FULLY COUPLED monolithic 3-field solver
# =============================================================================
#
# Governing equations (quasi-static Biot, plane-strain, backward Euler):
#
#   Momentum (i = x, y):
#       μ ∇²u_i + (μ+λ) ∂(∇·u)/∂x_i - α ∂p/∂x_i = 0
#
#   Flow:
#       Sε ∂p/∂t + α ∂(∇·u)/∂t - k ∇²p = 0
#       Sε = 1/M + α²/(λ+2μ)
#
# Monolithic block system (backward Euler, implicit in all fields):
#
#   [ K_uu    K_uv    B^T_x ] { u^{n+1} }   { 0                       }
#   [ K_vu    K_vv    B^T_y ] { v^{n+1} } = { 0                       }
#   [ B_x     B_y     S/dt+H] { p^{n+1} }   { Sε/dt·p^n + α/dt·∇·u^n }
#
# where:
#   K block   = Laplacian(μ) + GradDiv(μ+λ)      [elastic stiffness]
#   B^T_I     = ScalarGrad{Linear,I}(α, p)        [pressure → momentum coupling]
#   B_J       = VectorDiv{Linear,J}(α/dt, u_J)    [displacement → pressure coupling]
#   S/dt+H    = Time{Euler}(Sε) - Laplacian(k)    [storage + Darcy flow]
#
# The RHS coupling term α/dt·∇·u^n is computed via grad! before each solve.
#
# ─────────────────────────────────────────────────────────────────────────────
# General coupling operators (Discretise_1_schemes.jl):
#
#   ScalarGrad{T,I}(flux, phi):  ∂φ/∂x_I  for any scalar field φ
#     face coeff = flux * face.e[I] * face.area / face.delta
#     → routes φ-column into the equation row that contains this term
#
#   VectorDiv{T,J}(flux, phi):  ∂u_J/∂x_J  for any vector component u_J
#     face coeff = flux * face.e[J] * face.area / face.delta
#     → routes u_J-column into the equation row that contains this term
#
# ─────────────────────────────────────────────────────────────────────────────
# Problem: 1-D Terzaghi consolidation (same as fixed-stress example)
# Mesh:    quad40.unv scaled to 10 m × 10 m
# BCs:     inlet (x=0): u=0, v=0, p=0 (fixed + drained)
#          outlet (x=L): Zerogradient u,v, p (free + undrained)
#          top/bottom:  Zerogradient u, v=0, Zerogradient p
# IC:      p = p₀, u = v = 0
# =============================================================================

using XCALibre
using Accessors
using LinearAlgebra
using Printf
using Statistics

# ── Mesh ──────────────────────────────────────────────────────────────────────
grids_dir = pkgdir(XCALibre, "examples", "0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.01)   # 10 m × 10 m
backend   = CPU(); workgroup = 1024
hardware  = Hardware(backend=backend, workgroup=workgroup)
mesh_dev  = adapt(backend, mesh)
n_cells   = length(mesh_dev.cells)
L_domain  = 10.0

# ── Material parameters ───────────────────────────────────────────────────────
E       = 1e4
nu      = 0.3
mu_val  = E / (2*(1 + nu))
lam_val = E*nu / ((1 + nu)*(1 - 2*nu))
alpha   = 1.0
M_biot  = 1e4
k_perm  = 1e-4

M_oedo = lam_val + 2*mu_val
Se     = 1.0/M_biot + alpha^2/M_oedo
c_v    = k_perm / Se
p0     = 1e3

@info "Monolithic Biot: E=$E, ν=$nu, α=$alpha, c_v=$(round(c_v,sigdigits=3)) m²/s"

# ── Time stepping ─────────────────────────────────────────────────────────────
T_end   = 1.0 * L_domain^2 / c_v
n_steps = 200
dt_val  = T_end / n_steps

# ── Fields ────────────────────────────────────────────────────────────────────
u = ScalarField(mesh_dev); initialise!(u, 0.0)
v = ScalarField(mesh_dev); initialise!(v, 0.0)
p = ScalarField(mesh_dev); initialise!(p, p0)

# Source field: α/dt · ∇·u^n, updated before each monolithic solve
divu_src = ScalarField(mesh_dev); initialise!(divu_src, 0.0)

# ── Gradient helpers for ∇·u^n ────────────────────────────────────────────────
∇u = Grad{Gauss}(u); uf = FaceScalarField(mesh_dev)
∇v_fld = Grad{Gauss}(v); vf = FaceScalarField(mesh_dev)

# ── Solver / config ───────────────────────────────────────────────────────────
solvers = (
    u = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                    convergence=1e-10, relax=1.0),
    v = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                    convergence=1e-10, relax=1.0),
    p = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                    convergence=1e-10, relax=1.0),
)
schemes = (u = Schemes(laplacian=Linear),
           v = Schemes(laplacian=Linear),
           p = Schemes(laplacian=Linear))
runtime = Runtime(iterations=1, write_interval=-1, time_step=dt_val)

BCs = assign(
    region = mesh_dev,
    (
        u = [Dirichlet(:inlet,  0.0),
             Zerogradient(:outlet),
             Zerogradient(:bottom),
             Zerogradient(:top)],
        v = [Dirichlet(:inlet,  0.0),
             Dirichlet(:outlet, 0.0),
             Dirichlet(:bottom, 0.0),
             Dirichlet(:top,    0.0)],
        p = [Dirichlet(:inlet,  0.0),
             Zerogradient(:outlet),
             Zerogradient(:bottom),
             Zerogradient(:top)],
    )
)
config = Configuration(solvers=solvers, schemes=schemes,
                       runtime=runtime, hardware=hardware, boundaries=BCs)

# ── Constant flux scalars ─────────────────────────────────────────────────────
mu_cst    = ConstantScalar(mu_val)
lam_mu    = ConstantScalar(lam_val + mu_val)    # μ+λ for GradDiv
alpha_cst = ConstantScalar(alpha)               # Biot α for ScalarGrad
k_cst     = ConstantScalar(k_perm)
Se_cst    = ConstantScalar(Se)
# VectorDiv flux = α/dt (updated each step if dt changes; constant here)
alpha_dt_cst = ConstantScalar(alpha / dt_val)

# ── Monolithic equations (PDEOperator DSL) ───────────────────────────────────
#
# Sign conventions:
#   Biot momentum: -μ∇²u_i - (μ+λ)∂(∇·u)/∂x_i + α ∂p/∂x_i = 0
#     → -Laplacian, -GradDiv, +ScalarGrad
#   Biot flow:     Sε/dt p + α/dt ∇·u^{n+1} - k∇²p = Sε/dt p^n + α/dt ∇·u^n
#     → +Time, +VectorDiv, -Laplacian
#
# Self-field terms (Laplacian, GradDiv diagonal) are OperatorTemplates —
# they bind to the equation's own field when L(field) is called.
# Cross-field terms (ScalarGrad, VectorDiv, GradDiv off-diagonal) carry their
# target field explicitly and are kept as pre-bound Operators unchanged.

L_u = (
    - Laplacian{Linear}(mu_cst)               # -μ∇²u  (binds to u)
    - GradDiv{Linear,1,1}(lam_mu)             # -(μ+λ)∂²u/∂x²  (binds to u)
    - GradDiv{Linear,1,2}(lam_mu, v)          # -(μ+λ)∂²v/∂x∂y  (v pre-bound)
    + ScalarGrad{Linear,1}(alpha_cst, p)      # +α∂p/∂x  (p pre-bound)
    == Source(0.0)
) → BCs.u → solvers.u

L_v = (
    - Laplacian{Linear}(mu_cst)               # -μ∇²v  (binds to v)
    - GradDiv{Linear,2,1}(lam_mu, u)          # -(μ+λ)∂²u/∂y∂x  (u pre-bound)
    - GradDiv{Linear,2,2}(lam_mu)             # -(μ+λ)∂²v/∂y²  (binds to v)
    + ScalarGrad{Linear,2}(alpha_cst, p)      # +α∂p/∂y  (p pre-bound)
    == Source(0.0)
) → BCs.v → solvers.v

L_p = (
    Time{Euler}(Se_cst)                       # Sε/dt p  (binds to p)
    - Laplacian{Linear}(k_cst)                # -k∇²p  (binds to p)
    + VectorDiv{Linear,1}(alpha_dt_cst, u)   # +α/dt ∂u/∂x  (u pre-bound)
    + VectorDiv{Linear,2}(alpha_dt_cst, v)   # +α/dt ∂v/∂y  (v pre-bound)
    == Source(divu_src)                       # +α/dt ∇·u^n on RHS (updated each step)
) → BCs.p → solvers.p

u_eqn = L_u(u)
v_eqn = L_v(v)
p_eqn = L_p(p)

# ── Monolithic 3-field system ─────────────────────────────────────────────────
# Field order determines block layout: [u | v | p]
sys = MonolithicSystem([u_eqn, v_eqn, p_eqn], [u, v, p])

# ── Time loop ─────────────────────────────────────────────────────────────────
@info "Starting monolithic Biot consolidation..."
@printf("  step    t[s]      T_v     <p>[Pa]  U[%%]   mono_res\n")

p_mean_init = mean(p.values)

for step in 1:n_steps
    t   = step * dt_val
    T_v = c_v * t / L_domain^2

    # Compute α/dt · ∇·u^n  (RHS coupling — must be done BEFORE the solve)
    grad!(∇u,    uf, u, BCs.u, nothing, config)
    grad!(∇v_fld, vf, v, BCs.v, nothing, config)
    @. divu_src.values = alpha / dt_val * (∇u.result.x.values + ∇v_fld.result.y.values)

    # Solve the monolithic 3×3 block system (u, v, p simultaneously)
    res = solve_monolithic!(sys, (BCs.u, BCs.v, BCs.p), config)

    if step % 20 == 0 || step == 1
        p_mean = mean(p.values)
        U_pct  = 100 * (1 - p_mean / p_mean_init)
        @printf("  %4d  %8.2f  %6.3f  %9.1f  %5.1f%%  %.2e\n",
                step, t, T_v, p_mean, U_pct, res)
    end
end

# ── Analytical solution ────────────────────────────────────────────────────────
function terzaghi_p(x, t, p0, L, cv; N=50)
    s = 0.0
    for n in 0:N
        m = 2n + 1
        s += (-1)^n / m * cos(m*π*x/(2L)) * exp(-m^2*π^2*cv*t/(4L^2))
    end
    return 4*p0/π * s
end

xs    = [mesh_dev.cells[i].centre[1] for i in 1:n_cells]
t_end = n_steps * dt_val
T_v_end = c_v * t_end / L_domain^2

p_anal  = [terzaghi_p(x, t_end, p0, L_domain, c_v) for x in xs]
p_err   = maximum(abs.(p.values .- p_anal))

println()
println("Biot Consolidation — MONOLITHIC 3-field solver (Terzaghi 1-D benchmark)")
println("  T_v = $(round(T_v_end, digits=3))")
@printf("  Max |p_FVM - p_analytical| = %.2e Pa  (p₀ = %.0f Pa)\n", p_err, p0)
@printf("  Relative L∞ error           = %.2e\n", p_err / p0)
@printf("  Degree of consolidation     = %.1f%%\n",
        100*(1 - mean(p.values)/p0))
println()
@info "Monolithic Biot consolidation complete."

# =============================================================================
# HOW THE COUPLING WORKS
#
# ScalarGrad{Linear,I}(α, p) — I-th gradient component of scalar φ:
#   term.phi = p → monolithic assembler routes to the p column block
#   face coefficient = α * face.e[I] * area / delta  ≈ ∂p/∂x_I
#
# VectorDiv{Linear,J}(α/dt, u_J) — J-th component of ∇·u:
#   term.phi = u_J → monolithic assembler routes to the u_J column block
#   face coefficient = α/dt * face.e[J] * area / delta  ≈ ∂u_J/∂x_J
#
# Routing: `field_to_idx[objectid(term.phi.values)]` maps each operator's
# phi to a block column.  MonolithicSystem handles N fields automatically.
# =============================================================================
