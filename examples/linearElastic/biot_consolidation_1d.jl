# =============================================================================
# Biot Consolidation — 1-D Terzaghi Problem (fixed-stress sequential split)
# =============================================================================
#
# Governing equations (quasi-static Biot, plane-strain, effectively 1-D in x):
#
#   Mechanical (quasi-static):
#       μ ∇²u + (μ+λ) ∇(∇·u) = α ∇p
#
#   Flow (transient):
#       Sε ∂p/∂t - k ∇²p = -α ∂(∇·u)/∂t
#       Sε = 1/M + α²/(λ+2μ)    (constrained specific storage)
#
# ─────────────────────────────────────────────────────────────────────────────
# Fixed-stress sequential split (unconditionally stable):
#
#   Each time step n → n+1:
#     1. Mechanical (monolithic u,v given p^n):
#          [K  G] {u}   {α ∂p^n/∂x}
#          [Gᵀ K] {v} = {α ∂p^n/∂y}
#        → solve MonolithicSystem([u_eqn, v_eqn])
#          (body force source = α∇p^n updated via grad! before each solve)
#
#     2. Flow (scalar p given u^{n+1}, u^n):
#          Sε ∂p/∂t - k ∇²p = -α (∇·u^{n+1} - ∇·u^n) / Δt
#        → Time{Euler} + Laplacian + explicit volumetric-strain-rate source
#
# ─────────────────────────────────────────────────────────────────────────────
# Mesh:   quad40.unv (40×40), scaled to 10 m × 10 m  (dx = 0.25 m)
# BCs:    inlet (x=0): u=0, v=0 (fixed), p=0 (drained)
#         outlet (x=L): Zerogradient u, v=0 (roller), Zerogradient p (undrained)
#         top/bottom: Zerogradient u, v=0 (rollers), Zerogradient p
# IC:     p(x,0) = p₀  (uniform excess pressure),  u=v=0
#
# Analytical solution (Terzaghi 1923):
#   p(x,t) = (4p₀/π) Σ_{n=0}^∞  [(-1)^n / (2n+1)]
#              × cos[(2n+1)πx/(2L)]
#              × exp[-(2n+1)²π²c_v t / (4L²)]
#   where c_v = k / Sε
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

L_domain = 10.0     # consolidation length [m] (x-direction)

# ── Material parameters ───────────────────────────────────────────────────────
E       = 1e4          # Young's modulus [Pa]
nu      = 0.3          # Poisson's ratio
mu_val  = E / (2*(1 + nu))
lam_val = E*nu / ((1 + nu)*(1 - 2*nu))
alpha   = 1.0          # Biot coefficient
M_biot  = 1e4          # Biot modulus [Pa]
k_perm  = 1e-4         # permeability × unit weight [m/s]

M_oedo  = lam_val + 2*mu_val
Se      = 1.0/M_biot + alpha^2/M_oedo    # constrained specific storage
c_v     = k_perm / Se                    # consolidation coefficient

p0 = 1e3    # initial excess pore pressure [Pa]

@info "Biot: E=$(E) Pa, ν=$(nu), α=$(alpha), M=$(M_biot) Pa"
@info "      c_v=$(round(c_v, sigdigits=3)) m²/s, Sε=$(round(Se, sigdigits=3))"

# ── Time stepping ─────────────────────────────────────────────────────────────
# T_end corresponds to T_v = c_v·t/L² = 1 (roughly 90% consolidation)
T_end   = 1.0 * L_domain^2 / c_v
n_steps = 200
dt_val  = T_end / n_steps
@info "      T_end=$(round(T_end, sigdigits=3)) s,  dt=$(round(dt_val, sigdigits=3)) s"

# ── Fields ────────────────────────────────────────────────────────────────────
u = ScalarField(mesh_dev); initialise!(u, 0.0)   # x-displacement
v = ScalarField(mesh_dev); initialise!(v, 0.0)   # y-displacement (rollers → ~0)
p = ScalarField(mesh_dev); initialise!(p, p0)     # excess pore pressure

# Source fields: updated each step, stored as live mutable references
# p_grad_x = α * ∂p/∂x  (body force in u-equation)
# p_grad_y = α * ∂p/∂y  (body force in v-equation)
# div_u_src = -α * (∇·u^{n+1} - ∇·u^n) / dt  (coupling source in p-equation)
p_grad_x  = ScalarField(mesh_dev); initialise!(p_grad_x,  0.0)
p_grad_y  = ScalarField(mesh_dev); initialise!(p_grad_y,  0.0)
div_u_src = ScalarField(mesh_dev); initialise!(div_u_src, 0.0)

# ── Gradient / divergence helpers ─────────────────────────────────────────────
∇p = Grad{Gauss}(p); pf = FaceScalarField(mesh_dev)
∇u = Grad{Gauss}(u); uf = FaceScalarField(mesh_dev)
∇v_field = Grad{Gauss}(v); vf = FaceScalarField(mesh_dev)

n_cells  = length(mesh_dev.cells)
div_u_new = zeros(n_cells)
div_u_old = zeros(n_cells)

# ── Solver and configuration ──────────────────────────────────────────────────
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
        p = [Dirichlet(:inlet,  0.0),     # drained at x=0
             Zerogradient(:outlet),        # undrained at x=L
             Zerogradient(:bottom),
             Zerogradient(:top)],
    )
)
config = Configuration(solvers=solvers, schemes=schemes,
                       runtime=runtime, hardware=hardware, boundaries=BCs)

# ── Constant flux scalars ─────────────────────────────────────────────────────
mu_cst    = ConstantScalar(mu_val)
alpha_cst = ConstantScalar(lam_val + mu_val)    # μ+λ for GradDiv
k_cst     = ConstantScalar(k_perm)
Se_cst    = ConstantScalar(Se)

# ── Mechanical equations ───────────────────────────────────────────────────────
# OLD Model API required for MonolithicSystem.
# Source(p_grad_x) stores a LIVE reference to p_grad_x.values.
# Updating p_grad_x.values before solve_monolithic! is sufficient —
# no equation rebuild needed.
#
# Equation: -μ∇²u - (μ+λ)∇(∇·u) + 0 = α∇p^n
# FVM sign convention: operator terms go on LHS (negative in our DSL),
# body force on RHS as Source. Since α=1 here:
u_eqn = (
    - Laplacian{Linear}(mu_cst,    u)
    - GradDiv{Linear,1,1}(alpha_cst, u)
    - GradDiv{Linear,1,2}(alpha_cst, v)
    == Source(p_grad_x)      # ← live ref, updated via grad! each step
) → ScalarEquation(u, BCs.u)

v_eqn = (
    - Laplacian{Linear}(mu_cst,    v)
    - GradDiv{Linear,2,1}(alpha_cst, u)
    - GradDiv{Linear,2,2}(alpha_cst, v)
    == Source(p_grad_y)      # ← live ref
) → ScalarEquation(v, BCs.v)

sys_mech = MonolithicSystem([u_eqn, v_eqn], [u, v])

# ── Pressure equation (PDEOperator DSL) ───────────────────────────────────────
# Sε ∂p/∂t - k∇²p = -α Δ(∇·u)/Δt
# Time{Euler}: diagonal += Sε*vol/dt, RHS += Sε*vol/dt * p^n
# Source(div_u_src): RHS += div_u_src[i] * vol  (live ref, per-cell coupling)
# Use direct form here for robustness (equivalent effect); abstract PDE form
# Time{Euler}(Se) - Lap(k) == Source(div)  can be used with → BCs in other contexts.
# Abstract PDE -> BCs with → , then apply to field with L(p)
pde_p = (
    Time{Euler}(Se_cst)
    - Laplacian{Linear}(k_cst)
    == Source(div_u_src)
)
L_p = pde_p → BCs.p
p_eqn = L_p(p)
@reset p_eqn.setup = solvers.p
@reset p_eqn.preconditioner = set_preconditioner(solvers.p.preconditioner, p_eqn)
@reset p_eqn.solver = _workspace(solvers.p.solver, XCALibre._b(p_eqn))

# ── Time loop ─────────────────────────────────────────────────────────────────
@info "Starting fixed-stress Biot consolidation loop..."
@printf("  step    t[s]      T_v     <p>[Pa]  U[%%]\n")
@printf("  %-4s  %-8s  %-6s  %-9s  %-5s\n", "----", "--------", "------", "---------", "-----")

p_mean_init = mean(p.values)

for step in 1:n_steps
    t   = step * dt_val
    T_v = c_v * t / L_domain^2

    # ── Step 1: Update ∇p body force, then solve elastic block ────────────────
    grad!(∇p, pf, p, BCs.p, nothing, config)
    # Store α * ∇p into live source fields (α=1 here; multiply if α≠1)
    @. p_grad_x.values = alpha * ∇p.result.x.values
    @. p_grad_y.values = alpha * ∇p.result.y.values

    mech_res = solve_monolithic!(sys_mech, (BCs.u, BCs.v), config)

    # ── Step 2: Compute Δ(∇·u)/Δt, update source, solve pressure ─────────────
    @. div_u_old = div_u_new

    # ∇·u = ∂u/∂x + ∂v/∂y  (volumetric strain)
    grad!(∇u, uf, u, BCs.u, nothing, config)
    grad!(∇v_field, vf, v, BCs.v, nothing, config)
    @. div_u_new = ∇u.result.x.values + ∇v_field.result.y.values

    # Coupling source: -α * (∇·u^{n+1} - ∇·u^n) / dt
    @. div_u_src.values = -alpha * (div_u_new - div_u_old) / dt_val

    solve_equation!(p_eqn, config)

    if step % 20 == 0 || step == 1
        p_mean = mean(p.values)
        U_pct  = 100 * (1 - p_mean / p_mean_init)
        @printf("  %4d  %8.2f  %6.3f  %9.1f  %5.1f%%\n",
                step, t, T_v, p_mean, U_pct)
    end
end

# ── Analytical solution (Terzaghi) ────────────────────────────────────────────
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

p_anal = [terzaghi_p(x, t_end, p0, L_domain, c_v) for x in xs]
p_err  = maximum(abs.(p.values .- p_anal))

println()
println("Biot Consolidation — Terzaghi 1-D benchmark  (fixed-stress split)")
println("  T_v = $(round(T_v_end, digits=3))")
@printf("  Max |p_FVM - p_analytical| = %.2e Pa  (p₀ = %.0f Pa)\n", p_err, p0)
@printf("  Relative L∞ error           = %.2e\n", p_err / p0)
@printf("  Degree of consolidation     = %.1f%%\n",
        100*(1 - mean(p.values)/p0))
println()
@info "Biot consolidation example complete."

# =============================================================================
# OPERATOR COMPOSITION HIGHLIGHTS
#
#   1. MonolithicSystem([u_eqn, v_eqn], [u, v])
#      — elastic block with GradDiv off-diagonal coupling (same as linear_elastic_2d)
#      — Source(live_field) means body force updates without rebuilding equations
#
#   2. PDEOperator DSL for pressure:
#        L_p = Time{Euler}(Sε) - Laplacian{Linear}(k) == Source(div_u_src)
#        solve_equation!(L_p(p), config)    ← transient solve each step
#
#   3. Fixed-stress split:
#        Update ∇p → solve elastic (u,v) → compute Δ(∇·u)/Δt → solve flow (p)
#
#   4. To extend to monolithic 3-field Biot (u, v, p simultaneously):
#        see biot_consolidation_monolithic.jl — uses ScalarGrad and VectorDiv
#        for the off-diagonal coupling blocks.
# =============================================================================
