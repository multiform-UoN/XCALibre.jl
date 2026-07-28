# =============================================================================
# Transient ADR — abstract operator composition for explicit vs implicit time stepping
# =============================================================================
#
# PDE:  ∂C/∂t - D·∇²C + k·C = 0,   x ∈ [0, L],  t > 0
#
# BCs:  C(0, t) = 1,  C(L, t) = 0
# IC:   C(x, 0) = 0
#
# Analytical steady state (t → ∞):
#   C∞(x) = sinh(μ·(L-x)) / sinh(μ·L),   μ = sqrt(k/D)
#
# ─────────────────────────────────────────────────────────────────────────────
# KEY IDEA: define the SPATIAL operator L once, then compose time integration
# separately.  The same L is reused for:
#
#   Explicit Euler (Forward):
#     vol·(C^{n+1} - C^n)/dt = -r^n,   r^n = A·C^n - b
#     → C^{n+1} = C^n - (dt/vol)·r^n         [no linear solve, CFL-limited]
#
#   Implicit Euler (Backward):
#     (vol/dt + A)·C^{n+1} = (vol/dt)·C^n + b
#     → L_impl = L + Time{Euler}(1)           [unconditionally stable, one solve/step]
#
#   Crank–Nicolson (trapezoidal):
#     (vol/dt + A/2)·C^{n+1} = (vol/dt - A/2)·C^n + b
#     → L_half = L * 0.5; L_cn = L_half + Time{Euler}(1)
#       Phase 4 split assembly: assemble → subtract explicit half → solve_preassembled!
#
# ─────────────────────────────────────────────────────────────────────────────
# Mesh:  quad40.unv  (1000×1000 domain, 40×40 cells, x ∈ [0, 1000])
#   inlet  (x=0):   C = 1
#   outlet (x=L):   C = 0
#   bottom/top:     Zerogradient
# =============================================================================

using XCALibre
using Accessors
using LinearAlgebra
using Printf
using Statistics

# ── Mesh ──────────────────────────────────────────────────────────────────────
grids_dir = pkgdir(XCALibre, "examples", "0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=1.0)
backend  = CPU(); workgroup = 1024
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# ── Parameters ────────────────────────────────────────────────────────────────
L_domain = 1000.0          # domain length in x
D_val    = 50.0            # diffusion coefficient  (dx=25, dt=2 → Fo=0.32 < 0.5 ✓)
k_val    = 1e-4            # linear reaction rate
dt       = 2.0             # time step
n_steps  = 500             # total steps  (T = 1000 s)

D = ConstantScalar(D_val)
k = ConstantScalar(k_val)

# ── Boundary conditions ───────────────────────────────────────────────────────
solvers_cfg = (
    C = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                    convergence=1e-10, relax=1.0),
)
schemes = (C = Schemes(laplacian=Linear),)
runtime = Runtime(iterations=1, write_interval=-1, time_step=dt)

BCs = assign(
    region = mesh_dev,
    (
        C = [
            Dirichlet(:inlet,  1.0),
            Dirichlet(:outlet, 0.0),
            Zerogradient(:bottom),
            Zerogradient(:top),
        ],
    )
)

config = Configuration(solvers=solvers_cfg, schemes=schemes,
                       runtime=runtime, hardware=hardware, boundaries=BCs)

# ── Fields (one per scheme) ───────────────────────────────────────────────────
C_exp = ScalarField(mesh_dev); initialise!(C_exp, 0.0)
C_imp = ScalarField(mesh_dev); initialise!(C_imp, 0.0)

# =============================================================================
# STEP 1 — define the SPATIAL operator L (no time term, no field bound)
#
#   L represents:   -D·∇²C + k·C  (the "stiffness" of the steady problem)
#
#   Boundary conditions and solver settings are attached once.
#   L can now be applied to any scalar field with L(phi).
# =============================================================================
L = (
    - Laplacian{Linear}(D)
    + Si(k)
    == Source(0.0)
) → BCs.C
L = L → solvers_cfg.C

# =============================================================================
# STEP 2a — EXPLICIT EULER
#
#   Evaluate spatial residual r^n = A·C^n - b, then step forward:
#     C^{n+1}_i = C^n_i - (dt / vol_i) · r^n_i
#
#   No linear solve required — just a matrix-vector product + vector update.
#   Stability: Fourier number  Fo = D·dt/dx² < 0.5  must hold.
#     dx = 1000/40 = 25,  Fo = 50·2/625 = 0.16  ✓
#
#   Build the equation once (allocates sparse matrix).
#   Reuse it every step — residual! re-assembles A (fast, no reallocation).
# =============================================================================
vols = [mesh_dev.cells[i].volume for i in eachindex(mesh_dev.cells)]
r    = zeros(length(mesh_dev.cells))

C_exp_eqn = L(C_exp)   # bind L to C_exp — creates ModelEquation with stored BCs/setup

@info "Explicit Euler time loop..."
for step in 1:n_steps
    residual!(r, C_exp_eqn, config)            # r = A·C^n - b  (pure spatial)
    @. C_exp.values -= (dt / vols) * r         # forward Euler update (no solve)

    if step % 100 == 0
        @printf("  [explicit] step=%4d  t=%6.1f  mean(C)=%.5f  |r|∞=%.2e\n",
                step, step*dt, mean(C_exp.values), maximum(abs.(r)))
    end
end

# =============================================================================
# STEP 2b — IMPLICIT EULER
#
#   Add the Time{Euler} template to L — spatial operator L unchanged:
#     L_impl = L + Time{Euler}(1)
#
#   Time{Euler} contributes to each cell:
#     diagonal:  +vol/dt     (mass matrix / dt)
#     RHS:       +vol/dt · C^n_i   (old-time term)
#
#   The resulting linear system per step:
#     (A + vol/dt·I) · C^{n+1} = b + (vol/dt)·C^n
#
#   Unconditionally stable — dt can be arbitrarily large.
#   One linear solve per step (BiCGSTAB, warm-started).
# =============================================================================
L_impl = L + Time{Euler}(ConstantScalar(1.0))  # spatial + time, BCs/setup inherited
C_imp_eqn = L_impl(C_imp)

# Initialise solver workspace once (avoids re-allocation every step)
@reset C_imp_eqn.preconditioner = set_preconditioner(
    solvers_cfg.C.preconditioner, C_imp_eqn)
@reset C_imp_eqn.solver = _workspace(solvers_cfg.C.solver, XCALibre._b(C_imp_eqn))

@info "Implicit Euler time loop..."
for step in 1:n_steps
    res = solve_equation!(C_imp_eqn, config)   # assemble + solve each step

    if step % 100 == 0
        @printf("  [implicit] step=%4d  t=%6.1f  mean(C)=%.5f  solver_res=%.2e\n",
                step, step*dt, mean(C_imp.values), res)
    end
end

# =============================================================================
# STEP 2c — CRANK–NICOLSON
#
# The CN update (trapezoidal rule) is:
#
#   (vol/dt)·C^{n+1} + (A/2)·C^{n+1} = (vol/dt)·C^n - (A/2)·C^n + b
#
# Rearranged using r^n = A·C^n - b (explicit residual at current step):
#
#   (vol/dt·I + A/2)·C^{n+1} = vol/dt·C^n + b - (1/2)·r^n
#   ↑ LHS system                ↑ standard implicit RHS, minus half the explicit residual
#
# Implementation with L * 0.5:
#   L_half   = L * 0.5                    # spatial operator scaled by 1/2
#   L_cn_lhs = L_half + Time{Euler}(1.0)  # (A/2 + vol/dt) system
#
# Each step:
#   r_n   = residual!(r_cn, L_half(C_cn), config)   # (1/2)·(A·C^n - b)
#   assemble the implicit LHS once per step, modify b by subtracting r_n
#   → solve_equation! handles the rest via stored setup
# =============================================================================

C_cn = ScalarField(mesh_dev); initialise!(C_cn, 0.0)
r_cn = zeros(length(mesh_dev.cells))

# Spatial operator scaled by 1/2 (for the CN trapezoidal split)
L_half = L * 0.5

# Build the CN LHS operator: A/2 + vol/dt·I
L_cn = L_half + Time{Euler}(ConstantScalar(1.0))
C_cn_eqn = L_cn(C_cn)

# Initialise solver workspace
@reset C_cn_eqn.preconditioner = set_preconditioner(
    solvers_cfg.C.preconditioner, C_cn_eqn)
@reset C_cn_eqn.solver = _workspace(solvers_cfg.C.solver, XCALibre._b(C_cn_eqn))

# Pre-built half-spatial equation for evaluating r^n each step
C_cn_half_eqn = L_half(C_cn)

@info "Crank-Nicolson time loop..."
for step in 1:n_steps
    # Explicit half: r_cn = (A/2)·C^n - b/2  (using the half-scaled spatial operator)
    residual!(r_cn, C_cn_half_eqn, config)

    # Assemble LHS: (A/2 + vol/dt·I)·C^{n+1}
    # and implicit RHS: (vol/dt)·C^n + b/2
    assemble_matrix!(C_cn_eqn, config)
    assemble_rhs!(C_cn_eqn, C_cn_eqn.model.sources[1], config)

    # CN correction: subtract explicit half r_cn from b → correct CN RHS
    # Final RHS: (vol/dt)·C^n + b/2 - ((A/2)·C^n - b/2) = (vol/dt)·C^n + b - (A/2)·C^n
    _b(C_cn_eqn) .-= r_cn

    # Solve without re-assembling (preserves the manual b modification above)
    res = solve_preassembled!(C_cn_eqn, config)

    if step % 100 == 0
        @printf("  [CN      ] step=%4d  t=%6.1f  mean(C)=%.5f  solver_res=%.2e\n",
                step, step*dt, mean(C_cn.values), res)
    end
end

# =============================================================================
# STEP 2d — RUNGE-KUTTA 4 (RK4)
#
# RK4 is a classical explicit 4-stage time integration method.
# It achieves 4th-order temporal accuracy in exchange for 4 residual
# evaluations per time step (vs. 1 for explicit Euler).
#
# For the ODE  M·dC/dt = −F(C),   M = diag(vol_i):
#
#   k₁ = −F(Cⁿ)                      / vol    ← stage 1 slope
#   k₂ = −F(Cⁿ + dt/2 · k₁)         / vol    ← stage 2 slope (midpoint via k₁)
#   k₃ = −F(Cⁿ + dt/2 · k₂)         / vol    ← stage 3 slope (midpoint via k₂)
#   k₄ = −F(Cⁿ + dt   · k₃)         / vol    ← stage 4 slope (endpoint via k₃)
#
#   Cⁿ⁺¹ = Cⁿ + dt/6 · (k₁ + 2k₂ + 2k₃ + k₄)
#
# Each stage evaluates the spatial residual F at a different C, which in
# XCALibre means a sparse matrix-vector product  r = A·C − b  using the
# PRE-ASSEMBLED constant matrix.  No solve, no assembly — just 4 MV products.
#
# Stability for diffusion (1D):
#   RK4 stability region includes |λ·dt| ≤ 2.79 for imaginary eigenvalues.
#   For the Laplacian, max |λ| ≈ 4D/dx².  Compare with Explicit Euler (≤ 2):
#   RK4 allows a ~40% larger time step than Forward Euler for the same dt.
#   Our dt=2.0, dx=25, D=50 → Fo = D·dt/dx² = 0.16 ≪ 0.5 (both stable).
#
# Key implementation primitive:
#   residual!(r, eqn, config; assemble=false)
#   → computes r = A·phi.values − b using the stored (pre-assembled) A.
#     Changing phi.values between calls gives the residual at different stages
#     without touching the matrix.
# =============================================================================

C_rk4 = ScalarField(mesh_dev); initialise!(C_rk4, 0.0)

# Build the same spatial equation L bound to C_rk4 (reuse the spatial L defined above)
C_rk4_eqn = L(C_rk4)

# ── ONE-TIME MATRIX ASSEMBLY ──────────────────────────────────────────────────
# The operator L = −D·∇² + k·I has a CONSTANT matrix A (no field dependence).
# Assemble once here; residual! with assemble=false reuses A for all 4 stages.
discretise!(C_rk4_eqn, C_rk4, config)
apply_boundary_conditions!(C_rk4_eqn, config)

# Pre-allocate stage slopes and scratch arrays (avoids allocation inside loop)
n_cells_rk = length(mesh_dev.cells)
r_rk = zeros(n_cells_rk)
k1   = zeros(n_cells_rk)
k2   = zeros(n_cells_rk)
k3   = zeros(n_cells_rk)
k4   = zeros(n_cells_rk)
C_rk4_save = zeros(n_cells_rk)   # save Cⁿ before stage perturbations

@info "RK4 time loop..."
for step in 1:n_steps
    # Save Cⁿ — stages 2–4 perturb C_rk4.values, so we restore from here
    C_rk4_save .= C_rk4.values

    # ── Stage 1: slope at Cⁿ ─────────────────────────────────────────────────
    residual!(r_rk, C_rk4_eqn, config; assemble=false)   # r = A·Cⁿ − b
    @. k1 = -(1 / vols) * r_rk                            # k₁ = dC/dt at Cⁿ

    # ── Stage 2: slope at Cⁿ + dt/2·k₁ ─────────────────────────────────────
    @. C_rk4.values = C_rk4_save + (dt/2) * k1
    residual!(r_rk, C_rk4_eqn, config; assemble=false)
    @. k2 = -(1 / vols) * r_rk

    # ── Stage 3: slope at Cⁿ + dt/2·k₂ ─────────────────────────────────────
    @. C_rk4.values = C_rk4_save + (dt/2) * k2
    residual!(r_rk, C_rk4_eqn, config; assemble=false)
    @. k3 = -(1 / vols) * r_rk

    # ── Stage 4: slope at Cⁿ + dt·k₃ ───────────────────────────────────────
    @. C_rk4.values = C_rk4_save + dt * k3
    residual!(r_rk, C_rk4_eqn, config; assemble=false)
    @. k4 = -(1 / vols) * r_rk

    # ── RK4 weighted update ──────────────────────────────────────────────────
    @. C_rk4.values = C_rk4_save + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    if step % 100 == 0
        @printf("  [RK4     ] step=%4d  t=%6.1f  mean(C)=%.5f\n",
                step, step*dt, mean(C_rk4.values))
    end
end

# =============================================================================
# VERIFICATION — compare all schemes to analytical steady state
# =============================================================================
xs = [mesh_dev.cells[i].centre[1] for i in eachindex(mesh_dev.cells)]
mu  = sqrt(k_val / D_val)
C_ss = sinh.(mu .* (L_domain .- xs)) ./ sinh(mu * L_domain)

err_exp = maximum(abs.(C_exp.values  .- C_ss))
err_imp = maximum(abs.(C_imp.values  .- C_ss))
err_cn  = maximum(abs.(C_cn.values   .- C_ss))
err_rk4 = maximum(abs.(C_rk4.values  .- C_ss))

println()
println("Steady-state comparison  (T = $(n_steps*dt) s):")
@printf("  Explicit Euler  max|C - C∞| = %.2e\n", err_exp)
@printf("  Implicit Euler  max|C - C∞| = %.2e\n", err_imp)
@printf("  Crank-Nicolson  max|C - C∞| = %.2e\n", err_cn)
@printf("  RK4             max|C - C∞| = %.2e\n", err_rk4)
println()
@info "Transient ADR example complete."

# =============================================================================
# SUMMARY OF OPERATOR COMPOSITION PATTERN
#
#   L          = spatial PDE operator  (reusable across time schemes)
#   L(phi)     = bound equation  (ModelEquation, ready to assemble/solve)
#
#   Explicit Euler:
#     r = residual!(r, L(phi), config)
#     phi.values .-= (dt/vol) .* r           # no solve, CFL-limited
#
#   Implicit Euler:
#     L_impl = L + Time{Euler}(ConstantScalar(1.0))
#     solve_equation!(L_impl(phi), config)   # one solve/step, unconditionally stable
#
#   Crank-Nicolson:
#     L_half = L * 0.5                       # scale spatial op
#     L_cn   = L_half + Time{Euler}(1.0)    # trapezoidal LHS
#     r_n    = residual!(r, L_half(phi), ...) # explicit half-residual
#     assemble_matrix! + assemble_rhs! + subtract r_n before solve_preassembled!
#
#   RK4 (explicit, 4th-order, 4 residual evaluations per step):
#     eqn = L(phi); discretise!(eqn, phi, config); apply_boundary_conditions!(eqn, config)
#     for each step:
#       k1 = −residual!(r, eqn, config; assemble=false) / vol   # at Cⁿ
#       k2 = −residual!(r, eqn, config; assemble=false) / vol   # at Cⁿ + dt/2·k₁
#       k3 = −residual!(r, eqn, config; assemble=false) / vol   # at Cⁿ + dt/2·k₂
#       k4 = −residual!(r, eqn, config; assemble=false) / vol   # at Cⁿ + dt·k₃
#       phi.values .= Cⁿ + dt/6·(k1 + 2k2 + 2k3 + k4)
#
#   Newton:     newton_solve!(L, phi, config)            # nonlinear L
#
#   Steady:     solve_equation!(L(phi), config)          # no time term at all
#
# The key primitive for explicit schemes is residual!(r, eqn, config; assemble=false):
#   → computes r = A·phi.values − b using the pre-assembled constant matrix A.
#   → changing phi.values gives the residual at different stage points with
#     zero assembly overhead (one sparse MV per call).
# =============================================================================
