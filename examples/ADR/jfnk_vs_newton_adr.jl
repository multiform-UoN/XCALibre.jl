# =============================================================================
# JFNK vs Newton-Exact: Nonlinear scalar diffusion-reaction
# =============================================================================
#
# PDE:  -D·∇²C + β·C³ = S(x),   x ∈ [0, L],   y ∈ [0, H]
# BCs:  C(0, y) = 0,   C(L, y) = 0,   ∂C/∂n = 0 on top/bottom
# Exact solution:  C*(x) = sin(πx/L)
# Source:          S(x) = D(π/L)²·sin(πx/L) + β·sin³(πx/L)
#
# ─────────────────────────────────────────────────────────────────────────────
# TWO SOLVERS COMPARED
# ─────────────────────────────────────────────────────────────────────────────
#
# METHOD 1 — Newton-Exact  (sparse Jacobian via ForwardDiff)
#   At each outer Newton step k:
#     1. Linearise β·C³ at C^k via ForwardDiff:
#          β·C³ ≈ 3β·(C^k)²·C − 2β·(C^k)³
#     2. Assemble sparse Jacobian  J ∈ ℝ^{N×N}   (N = # cells)
#     3. Solve  J·δC = −F(C^k)  via BiCGSTAB
#     4. Update C^{k+1} = C^k + δC
#   Storage: O(N·nnz_per_row) for the sparse J at every outer step.
#
# METHOD 2 — Jacobian-Free Newton-Krylov (JFNK)
#   At each outer Newton step k:
#     1. Evaluate F(C^k) matrix-free:
#          F_i(C) = [A_diff·C − b_diff]_i + β·C_i³·vol_i
#        A_diff is assembled ONCE (constant linear diffusion matrix).
#     2. Check convergence: ‖F‖ < tol
#     3. Solve J·δC = −F via inner GMRES, approximating the Jacobian action as
#          J·v ≈ [F(C + ε·v) − F(C)] / ε    (directional finite difference)
#        No Jacobian matrix is ever formed or stored.
#     4. Update C^{k+1} = C^k + δC
#   Storage: O(N·m) for GMRES Krylov basis (m ≪ N).
#
# ─────────────────────────────────────────────────────────────────────────────
# WHEN TO PREFER EACH METHOD
# ─────────────────────────────────────────────────────────────────────────────
#
# Newton-Exact wins when:
#   • N is small — sparse J fits in memory, assembly is cheap.
#   • Few outer Newton steps — quadratic convergence amortises J cost.
#   • Good preconditioner is available (ILU, AMG) — inner solve is fast.
#   • The nonlinearity is smooth and AutoDiff can trace through all physics.
#
# JFNK wins when:
#   • N is very large (GPU / distributed) — J storage O(N·nnz) is infeasible.
#   • Physics is complex / multi-physics / black-box — no Jacobian available.
#   • Inner GMRES converges in few steps (well-conditioned J).
#   • Memory is the bottleneck, not FLOPs.
#   • Parallelisation: F(C) can be evaluated in parallel; J is harder.
#
# ─────────────────────────────────────────────────────────────────────────────
# JFNK THEORY: INEXACT NEWTON + FINITE-DIFFERENCE JACOBIAN
# ─────────────────────────────────────────────────────────────────────────────
#
# The Newton iteration solves F(C) = 0 by iterating:
#   J(C^k) · δC^k = −F(C^k),    C^{k+1} = C^k + δC^k
#
# JFNK avoids forming J by approximating J·v during GMRES via:
#   J·v ≈ [F(C + ε·v) − F(C)] / ε
#
# Choice of ε (Walker & Pernice 1998):
#   ε = √(machine_eps) · (1 + ‖C‖)          (avoids cancellation when ‖v‖≈1)
#
# Inexact Newton condition (Eisenstat-Walker):
#   Choose inner tolerance η_k adaptively so GMRES does just enough work:
#     η_k = min(η_max,  √(‖F^k‖ / ‖F^{k-1}‖))
#   This gives superlinear (→ quadratic) outer convergence without wasting
#   inner GMRES iterations when the outer iterate is still far from the root.
#
# ─────────────────────────────────────────────────────────────────────────────
# XCALIBRE PRIMITIVES USED
# ─────────────────────────────────────────────────────────────────────────────
#
# Newton-Exact:
#   newton_solve!(eqn, config)
#     → calls linearize_physics (ForwardDiff), assembles J, solves via Krylov
#
# JFNK:
#   residual!(r, eqn, config; assemble=false)
#     → computes r = A·phi.values − b using the PRE-ASSEMBLED constant matrix.
#       No ForwardDiff, no new allocations, one sparse MV per call.
#   LinearOperator (LinearOperators.jl) + Krylov.gmres
#     → wraps the J·v finite-difference function as a matrix-compatible object.
# =============================================================================

using XCALibre
using Accessors
using LinearAlgebra
using Printf
using LinearOperators
import Krylov

# ── Mesh ──────────────────────────────────────────────────────────────────────
grids_dir = pkgdir(XCALibre, "examples", "0_GRIDS")
mesh      = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=1.0)
backend   = CPU(); workgroup = 1024
hardware  = Hardware(backend=backend, workgroup=workgroup)
mesh_dev  = adapt(backend, mesh)

# ── Problem parameters ────────────────────────────────────────────────────────
const L_domain = 1000.0   # domain length in x  (matches quad40.unv, [0,1000]×[0,1000])
const D_val    = 50.0     # diffusion coefficient
const β        = 5e-2     # nonlinear reaction coefficient
const tol_nl   = 1e-8     # outer Newton convergence tolerance  ‖F‖ < tol_nl

# ── Manufactured source for exact solution C*(x) = sin(πx/L) ─────────────────
# Verify: -D ∂²C*/∂x² = D(π/L)²·sin(πx/L)   and   β·(C*)³ = β·sin³(πx/L)
# → S(x) = D(π/L)²·sin(πx/L) + β·sin³(πx/L)
xs      = [mesh_dev.cells[i].centre[1] for i in eachindex(mesh_dev.cells)]
C_exact = @. sin(π * xs / L_domain)
S_vals  = @. D_val * (π/L_domain)^2 * sin(π*xs/L_domain) + β * sin(π*xs/L_domain)^3

S_field = ScalarField(mesh_dev)
S_field.values .= S_vals

# ── Boundary conditions ───────────────────────────────────────────────────────
BCs = assign(
    region = mesh_dev,
    (
        C = [
            Dirichlet(:inlet,  0.0),      # C(0) = sin(0) = 0
            Dirichlet(:outlet, 0.0),      # C(L) = sin(π) = 0
            Zerogradient(:bottom),
            Zerogradient(:top),
        ],
    )
)

# ── Shared solver / scheme settings ──────────────────────────────────────────
D_cs = ConstantScalar(D_val)

solvers_cfg = (
    C = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                    convergence=tol_nl, relax=1.0, itmax=500),
)
schemes  = (C = Schemes(laplacian=Linear),)
config   = Configuration(
    solvers    = solvers_cfg,
    schemes    = schemes,
    runtime    = Runtime(iterations=50, write_interval=-1, time_step=1.0),
    hardware   = hardware,
    boundaries = BCs,
)

n_cells = length(mesh_dev.cells)
vols    = [mesh_dev.cells[i].volume for i in eachindex(mesh_dev.cells)]

# =============================================================================
# METHOD 1 — Newton-Exact  (sparse Jacobian assembled at every outer step)
# =============================================================================
#
# NonLinearSi stores the function c → β·c³ and its derivative.
# At each outer step, linearize_physics (ForwardDiff) evaluates the derivative
# cell-by-cell and assembles the sparse Jacobian:
#   J = -D·∇²  +  diag(3β·(C^k)²)
# XCALibre then solves  J·δC = −F  via BiCGSTAB with Jacobi preconditioning.
#
# Key cost: one ForwardDiff pass (O(N)) + full sparse assembly (O(N·nnz)) per
# outer Newton iteration.  The assembled J is stored for the inner solve.
# =============================================================================

C_newton = ScalarField(mesh_dev); initialise!(C_newton, 0.0)

# New PDEOperator DSL: NonLinearSi(func) now returns an OperatorTemplate — field deferred.
# L_nl is a reusable PDEOperator; newton_solve!(L_nl, C_newton, config) binds C_newton
# internally and drives the outer Newton loop.
L_nl = (
      - Laplacian{Linear}(D_cs)
      + NonLinearSi(c -> β * c^3)
      == Source(S_field)
) → BCs.C
L_nl = L_nl → solvers_cfg.C

@info "METHOD 1: Newton-Exact (sparse Jacobian via ForwardDiff)..."
t_newton = @elapsed result_newton = newton_solve!(L_nl, C_newton, config;
                                                  tol=tol_nl, verbose=false)

err_newton = maximum(abs.(C_newton.values .- C_exact))
@printf("\nNewton-Exact:  %d outer iters,  max|C − C*| = %.2e  (%.3f s)\n",
        result_newton.iterations, err_newton, t_newton)
println("  Outer residual history (‖F‖):")
for (i, r) in enumerate(result_newton.residuals)
    @printf("    step %2d  ‖F‖ = %.4e\n", i, r)
end

# =============================================================================
# METHOD 2 — Jacobian-Free Newton-Krylov (JFNK)
# =============================================================================
#
# The nonlinear residual at cell i is:
#   F_i(C) = [−D·∇²C + β·C³ − S]_i · vol_i
#
# Now that NonLinearSi has a scheme_source! method, explicit_residual! evaluates
# the full nonlinear F(C) in a single kernel pass — no splitting, no workaround.
# The scheme evaluates f(C_i) = β·C_i³ explicitly at the current cell value,
# exactly as needed for the JFNK residual function.
#
# J·v is approximated by a directional finite difference of F:
#   J·v ≈ (F(C + ε·v) − F(C)) / ε
# This requires TWO evaluations of F per GMRES step (one is reused from the
# previous step via the Krylov recurrence).
#
# The inner GMRES convergence tolerance is chosen via the Eisenstat-Walker
# criterion to balance outer vs inner work adaptively.
# =============================================================================

C_jfnk = ScalarField(mesh_dev); initialise!(C_jfnk, 0.0)

# ── Build the full nonlinear equation — same L_nl as Newton-Exact ─────────────
# explicit_residual! now works directly on NonLinearSi equations:
# it calls scheme_source! for each term, which for NonLinearSi evaluates f(phi[i])
# at the current cell value — no sparse matrix needed, one kernel pass.
eqn_jfnk = L_nl(C_jfnk)

# ── Nonlinear residual F(C) via explicit_residual! ────────────────────────────
# F_i(C) = [−D·∇²C + β·C³ − S]_i · vol_i  (full nonlinear, matrix-free)
function compute_F!(r::AbstractVector, C_field::ScalarField,
                    eqn::ModelEquation, config)
    explicit_residual!(r, eqn, C_field, config)
end

# ── Jacobian-vector product via finite difference ─────────────────────────────
# Approximates J·v = ∂F/∂C · v ≈ (F(C + ε·v) − F(C)) / ε
# r0 must already contain F(C) at the current iterate (avoids one F evaluation).
function jac_vec!(y::AbstractVector, v::AbstractVector,
                  C_field::ScalarField, r0::AbstractVector,
                  eqn::ModelEquation,
                  C_save::AbstractVector, r_tmp::AbstractVector, config)
    # Perturbation size: sqrt(eps) * (1 + ‖C‖) avoids cancellation
    ε = sqrt(eps(eltype(C_field.values))) * (1.0 + norm(C_field.values))

    # Save current state, perturb by ε·v
    C_save .= C_field.values
    @. C_field.values = C_save + ε * v

    # Evaluate F at perturbed state
    compute_F!(r_tmp, C_field, eqn, config)

    # Finite difference: (F(C + ε·v) − F(C)) / ε
    @. y = (r_tmp - r0) / ε

    # Restore C
    C_field.values .= C_save
end

# ── Outer Newton + inner GMRES (Eisenstat-Walker inexact Newton) ──────────────
r0      = zeros(n_cells)    # nonlinear residual F(C^k)
C_save  = zeros(n_cells)    # pre-allocated save buffer
r_tmp   = zeros(n_cells)    # pre-allocated scratch for J·v evaluations

jfnk_residuals = Float64[]

@info "METHOD 2: JFNK (matrix-free, Jacobian-Free Newton-Krylov)..."
t_jfnk = @elapsed begin
    local converged_jfnk = false
    for iter in 1:100
        # ── 1. Evaluate nonlinear residual F(C^k) ─────────────────────────────
        compute_F!(r0, C_jfnk, eqn_jfnk, config)
        rnorm = norm(r0)
        push!(jfnk_residuals, rnorm)

        # ── 2. Convergence check ──────────────────────────────────────────────
        if rnorm < tol_nl
            converged_jfnk = true
            break
        end

        # ── 3. Inexact Newton forcing term (Eisenstat-Walker type 2) ──────────
        # Adaptive inner tolerance: η_k = min(0.5, sqrt(‖F^k‖/‖F^{k-1}‖))
        # Tightens automatically as the outer Newton converges.
        η = length(jfnk_residuals) > 1 ?
            min(0.5, sqrt(jfnk_residuals[end] / jfnk_residuals[end-1])) :
            0.5

        # ── 4. Build the J·v LinearOperator (no Jacobian stored) ─────────────
        # LinearOperator wraps an arbitrary function as a matrix-compatible object.
        # Krylov.gmres will call (y, v) -> jac_vec!(...) at each GMRES iteration.
        J_op = LinearOperator(Float64, n_cells, n_cells, false, false,
            (y, v) -> jac_vec!(y, v, C_jfnk, r0, eqn_jfnk,
                               C_save, r_tmp, config))

        # ── 5. Inner GMRES: solve J·δC = −F to the inexact Newton tolerance ───
        # δC is the Newton search direction. GMRES never needs to store J,
        # only the Krylov basis vectors (m ≪ N columns).
        δC, stats = Krylov.gmres(J_op, -r0; rtol=η, atol=0.0, itmax=200)

        # ── 6. Newton update ──────────────────────────────────────────────────
        @. C_jfnk.values += δC

        @printf("    outer %2d  ‖F‖ = %.4e  inner_tol = %.2e  GMRES iters = %d\n",
                iter, rnorm, η, stats.niter)
    end
    converged_jfnk || @warn "JFNK did not converge in 100 outer iterations"
end

err_jfnk = maximum(abs.(C_jfnk.values .- C_exact))
@printf("\nJFNK:         %d outer iters,  max|C − C*| = %.2e  (%.3f s)\n",
        length(jfnk_residuals), err_jfnk, t_jfnk)

# =============================================================================
# COMPARISON SUMMARY
# =============================================================================
println()
println("═══════════════════════════════════════════════════════════════════")
println("                    COMPARISON SUMMARY")
println("═══════════════════════════════════════════════════════════════════")
@printf("  %-22s %6s %12s %10s\n", "Method", "Iters", "max|C−C*|", "Time(s)")
@printf("  %-22s %6d %12.2e %10.3f\n",
        "Newton-Exact (FD+J)", result_newton.iterations, err_newton, t_newton)
@printf("  %-22s %6d %12.2e %10.3f\n",
        "JFNK (matrix-free)", length(jfnk_residuals), err_jfnk, t_jfnk)
println()
println("Notes on results:")
println("  • Both methods achieve the same accuracy (discretisation error).")
println("  • Newton-Exact is faster here: N=1600 DOF, J fits easily in memory.")
println("  • JFNK advantage grows with N: Jacobian storage O(N·nnz) becomes")
println("    prohibitive for large GPU meshes (N ~ 10^6–10^8 cells).")
println("  • JFNK inner iterations reflect the condition number of J.")
println("  • Inexact Newton criterion (Eisenstat-Walker) reduces inner")
println("    GMRES work early on and tightens automatically near convergence.")
println()
println("Residual convergence (outer Newton):")
@printf("  %-5s %14s %14s\n", "Step", "Newton-Exact", "JFNK")
for i in 1:max(length(result_newton.residuals), length(jfnk_residuals))
    r_ne = i ≤ length(result_newton.residuals) ? result_newton.residuals[i] : NaN
    r_jf = i ≤ length(jfnk_residuals)          ? jfnk_residuals[i]          : NaN
    @printf("  %-5d %14.4e %14.4e\n", i, r_ne, r_jf)
end
println("═══════════════════════════════════════════════════════════════════")
@info "JFNK vs Newton-Exact example complete."

# =============================================================================
# SUMMARY OF KEY DESIGN PATTERNS
#
#   Explicit Jacobian (Newton-Exact):
#     L = (... + NonLinearSi(f) == ...) → BCs → solver   # new DSL: field deferred
#     newton_solve!(L, C, config)
#       → ForwardDiff linearises f at each outer step
#       → assembles sparse J, solves via BiCGSTAB
#
#   JFNK:
#     eqn_lin = L_linear(C)                     # constant linear part
#     discretise!(eqn_lin, C, config)            # assemble A_diff ONCE
#     apply_boundary_conditions!(eqn_lin, config)
#
#     compute_F!(r, C, eqn_lin, ...) using residual!(...; assemble=false) + manual nonlinear term
#     jac_vec!(y, v, ...) using two compute_F! calls + finite difference
#
#     J_op = LinearOperator(n, n, false, false, (y,v) -> jac_vec!(...))
#     δC, _ = Krylov.gmres(J_op, -F; rtol=η_k)
#     C.values .+= δC
# =============================================================================
