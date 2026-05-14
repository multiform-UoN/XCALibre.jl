# =============================================================================
# LAPLACE: ONE PROBLEM — THREE COMPUTATIONAL REALIZATIONS
# =============================================================================
#
# Problem
# --------
#   −∇²φ = 0  on [0,1]²
#   φ = 1   on left wall (x = 0)
#   φ = 0   on right wall (x = 1)
#   ∂φ/∂n = 0  on upper/lower walls (insulated)
#
# Exact solution:  φ(x, y) = 1 − x
#
# =============================================================================
# ARCHITECTURAL OVERVIEW
# =============================================================================
#
# This example demonstrates the layered architecture of XCALibre.jl.
# The key insight is:
#
#   SEMANTIC DEFINITION (PDE + BCs) is kept SEPARATE from
#   COMPUTATIONAL REALIZATION (how the algebra is executed).
#
# Layer 1 — PDE/operator semantics (user-facing, physics-level)
# ─────────────────────────────────────────────────────────────
#   Dirichlet(:left_wall, 1.0)   — prescribes φ = 1 at left boundary
#   Zerogradient(:upper_wall)    — prescribes ∂φ/∂n = 0 at top boundary
#   -Laplacian{Linear}(gamma)    — the −∇·(γ∇φ) operator
#
#   These objects are INDEPENDENT of how the equations are solved.
#   Changing the solver or backend does not change how BCs are declared.
#
# Layer 2 — algebraic realization (how the physics is turned into algebra)
# ─────────────────────────────────────────────────────────────────────────
#   A. Assembled sparse FV:
#        @define_boundary Dirichlet → (ap, bp) pairs → CSR matrix modification
#        Result: A·φ = b, solved by a Krylov method.
#        PRIMARY path — production-ready, default, handles all BCs and operators.
#
#   B. Matrix-free residual evaluation:
#        Operator-specific kernels: Dirichlet → D_f*(φ_P−value), Zerogradient → 0
#        full_residual! → R(φ) without forming A.
#        COMPLEMENTARY path — for JFNK inner loops, GPU Newton,
#        convergence monitoring without re-assembling A.
#
# Layer 3 — computational backend (where the computation runs)
# ─────────────────────────────────────────────────────────────
#   CPU()  — standard iterative solve
#   CUDABackend() / ROCBackend() — GPU execution (same kernels, different backend)
#
# IMPORTANT — When to use each realization
# -----------------------------------------
#   Assembled FV (Realization A): the standard choice for ALL production workflows.
#     Use the low-level API (Realization A1) when you need direct access to A and b.
#     Use the PDEOperator DSL (Realization A2) for cleaner composition and reusability.
#     Both build the SAME CSR matrix; the DSL is syntactic sugar, not a new path.
#
#   Matrix-free (Realization B): a SPECIALISED path for:
#     1. JFNK: inner Krylov loop uses R(φ) and J(φ)·v; no sparse matrix needed.
#     2. GPU Newton: residual kernels avoid irregular CSR access patterns.
#     3. Residual monitoring alongside the assembled solver (hybrid use).
#     4. Nonlinear problems where linearize_physics would be expensive.
#
#   Matrix-free does NOT replace assembled FV; it COMPLEMENTS it.
#   For simple linear problems on CPU, assembled FV is always preferred.
#
# NOTE ON BC LOWERING IN REALIZATION B
# -------------------------------------
# Realization B uses operator-specific kernels for BCs — the Laplacian stencil
# formula is embedded in the Dirichlet kernel body. This is correct: the BC
# residual formula depends on both the BC type AND the operator (Laplacian here).
# For nonlinear BCs (NonLinearRobin), see bc_ops.jl and prototype_D_bc_lowering.jl.
# Standard linear BCs use @define_boundary as the single source of truth; the
# operator-specific kernels below are consistent with that but live separately.

using XCALibre
using KernelAbstractions
using Atomix
using LinearAlgebra
using SparseArrays
using StaticArrays
using SparseMatricesCSR
using Test
using Printf

# Interior residual kernel (Prototype A) — provides laplacian_residual!
include("prototype_A_laplacian_residual.jl")

# =============================================================================
# SHARED SETUP — mesh, backend, fields
# =============================================================================
# All three realizations solve exactly the same problem on exactly the same mesh.
# The PDE semantics (BCs, gamma) are defined ONCE and reused.

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_file = joinpath(grids_dir, "laplace_unit_5by5.unv")

println()
println("=" ^ 68)
println("LAPLACE — one problem, three computational realizations")
println("Problem: −∇²φ = 0, φ=1 on left, φ=0 on right, ∂φ/∂n=0 top/bottom")
println("Exact solution: φ(x,y) = 1 − x")
println("=" ^ 68)

mesh = UNV2D_mesh(mesh_file)

backend   = CPU()   # ← change to CUDABackend() for GPU
activate_multithread(backend)
workgroup = Threads.nthreads() > 1 ? cld(length(mesh.cells), Threads.nthreads()) : 4
mesh_dev  = adapt(backend, mesh)
n_cells   = length(mesh_dev.cells)
n_bfaces  = length(mesh_dev.boundary_cellsID)
println("  mesh: $(n_cells) cells | $(n_bfaces) boundary faces")

# ── LAYER 1: Semantic PDE definition — defined ONCE, shared by all realizations ──
#
# These objects express the PHYSICS. They do not say anything about:
# - whether a sparse matrix will be assembled
# - which linear solver will be used
# - which hardware backend runs the computation

phi   = ScalarField(mesh_dev);     initialise!(phi,   0.0)
gamma = FaceScalarField(mesh_dev); initialise!(gamma, 1.0)  # diffusivity γ = 1
source = zeros(Float64, n_cells)                             # no source term

# The BCs (Dirichlet, Zerogradient) are semantic objects: they declare the physics.
BCs = assign(region=mesh_dev, (C = [
    Dirichlet(:left_wall,   1.0),   # φ = 1 at x=0
    Dirichlet(:right_wall,  0.0),   # φ = 0 at x=1
    Zerogradient(:upper_wall),      # ∂φ/∂n = 0 (insulated)
    Zerogradient(:bottom_wall),     # ∂φ/∂n = 0 (insulated)
],))

# Shared solver setup — also semantic: "use BiCGSTAB with Jacobi preconditioner"
solver_setup = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                           convergence=1e-10, relax=1.0, rtol=1e-8)

# Helper: set phi to the exact solution φ = 1 − x
function set_exact!(phi)
    for i in 1:length(phi.values)
        phi.values[i] = 1.0 - mesh_dev.cells[i].centre[1]
    end
end

# Helper: reset phi to zero (starting point for solvers)
reset_phi!() = initialise!(phi, 0.0)

# Shared config
config = Configuration(
    hardware   = Hardware(backend=backend, workgroup=workgroup),
    runtime    = Runtime(iterations=200, write_interval=-1, time_step=1.0),
    schemes    = (C = Schemes(laplacian=Linear),),
    solvers    = (C = solver_setup,),
    boundaries = (C = BCs.C,)
)

# =============================================================================
# MATRIX-FREE KERNEL FUNCTIONS (used by Realization B)
# =============================================================================
# These functions implement the same BC physics as @define_boundary, but as
# explicit scatter-add kernels that avoid assembling a sparse matrix.
#
# Design: the Laplacian stencil formula (D_f = gamma*norm(Ef)/delta) is embedded
# in the kernel body — the BC residual depends on both the BC type AND the
# operator. This is why these live here (Laplacian-specific), not in a generic
# "bc_residual" interface on BC types.
#
# For nonlinear BCs that cannot use @define_boundary, see bc_ops.jl.

@kernel function _dirichlet_residual_kernel!(
    r::AbstractArray{F},
    phi::AbstractArray{F},
    phi_bc::F,
    gamma::AbstractArray{F},
    faces,
    cells,
    boundary_cellsID,
    patch_start::Int,
    patch_stop::Int
) where F
    k = @index(Global)
    fID = k + patch_start - 1
    @inbounds begin
        if fID <= patch_stop
            cellID = boundary_cellsID[fID]
            face   = faces[fID]
            (; area, delta, normal, e) = face
            Sf  = area * normal
            Ef  = ((Sf ⋅ Sf) / (Sf ⋅ e)) * e
            D_f = gamma[fID] * norm(Ef) / delta
            Atomix.@atomic r[cellID] += D_f * (phi[cellID] - phi_bc)
        end
    end
end

function _mf_add_bc!(r, phi_vals, gamma_vals, BC::Dirichlet, mesh, backend, workgroup)
    phi_bc = Float64(BC.value)
    ps, pe = BC.IDs_range.start, BC.IDs_range.stop
    ndrange = pe - ps + 1; ndrange == 0 && return
    (; faces, cells, boundary_cellsID) = mesh
    k! = _dirichlet_residual_kernel!(_setup(backend, workgroup, ndrange)...)
    k!(r, phi_vals, phi_bc, gamma_vals, faces, cells, boundary_cellsID, ps, pe)
    KernelAbstractions.synchronize(backend)
end
_mf_add_bc!(r, phi_vals, gamma_vals, ::Zerogradient, mesh, backend, workgroup) = nothing
function _mf_add_bc!(r, phi_vals, gamma_vals, BC, mesh, backend, workgroup)
    @warn "BC type $(typeof(BC)) not implemented in matrix-free path; skipping"
end

function full_residual!(r, phi_vals, gamma_vals, source, BCs::Tuple, mesh, backend, workgroup)
    fill!(r, 0.0)
    laplacian_residual!(r, phi_vals, gamma_vals, source, mesh, backend, workgroup)
    for BC in BCs
        _mf_add_bc!(r, phi_vals, gamma_vals, BC, mesh, backend, workgroup)
    end
end

function full_jvp!(Jv, v, phi_vals, gamma_vals, source, BCs::Tuple, mesh, backend, workgroup;
                   ε = sqrt(eps(eltype(phi_vals))))
    n  = length(phi_vals)
    r0 = zeros(eltype(phi_vals), n)
    r1 = zeros(eltype(phi_vals), n)
    full_residual!(r0, phi_vals,           gamma_vals, source, BCs, mesh, backend, workgroup)
    full_residual!(r1, phi_vals .+ ε .* v, gamma_vals, source, BCs, mesh, backend, workgroup)
    @. Jv = (r1 - r0) / ε
end

# =============================================================================
# REALIZATION A1 — Assembled FV: low-level API
# =============================================================================
# This is the raw FV path. You build the equation object, call discretise! to
# fill the CSR matrix, apply_boundary_conditions! to modify the matrix rows
# at boundary faces, then solve.
#
# Use this when: you need direct access to A and b (custom assembly, debugging,
# computing derived quantities from the stiffness matrix, or understanding
# how the assembled system looks).
#
# What the @define_boundary functors do (hidden inside apply_boundary_conditions!):
#   For each Dirichlet face:  A[cID,cID] += D_f,  b[cID] += D_f * value
#   For each Zerogradient face: no change to A or b (zero normal flux)
# This is "algebraic row modification" — modifying the assembled sparse system.

println()
println("─" ^ 68)
println("REALIZATION A1 — Assembled FV: low-level (discretise! + applyBCs! + krylov)")
println("─" ^ 68)
println("  Same Dirichlet/Zerogradient BCs as the semantic definition above.")
println("  These are lowered into (ap, bp) pairs via @define_boundary functors.")

L_plain = ((-Laplacian{Linear}(gamma)) → BCs.C) → solver_setup
eqn_a1 = L_plain(phi)

reset_phi!()
solve_equation!(eqn_a1, config)

err_a1 = maximum(i -> abs(phi.values[i] - (1 - mesh_dev.cells[i].centre[1])), 1:n_cells)
@printf "  Solution error vs exact φ=1−x:  max|φ−φ_exact| = %.2e\n" err_a1
@test err_a1 < 1e-6
phi_a1 = copy(phi.values)   # save BEFORE second reset below
println("  Realization A1 ✓")

# Re-assemble explicitly (with phi=0) to inspect A and b.
# For the linear Laplacian, A is phi-independent — same matrix regardless of phi.
reset_phi!()
discretise!(eqn_a1, phi, config)
apply_boundary_conditions!(eqn_a1, config)
A = _A(eqn_a1);   b = _b(eqn_a1)
nzval = A.parent.nzval
@printf "  Assembled matrix:  nnz = %d, size = %dx%d\n" length(nzval) n_cells n_cells
@printf "  nzval range:       [%.3f, %.3f]  (positive diagonal = correct sign convention)\n" minimum(nzval) maximum(nzval)

# =============================================================================
# REALIZATION A2 — Assembled FV: PDEOperator DSL
# =============================================================================
# Identical assembled FV path, but expressed via the → DSL.
# L = operator → BCs → SolverSetup builds a reusable PDEOperator.
# Calling L(phi) binds the operator to the field.
#
# The DSL is SYNTACTIC SUGAR over the same CSR assembly.
# It produces EXACTLY the same matrix as Realization A1.
#
# Use this when: you want composable, reusable equation definitions.
# Example: solve the same operator on multiple fields:
#   eqn1 = L(phi1);  eqn2 = L(phi2)
# Or add operators together:
#   L = -Laplacian{Linear}(gamma) + Si(reaction) == Source(f)

println()
println("─" ^ 68)
println("REALIZATION A2 — Assembled FV: PDEOperator DSL (→ chains)")
println("─" ^ 68)
println("  Same assembled path as A1. The → DSL is syntactic sugar,")
println("  not a different execution path.")

L_dsl = ((-Laplacian{Linear}(gamma)) → BCs.C) → solver_setup

reset_phi!()
eqn_a2 = L_dsl(phi)          # bind to field — solver setup embedded in eqn_a2
solve_equation!(eqn_a2, config)

err_a2 = maximum(i -> abs(phi.values[i] - (1 - mesh_dev.cells[i].centre[1])), 1:n_cells)
@printf "  Solution error vs exact φ=1−x:  max|φ−φ_exact| = %.2e\n" err_a2
@test err_a2 < 1e-6

# Verify A1 and A2 give the SAME solution (they are the same path, not alternatives)
phi_a2 = copy(phi.values)
@test maximum(abs, phi_a1 .- phi_a2) < 1e-12
println("  Realization A2 ✓ (identical solution to A1 — same assembled path)")

# =============================================================================
# REALIZATION B — Matrix-free residual evaluation (complementary path)
# =============================================================================
# This realization evaluates R(φ) = A·φ − b without forming the sparse matrix.
# It uses the SAME semantic BCs (Dirichlet, Zerogradient) as Realizations A1/A2.
# The difference is in the lowering:
#
#   Assembled path (A1/A2):
#     @define_boundary Dirichlet Laplacian{Linear} → (ap, bp) → CSR row modify
#
#   Matrix-free path (B):
#     Operator-specific kernel: _dirichlet_residual_kernel! → D_f*(φ_P−value)
#     Zerogradient: zero contribution (no kernel call)
#
# The operator-specific kernel encodes the same D_f formula as @define_boundary.
# There is ONE source of truth: the Laplacian stencil formula. It appears in
# @define_boundary and in the kernel body — consistent by construction.
#
# IMPORTANT: This does NOT solve the equation — it evaluates the residual.
# For solving, combine with the assembled solver (hybrid) or a matrix-free
# Krylov method (JFNK, not yet implemented as a high-level API here).

println()
println("─" ^ 68)
println("REALIZATION B — Matrix-free residual evaluation (operator-specific @kernel)")
println("─" ^ 68)
println("  The SAME semantic BCs lower into operator-specific kernels.")
println("  No CSR matrix is assembled. Same Dirichlet/Zerogradient objects as A1/A2.")

BCs_tuple = BCs.C   # same BC objects, different lowering

# ── B1: R(φ_exact) ≈ 0 ──────────────────────────────────────────────────────
set_exact!(phi)
r_exact = zeros(Float64, n_cells)
full_residual!(r_exact, phi.values, gamma.values, source, BCs_tuple, mesh_dev, backend, workgroup)
@printf "  R(φ_exact):  max|r| = %.2e  (expected ~machine epsilon)\n" maximum(abs, r_exact)
@test maximum(abs, r_exact) < 1e-10

# ── B2: R(φ) matches assembled A·φ − b (both express the same physics) ───────
# Use the reference assembled matrix from A1 for comparison.
for i in 1:n_cells
    x, y = mesh_dev.cells[i].centre[1], mesh_dev.cells[i].centre[2]
    phi.values[i] = sin(π * x) * sin(π * y)   # non-trivial test field
end

r_mf = zeros(n_cells)
full_residual!(r_mf, phi.values, gamma.values, source, BCs_tuple, mesh_dev, backend, workgroup)
r_assembled = Vector(A * phi.values) .- b   # A, b from Realization A1
err_r = maximum(abs, r_mf .- r_assembled)
@printf "  R(φ) matrix-free vs assembled A·φ−b:  max|diff| = %.2e\n" err_r
@test err_r < 1e-10
println("  Both realizations express the same physics ✓")

# ── B3: J(φ)·v via FD-JVP — two full_residual! calls ────────────────────────
using Random; Random.seed!(7)
v = randn(n_cells)
Jv_mf        = zeros(n_cells)
Jv_assembled = Vector(A * v)
full_jvp!(Jv_mf, v, phi.values, gamma.values, source, BCs_tuple, mesh_dev, backend, workgroup)
err_jvp = maximum(abs, Jv_mf .- Jv_assembled)
@printf "  J(φ)·v FD-JVP vs assembled A·v:  max|diff| = %.2e  (O(ε) error)\n" err_jvp
@test err_jvp < 1e-4
println("  FD-JVP matches assembled matvec ✓")

# ── B4: Hybrid use — assembled solver + matrix-free convergence monitor ───────
# The most practical near-term use of the matrix-free path: run the assembled
# solver, but track true residual by calling full_residual! without re-assembling.
# The same full_residual! kernel works on GPU.
println()
println("  Hybrid demo: assembled solver + matrix-free residual monitor")
reset_phi!()
eqn_hybrid = L_dsl(phi)

r_monitor = zeros(n_cells)
for iter in 1:3
    solve_equation!(eqn_hybrid, config)
    full_residual!(r_monitor, phi.values, gamma.values, source, BCs_tuple, mesh_dev, backend, workgroup)
    @printf "    iter %d: max|R(φ)| (matrix-free) = %.3e\n" iter maximum(abs, r_monitor)
end
@test maximum(abs, r_monitor) < 1e-8
println("  Realization B ✓")

# =============================================================================
# SUMMARY
# =============================================================================

println()
println("=" ^ 68)
println("SUMMARY — Same PDE/BC semantics, three computational realizations")
println("=" ^ 68)
println()
println("  PDE SEMANTICS (shared by all realizations):")
println("    Dirichlet(:left_wall, 1.0)    → φ = 1 at x=0")
println("    Dirichlet(:right_wall, 0.0)   → φ = 0 at x=1")
println("    Zerogradient(:upper_wall)     → ∂φ/∂n = 0")
println("    −Laplacian{Linear}(gamma)     → −∇·(γ∇φ)")
println()
println("  ┌────────────────┬──────────────────────────┬─────────────────────────┐")
println("  │ Realization    │ BC lowering              │ Use case                │")
println("  ├────────────────┼──────────────────────────┼─────────────────────────┤")
println("  │ A1: Low-level  │ @define_boundary →       │ Explicit assembly,      │")
println("  │    Assembled   │ (ap,bp) → CSR row modify │ direct access to A, b   │")
println("  ├────────────────┼──────────────────────────┼─────────────────────────┤")
println("  │ A2: DSL        │ Same as A1               │ Composable/reusable     │")
println("  │    Assembled   │ (→ is syntactic sugar)   │ equation definitions    │")
println("  ├────────────────┼──────────────────────────┼─────────────────────────┤")
println("  │ B:  Matrix-    │ operator-specific @kernel│ JFNK inner loop,        │")
println("  │     free       │ → R(φ) via full_residual!│ GPU Newton, monitoring  │")
println("  └────────────────┴──────────────────────────┴─────────────────────────┘")
println()
println("  Realizations A1 and A2 are PRIMARY (production-ready, full operator/BC coverage).")
println("  Realization B is COMPLEMENTARY — specialised for matrix-free/JFNK/GPU contexts.")
println()
println("  For JFNK (next step), plug B into a matrix-free Krylov solver:")
println("    r  = full_residual!(r, φ_k, ...)       # R(φ_k)")
println("    Jv = full_jvp!(Jv, v, φ_k, ...)        # J(φ_k)·v  (two kernel calls)")
println("    # → feed r and Jv into GMRES/BiCGSTAB → matrix-free Newton update")
println()
println("All tests passed.")
