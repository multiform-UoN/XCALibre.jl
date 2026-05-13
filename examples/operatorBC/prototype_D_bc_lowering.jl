# =============================================================================
# PROTOTYPE D: BC Residual/JVP Interface
# =============================================================================
#
# PURPOSE
# -------
# Contrast two BC lowering strategies for Newton/JFNK systems:
#   1. Discretise_7: CPU Newton correction (row replacement, closure-based)
#   2. bc_residual interface: kernel-compatible residual evaluation (scatter-add)
#
# Both lower the SAME semantic BC objects (Dirichlet, Zerogradient, Robin).
# The difference is in how the lowering is performed and what it produces.
#
# DISCRETISE_7 PATTERN (CPU-only)
# --------------------------------
#   LocalScalarResidualBC(row; residual=r_func, jacobian=j_func)
#   boundary_actions(bc, values) → (SetNewtonRow(row, J(u), R(u)),)
#   apply_boundary_actions!(A_csc, b, actions)   # SparseMatrixCSC row surgery
#
#   - Closures embedded in structs (no GPU kernel dispatch)
#   - Returns Vector{AbstractBoundaryAction} (runtime polymorphism)
#   - SparseMatrixCSC row replacement — CPU-only backend
#   - Semantics: ROW REPLACEMENT — zeros BC row, sets J[row,row] and b[row]
#     Appropriate for assembled Newton correction systems J·δu = −R.
#
# BC_RESIDUAL INTERFACE (this prototype / bc_ops.jl)
# ---------------------------------------------------
#   bc_residual(bc::Dirichlet, phi_P, face, gamma_f) → scalar
#   bc_jvp_coeff(bc::Dirichlet, phi_P, face, gamma_f) → scalar
#   @kernel bc_residual_kernel!(r, phi, ..., bc::T, ...) → GPU scatter-add
#
#   - Methods on the EXISTING semantic BC types — no new wrapper types
#   - Typed tuple dispatch resolved at compile time
#   - GPU-compatible: @kernel + Atomix.@atomic
#   - Semantics: SCATTER-ADD — r[cID] += bc_residual(bc, ...)
#     Appropriate for matrix-free residual evaluation and JFNK inner loop.
#   - SAME BC objects usable in assembled path (via @define_boundary) AND
#     in this matrix-free path — the calling context determines the lowering.
#
# SEMANTIC NOTE
# --------------
# Discretise_7's SetNewtonRow replaces the entire row with the BC condition
# (strong Dirichlet imposition in the Newton system).  The scatter-add kernel
# adds the BC flux residual to existing interior contributions (weak flux
# imposition, same as the primary FV kernel).
# Both give correct physics at convergence for linear BCs.
# The difference is in the system structure: row replacement gives a sparser,
# well-conditioned BC block; scatter-add gives a consistent residual formulation
# compatible with matrix-free Krylov solvers.
#
# NONLINEAR BC EXTENSION
# -----------------------
# For NonLinearRobin (f(phi_P) flux condition):
#   bc_jvp_coeff needs ∂(bc_residual)/∂phi_P.  Options:
#     A. FD scalar:  (bc_residual(bc, phi_P+ε, face, γ) - bc_residual(...)) / ε
#        Allocation-free, device-safe, O(ε) accurate.
#     B. Analytical: provide derivative via NonlinearMap(f, df) at construction.
#     C. Enzyme.autodiff scalar on CPU (one-line drop-in).
#   All three keep bc_jvp_coeff type-stable and kernel-compatible.

using XCALibre
using KernelAbstractions
using Atomix
using StaticArrays
using LinearAlgebra
using SparseMatricesCSR
using Test
using Printf

include("prototype_A_laplacian_residual.jl")
include("bc_ops.jl")

# =============================================================================
# VALIDATION
# =============================================================================

println()
println("=" ^ 60)
println("PROTOTYPE D — BC Residual/JVP Interface (bc_residual on existing types)")
println("=" ^ 60)

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_file = joinpath(grids_dir, "laplace_unit_3by3.unv")
mesh      = UNV2D_mesh(mesh_file)
backend   = CPU()
workgroup = 4
mesh_dev  = adapt(backend, mesh)
n_cells   = length(mesh_dev.cells)

phi    = ScalarField(mesh_dev);     initialise!(phi,   0.0)
gamma  = FaceScalarField(mesh_dev); initialise!(gamma, 1.0)
source = zeros(Float64, n_cells)

for i in 1:n_cells
    x, y = mesh_dev.cells[i].centre[1], mesh_dev.cells[i].centre[2]
    phi.values[i] = sin(π * x) * sin(π * y)
end

BCs_eqn = assign(region=mesh_dev, (C = [
    Dirichlet(:left_wall,  0.0), Dirichlet(:right_wall,  0.0),
    Dirichlet(:upper_wall, 0.0), Dirichlet(:bottom_wall, 0.0),
],))

# Assemble reference
L = ((-Laplacian{Linear}(gamma)) → BCs_eqn.C) → SolverSetup(
    solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0)
eqn = L(phi)
config = Configuration(
    hardware=Hardware(backend=backend, workgroup=workgroup),
    runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0),
    schemes=(C=Schemes(laplacian=Linear),),
    solvers=(C=nothing,),
    boundaries=(C=BCs_eqn.C,)
)
discretise!(eqn, phi, config)
apply_boundary_conditions!(eqn, config)
A_ref = _A(eqn)

# Use semantic BC objects directly — no wrapper types needed
bc_ops = BCs_eqn.C

using Random; Random.seed!(42)
v = randn(n_cells)

# -----------------------------------------------------------------------
# Test 1: bc_residual(Dirichlet) == LocalScalarResidualBC.residual
# -----------------------------------------------------------------------
# Manually compute D_f for first Dirichlet BC face, compare scalar values.
BC1      = BCs_eqn.C[1]   # the semantic Dirichlet object IS the bc used in kernels
fID1     = BC1.IDs_range.start
face1    = mesh_dev.faces[fID1]
cID1     = mesh_dev.boundary_cellsID[fID1]
gamma_f1 = gamma.values[fID1]
phi_P1   = phi.values[cID1]
phi_bc1  = Float64(BC1.value)

r_op   = bc_residual(BC1, phi_P1, face1, gamma_f1)
lbc = LocalScalarResidualBC(cID1;
    residual = phi_P -> begin
        Sf = face1.area * face1.normal
        Ef = ((Sf ⋅ Sf) / (Sf ⋅ face1.e)) * face1.e
        D_f = gamma_f1 * norm(Ef) / face1.delta
        D_f * (phi_P - phi_bc1)
    end,
    jacobian = phi_P -> begin
        Sf = face1.area * face1.normal
        Ef = ((Sf ⋅ Sf) / (Sf ⋅ face1.e)) * face1.e
        gamma_f1 * norm(Ef) / face1.delta
    end
)
r_d7 = lbc.residual(phi_P1)

println()
println("  Test 1: bc_residual(Dirichlet) == LocalScalarResidualBC.residual")
@printf "    Dirichlet:    bc_residual  = %.8e\n" r_op
@printf "    Discretise_7: lbc.residual = %.8e\n" r_d7
@test r_op ≈ r_d7
println("    Match ✓")

# -----------------------------------------------------------------------
# Test 2: bc_jvp_coeff(Dirichlet) == LocalScalarResidualBC.jacobian
# -----------------------------------------------------------------------
coeff_op = bc_jvp_coeff(BC1, phi_P1, face1, gamma_f1)
jac_d7   = lbc.jacobian(phi_P1)

println()
println("  Test 2: bc_jvp_coeff(Dirichlet) == LocalScalarResidualBC.jacobian")
@printf "    Dirichlet:    bc_jvp_coeff = %.8e\n" coeff_op
@printf "    Discretise_7: lbc.jacobian  = %.8e\n" jac_d7
@test coeff_op ≈ jac_d7
println("    Match ✓")

# -----------------------------------------------------------------------
# Test 3: residual_ops! agrees with assembled A*phi - b
# -----------------------------------------------------------------------
r_ops   = zeros(n_cells)
residual_ops!(r_ops, phi.values, gamma.values, source, bc_ops, mesh_dev, backend, workgroup)
r_ref   = Vector(A_ref * phi.values) .- _b(eqn)
err_res = maximum(abs, r_ops .- r_ref)

println()
@printf "  Test 3: residual_ops! vs assembled A*phi - b: max|diff| = %.2e\n" err_res
@test err_res < 1e-10
println("    Agreement to floating-point precision ✓")

# -----------------------------------------------------------------------
# Test 4: jvp_ops! (FD-JVP) agrees with assembled A*v
# -----------------------------------------------------------------------
Jv_ops       = zeros(n_cells)
Jv_assembled = Vector(A_ref * v)
jvp_ops!(Jv_ops, v, phi.values, gamma.values, source, bc_ops, mesh_dev, backend, workgroup)
err_jvp = maximum(abs, Jv_ops .- Jv_assembled)
ε_step  = sqrt(eps(Float64))

println()
@printf "  Test 4: jvp_ops! (FD-JVP, ε=%.1e) vs assembled A*v: max|diff| = %.2e\n" ε_step err_jvp
@test err_jvp < 1e-4
println("    Agreement to O(ε) ✓")

# -----------------------------------------------------------------------
# Test 5: Zerogradient — confirm zero contribution
# -----------------------------------------------------------------------
BCs_mixed = assign(region=mesh_dev, (C = [
    Dirichlet(:left_wall, 0.0), Dirichlet(:right_wall, 1.0),
    Zerogradient(:upper_wall), Zerogradient(:bottom_wall)
],))
bc_ops_mixed = BCs_mixed.C   # use semantic objects directly
# phi = x is exact solution → residual must be ~0
for i in 1:n_cells
    phi.values[i] = mesh_dev.cells[i].centre[1]
end
r_mixed = zeros(n_cells)
residual_ops!(r_mixed, phi.values, gamma.values, source, bc_ops_mixed, mesh_dev, backend, workgroup)

println()
@printf "  Test 5: phi=x exact solution with Dirichlet+Zerogradient: max|r| = %.2e\n" maximum(abs, r_mixed)
@test maximum(abs, r_mixed) < 1e-10
println("    Exact solution has zero residual ✓")

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
println()
println("  COMPARISON: Discretise_7 (assembled) vs bc_residual interface (matrix-free)")
println()
println("  DISCRETISE_7:                         BC RESIDUAL INTERFACE (bc_ops.jl):")
println("  ──────────────────────────────────    ─────────────────────────────────────")
println("  LocalScalarResidualBC (closures)      bc_residual(bc::Dirichlet, ...)      ")
println("  boundary_actions() → alloc Tuple      @inline methods on existing BC types ")
println("  Vector{AbstractBoundaryAction}        Typed Tuple → compile-time dispatch  ")
println("  apply_boundary_actions!(A_csc, b)     bc_residual_kernel! (@kernel)        ")
println("  SparseMatrixCSC row surgery           Atomix.@atomic scatter-add           ")
println("  CPU-only                              GPU-compatible today                 ")
println("  Semantics: row replacement            Semantics: scatter-add (flux conv.)  ")
println("  Use: assemble J·δu = −R               Use: matrix-free R(u) and J(u)·v    ")
println()
println("  SAME semantic BC objects (Dirichlet, Zerogradient) used in BOTH paths.")
println("  The lowering choice is made by the kernel/function, not the BC type.")
println()
println("  Robin extension:")
println("    bc_jvp_coeff(bc::Robin, ...) — analytical derivative, allocation-free")
println()
println("All Prototype D tests PASSED")
