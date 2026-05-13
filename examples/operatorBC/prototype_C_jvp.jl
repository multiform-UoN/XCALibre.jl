# =============================================================================
# PROTOTYPE C: Matrix-Free Jacobian-Vector Products (JVPs)
# =============================================================================
#
# PURPOSE
# -------
# Three JVP implementations for the scalar Laplacian residual R(u):
#
#   PATH 1 — ASSEMBLED (reference):
#     Jv = A * v   (SpMV with pre-assembled CSR matrix)
#     Exact; no FD error. Requires the full sparse matrix in memory.
#
#   PATH 2 — LINEAR-KERNEL JVP:
#     Apply the interior residual kernel to v in place of phi, plus a
#     typed-dispatch BC JVP kernel.
#     Exact for operators where diffusivity D_f is phi-independent.
#     Same cost as one residual evaluation; zero FD error.
#     Not valid for nonlinear operators (NonlinearOperator) where D_f = f(phi).
#
#   PATH 3 — FINITE-DIFFERENCE JVP (universal):
#     Jv ≈ (R(u + ε·v) − R(u)) / ε  via two full_residual! calls.
#     Works unchanged for nonlinear R, NonLinearRobin BCs, etc.
#     GPU-compatible: swap backend = CUDABackend() with no other changes.
#     Error: O(ε·|R''|) ≈ 1e-7 for ε = √eps(Float64).
#
# RELATION TO JFNK AND linearize_physics
# ----------------------------------------
# JFNK inner Krylov loop needs only J(u)·v; it does NOT need the explicit
# Jacobian matrix.  PATH 3 provides this without linearize_physics:
#
#   linearize_physics: CPU pre-pass producing AffineOperator fields, then
#     a fresh discretise! + solve!.  Required to assemble J (or a
#     preconditioner); CPU-only due to ForwardDiff on cell arrays.
#
#   FD-JVP (PATH 3): two kernel launches per Krylov step; no matrix.
#     GPU-compatible today.  Replaces linearize_physics entirely when
#     used inside a matrix-free Krylov solver.
#
# GPU Newton via JFNK staged path:
#   1. (Ready now)  R(u_k) via full_residual! kernel         — GPU-compatible
#   2. (Ready now)  J(u_k)·v via FD-JVP (PATH 3)             — GPU-compatible
#   3. (Near-term)  Diagonal preconditioner from kernel sums  — GPU-compatible
#   4. (Long-term)  Assembled J for ILU/block preconditioner  — needs Enzyme
#                   device-side AD or GPU-parallel sparse fill
#
# NONLINEAR EXTENSION
# --------------------
# For nonlinear operators (D_f = f(phi)):
#   PATH 2 is incorrect — it ignores dD_f/dphi contributions.
#   PATH 3 captures these automatically via perturbation.
#   Future: ForwardDiff dual-number JVP (one pass, exact) requires
#   relaxing the gamma::AbstractArray{F}/phi::AbstractArray{F} type constraint
#   in laplacian_residual_kernel! to separate type parameters.

using XCALibre
using KernelAbstractions
using Atomix
using StaticArrays
using LinearAlgebra
using SparseMatricesCSR
using Test
using Printf

include("prototype_A_laplacian_residual.jl")
include("prototype_B_bc_residual.jl")

# =============================================================================
# BC JVP KERNELS — complement to Prototype B bc_residual kernels
# =============================================================================
# For Dirichlet: r_bc[cID] = D_f * (phi[cID] − phi_bc)
#                dr_bc/dphi[cID] = D_f   (constant; phi_bc is prescribed)
#                (J·v)[cID] += D_f * v[cID]
#
# For Zerogradient: r_bc = 0  →  (J·v)[cID] += 0  (nothing to do)
#
# Design: same typed-tuple static dispatch as Prototype B.
# No Vector{AbstractBoundaryAction}, no runtime polymorphism.

@kernel function dirichlet_bc_jvp_kernel!(
    Jv::AbstractArray{F},
    v::AbstractArray{F},
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
            # J[cID,cID] contribution = D_f → (J·v)[cID] += D_f * v[cID]
            Atomix.@atomic Jv[cellID] += D_f * v[cellID]
        end
    end
end

function dirichlet_bc_jvp!(Jv, v, gamma_vals, mesh, bc_range, backend, workgroup)
    patch_start = bc_range.start
    patch_stop  = bc_range.stop
    ndrange     = patch_stop - patch_start + 1
    ndrange == 0 && return
    (; faces, cells, boundary_cellsID) = mesh
    kernel! = dirichlet_bc_jvp_kernel!(_setup(backend, workgroup, ndrange)...)
    kernel!(Jv, v, gamma_vals, faces, cells, boundary_cellsID, patch_start, patch_stop)
    KernelAbstractions.synchronize(backend)
end

function add_bc_jvps!(Jv, v, gamma_vals, BCs::Tuple, mesh, backend, workgroup)
    for BC in BCs
        _add_bc_jvp!(Jv, v, gamma_vals, BC, mesh, backend, workgroup)
    end
end

_add_bc_jvp!(Jv, v, gamma_vals, BC::Dirichlet, mesh, backend, workgroup) =
    dirichlet_bc_jvp!(Jv, v, gamma_vals, mesh, BC.IDs_range, backend, workgroup)

_add_bc_jvp!(Jv, v, gamma_vals, ::Zerogradient, mesh, backend, workgroup) = nothing

_add_bc_jvp!(Jv, v, gamma_vals, BC, mesh, backend, workgroup) =
    @warn "BC type $(typeof(BC)) not implemented for JVP in prototype_C; skipping"

# =============================================================================
# PATH 2: LINEAR-KERNEL JVP
# =============================================================================
# Applies the interior residual kernel to v instead of phi.
# Exact for Laplacian{Linear} where D_f = gamma * area/delta is phi-independent.
# NOT valid for NonlinearOperator where D_f = f(phi): use PATH 3 instead.

function linear_kernel_jvp!(Jv, v, gamma_vals, BCs, mesh, backend, workgroup)
    fill!(Jv, 0)
    source_zero = zeros(eltype(v), length(v))
    # Interior: same kernel but with v as the field (source contribution is zero
    # because the source term does not depend on phi for standard Laplacian)
    laplacian_residual!(Jv, v, gamma_vals, source_zero, mesh, backend, workgroup)
    # BC: add D_f * v[cellID] for each Dirichlet face
    add_bc_jvps!(Jv, v, gamma_vals, BCs, mesh, backend, workgroup)
end

# =============================================================================
# PATH 3: FINITE-DIFFERENCE JVP (universal)
# =============================================================================
# Jv ≈ (R(u + ε·v) − R(u)) / ε
# Works for any R: linear, nonlinear, NonLinearRobin BCs.
# GPU-compatible: both full_residual! calls are @kernel functions.
#
# Optimal step choice (Hager-Zhang): ε = √eps(F) ≈ 1.49e-8 for Float64.
# For complex-step accuracy (no cancellation error), use ε in the imaginary
# direction: not implemented here but straightforward.

function fd_jvp!(Jv, v, phi_vals, gamma_vals, source, BCs, mesh, backend, workgroup;
                 ε = sqrt(eps(eltype(phi_vals))))
    n   = length(phi_vals)
    r0  = zeros(eltype(phi_vals), n)
    r1  = zeros(eltype(phi_vals), n)
    full_residual!(r0, phi_vals,           gamma_vals, source, BCs, mesh, backend, workgroup)
    full_residual!(r1, phi_vals .+ ε .* v, gamma_vals, source, BCs, mesh, backend, workgroup)
    @. Jv = (r1 - r0) / ε
end

# =============================================================================
# VALIDATION
# =============================================================================

println()
println("=" ^ 60)
println("PROTOTYPE C — Matrix-Free JVPs")
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

# Non-trivial phi profile to exercise nonzero residual and JVP
for i in 1:n_cells
    x, y = mesh_dev.cells[i].centre[1], mesh_dev.cells[i].centre[2]
    phi.values[i] = sin(π * x) * sin(π * y)
end

BCs_eqn = assign(region=mesh_dev, (C = [
    Dirichlet(:left_wall,   0.0), Dirichlet(:right_wall,  0.0),
    Dirichlet(:upper_wall,  0.0), Dirichlet(:bottom_wall, 0.0),
],))
BCs_tuple = Tuple(BCs_eqn.C)

# Assemble reference matrix (for PATH 1 and verification)
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

# Use a fixed random direction v for reproducibility
using Random; Random.seed!(42)
v = randn(n_cells)

# -----------------------------------------------------------------------
# PATH 1: assembled Jv = A * v  (reference)
# -----------------------------------------------------------------------
Jv_assembled = Vector(A_ref * v)

# -----------------------------------------------------------------------
# PATH 2: linear-kernel JVP
# -----------------------------------------------------------------------
Jv_linear = zeros(n_cells)
linear_kernel_jvp!(Jv_linear, v, gamma.values, BCs_tuple, mesh_dev, backend, workgroup)

# -----------------------------------------------------------------------
# PATH 3: FD JVP
# -----------------------------------------------------------------------
Jv_fd = zeros(n_cells)
fd_jvp!(Jv_fd, v, phi.values, gamma.values, source, BCs_tuple, mesh_dev, backend, workgroup)

# -----------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------
println()
println("  Test vector v: randn($(n_cells)), seed=42")
println()

err_linear = maximum(abs, Jv_linear .- Jv_assembled)
err_fd     = maximum(abs, Jv_fd     .- Jv_assembled)
ε_used     = sqrt(eps(Float64))

@printf "  PATH 1 assembled Jv   (reference):  max|Jv| = %.4e\n" maximum(abs, Jv_assembled)
@printf "  PATH 2 linear-kernel: max|Jv2-Jv1| = %.2e  (expected: ~1e-15)\n" err_linear
@printf "  PATH 3 FD-JVP (ε=%.1e): max|Jv3-Jv1| = %.2e  (expected: ~%.0e)\n" ε_used err_fd ε_used

println()
@test err_linear < 1e-10    # exact for linear operator
@test err_fd     < 1e-4     # O(ε) FD error

println("  Both JVP paths agree with assembled A*v ✓")
println()
println("  GPU READINESS:")
println("    PATH 2 (linear-kernel): GPU-ready — laplacian_residual! + dirichlet_bc_jvp!")
println("      both are @kernel with typed tuple BC dispatch")
println("    PATH 3 (FD-JVP):        GPU-ready — two full_residual! calls")
println("      works for nonlinear R; ε·v perturbation is a BLAS-1 operation")
println("    PATH 1 (assembled):     GPU-runnable (CUSPARSE SpMV) but needs matrix in memory")
println()
println("  JFNK SUMMARY:")
println("    Inner Krylov loop: PATH 3 only. No linearize_physics, no sparse assembly.")
println("    Outer Newton update: R(u_k) from full_residual! (already verified in Prototype B).")
println("    Preconditioner (optional): diagonal of J = Σ D_f per cell; computable by kernel.")
println("    Assembled preconditioner (ILU/block): still needs linearize_physics (CPU-only).")

println()
println("All Prototype C tests PASSED")
