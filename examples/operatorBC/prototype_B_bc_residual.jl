# =============================================================================
# PROTOTYPE B: Static BC Residual Kernel (GPU-compatible BC path)
# =============================================================================
#
# PURPOSE
# -------
# Demonstrate how boundary condition residual contributions can be expressed
# as GPU-compatible kernels WITHOUT using Vector{AbstractBoundaryAction} in
# the hot loop — the primary GPU-hostile pattern in Discretise_7.
#
# KEY INSIGHT
# -----------
# The existing apply_boundary_conditions_kernel! (Discretise_5) is already
# a GPU @kernel.  It uses a tuple of BC objects (type-parameterised) and
# dispatches via a for-loop with range checking.  This IS the correct GPU
# pattern.
#
# What is NOT GPU-compatible:
#   - apply_boundary_actions!(A::SparseMatrixCSC, b, actions::Vector{AbstractAction})
#     in Discretise_7: SparseMatrixCSC row surgery, runtime-polymorphic actions
#   - The action vocabulary is explicitly CPU-only (documented in Discretise_7)
#
# This prototype shows:
#   (A) A standalone Dirichlet BC residual kernel — type-safe, no dynamic dispatch
#   (B) A Zerogradient (zero-Neumann) BC residual kernel — trivially zero
#   (C) A dispatcher that adds BC residuals to an existing interior residual r
#       using static dispatch over a BC tuple (same pattern as apply_boundary_conditions_kernel!)
#
# RESIDUAL SEMANTICS
# ------------------
# For assembled system: A_full * phi - b_full = r_full
#   A_full = A_interior + A_bc
#   b_full = b_interior + b_bc
#
# For Dirichlet BC (Laplacian term, face fID → cell cellID):
#   A_bc[cellID, cellID] += D_f   (from @define_boundary Dirichlet Laplacian{Linear})
#   b_bc[cellID]         += D_f * phi_bc
#   → BC residual: r_bc[cellID] += D_f * (phi[cellID] - phi_bc)
#
# For Zerogradient: ap=0, bp=0 → zero BC residual

using XCALibre
using KernelAbstractions
using Atomix
using StaticArrays
using LinearAlgebra
using Test

# =============================================================================
# KERNEL A: Dirichlet BC residual contribution
# =============================================================================
# Iterates over ONE patch's boundary faces; one thread per face.
# Using Atomix.@atomic for thread safety (multiple faces can share a cell
# on non-structured meshes).
# phi_bc is passed as a scalar type parameter — zero heap allocation.

@kernel function dirichlet_bc_residual_kernel!(
    r::AbstractArray{F},
    phi::AbstractArray{F},
    phi_bc::F,
    gamma::AbstractArray{F},     # face diffusivity [n_faces_total]
    faces,
    cells,
    boundary_cellsID,
    patch_start::Int,
    patch_stop::Int
) where F
    k = @index(Global)            # 1-based index within the patch
    fID = k + patch_start - 1
    @inbounds begin
        if fID <= patch_stop
            cellID = boundary_cellsID[fID]
            face   = faces[fID]
            (; area, delta, normal, e) = face
            # ns = +1 for boundary faces (always "owner" side)
            Sf  = area * normal
            Ef  = ((Sf ⋅ Sf) / (Sf ⋅ e)) * e
            D_f = gamma[fID] * norm(Ef) / delta
            # Dirichlet BC: A[cellID,cellID] += D_f, b[cellID] += D_f*phi_bc
            # → residual contribution: r[cellID] += D_f*(phi_P - phi_bc)
            Atomix.@atomic r[cellID] += D_f * (phi[cellID] - phi_bc)
        end
    end
end

function dirichlet_bc_residual!(r, phi_vals, phi_bc, gamma_vals, mesh, bc_range, backend, workgroup)
    patch_start = bc_range.start
    patch_stop  = bc_range.stop
    ndrange     = patch_stop - patch_start + 1
    ndrange == 0 && return
    (; faces, cells, boundary_cellsID) = mesh
    kernel! = dirichlet_bc_residual_kernel!(_setup(backend, workgroup, ndrange)...)
    kernel!(r, phi_vals, phi_bc, gamma_vals, faces, cells, boundary_cellsID, patch_start, patch_stop)
    KernelAbstractions.synchronize(backend)
end

# =============================================================================
# KERNEL B: Zerogradient (zero-Neumann) BC residual
# =============================================================================
# For Zerogradient, ap = 0 and bp = 0, so the BC residual contribution is
# exactly zero.  No kernel is needed — included here for documentation.
#
# function zerogradient_bc_residual!(r, ...) = nothing  (zero contribution)

# =============================================================================
# DISPATCHER: static-dispatch BC residual over all patches
# =============================================================================
# Uses the same pattern as apply_boundary_conditions_kernel! — BC tuple is
# type-parameterised so dispatch is resolved at compile time, not runtime.
# This avoids the Vector{AbstractBoundaryAction} GPU-hostile pattern.

function add_bc_residuals!(r, phi_vals, gamma_vals, BCs::Tuple, mesh, backend, workgroup)
    for BC in BCs
        _add_bc_residual!(r, phi_vals, gamma_vals, BC, mesh, backend, workgroup)
    end
end

# Dirichlet — known at compile time via type dispatch
function _add_bc_residual!(r, phi_vals, gamma_vals, BC::Dirichlet, mesh, backend, workgroup)
    phi_bc = Float64(BC.value)  # scalar Dirichlet value
    dirichlet_bc_residual!(r, phi_vals, phi_bc, gamma_vals, mesh, BC.IDs_range, backend, workgroup)
end

# Zerogradient — zero contribution
function _add_bc_residual!(r, phi_vals, gamma_vals, BC::Zerogradient, mesh, backend, workgroup)
    nothing
end

# Fallback: other BCs not yet implemented in this prototype
function _add_bc_residual!(r, phi_vals, gamma_vals, BC, mesh, backend, workgroup)
    @warn "BC type $(typeof(BC)) not implemented in prototype_B; skipping"
end

# =============================================================================
# FULL RESIDUAL = INTERIOR + BC
# =============================================================================

function full_residual!(r_full, phi_vals, gamma_vals, source, BCs, mesh, backend, workgroup)
    fill!(r_full, 0.0)
    # Interior contribution (Prototype A kernel)
    (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh
    ndrange = length(mesh.cells)
    kernel! = laplacian_residual_kernel!(_setup(backend, workgroup, ndrange)...)
    kernel!(r_full, phi_vals, gamma_vals, source, faces, cells, cell_faces, cell_neighbours, cell_nsign)
    KernelAbstractions.synchronize(backend)
    # Boundary contribution (this file)
    add_bc_residuals!(r_full, phi_vals, gamma_vals, BCs, mesh, backend, workgroup)
end

# Import interior kernel from Prototype A
include("prototype_A_laplacian_residual.jl")

# =============================================================================
# VALIDATION
# =============================================================================

println()
println("=" ^ 60)
println("PROTOTYPE B — Static BC Residual Kernel")
println("=" ^ 60)

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_file = joinpath(grids_dir, "laplace_unit_3by3.unv")
mesh = UNV2D_mesh(mesh_file)
backend   = CPU()
workgroup = 4
mesh_dev  = adapt(backend, mesh)
n_cells   = length(mesh_dev.cells)

phi   = ScalarField(mesh_dev);    initialise!(phi,   0.0)
gamma = FaceScalarField(mesh_dev); initialise!(gamma, 1.0)
source = zeros(Float64, n_cells)

# Set phi to linear profile x
for i in 1:n_cells
    phi.values[i] = mesh_dev.cells[i].centre[1]
end

# Build BCs (same as Prototype A)
BCs_eqn = assign(region=mesh_dev, (C = [
    Dirichlet(:left_wall, 0.0), Dirichlet(:right_wall, 1.0),
    Zerogradient(:upper_wall), Zerogradient(:bottom_wall)
],))
BCs_tuple = Tuple(BCs_eqn.C)

# --- Test 1: interior-only residual (Prototype A) ---
r_interior = zeros(Float64, n_cells)
(; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh_dev
ndrange = length(mesh_dev.cells)
kernel! = laplacian_residual_kernel!(_setup(backend, workgroup, ndrange)...)
kernel!(r_interior, phi.values, gamma.values, source, faces, cells, cell_faces, cell_neighbours, cell_nsign)
KernelAbstractions.synchronize(backend)

# --- Test 2: full residual (interior + BC) ---
r_full = zeros(Float64, n_cells)
full_residual!(r_full, phi.values, gamma.values, source, BCs_tuple, mesh_dev, backend, workgroup)

# --- Test 3: assembled path residual for comparison ---
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
A_csr  = _A(eqn)
b_csr  = _b(eqn)
r_assembled = Vector(A_csr * phi.values) .- b_csr

println("\nTest: phi = x (exact solution for Laplace + Dirichlet)")
println("  Interior residual max |r|: ", maximum(abs, r_interior))
println("  Full residual   max |r|:   ", maximum(abs, r_full))
println("  Assembled path  max |r|:   ", maximum(abs, r_assembled))
println("  |full - assembled| max:    ", maximum(abs, r_full .- r_assembled))

# For the exact solution phi=x, the full residual should be ~0
@test maximum(abs, r_full) < 1e-10
@test maximum(abs, r_full .- r_assembled) < 1e-10

println("\nGPU COMPATIBILITY NOTE:")
println("  - Dirichlet kernel uses Atomix.@atomic (GPU-safe)")
println("  - BCs dispatched via typed tuple — zero runtime polymorphism")
println("  - No Vector{AbstractBoundaryAction} in hot loop")
println("  - The _add_bc_residual! dispatch is resolved at compile time")
println("  - Contrast with Discretise_7 apply_boundary_actions!:")
println("    that path uses SparseMatrixCSC mutation and is CPU-only by design")
println("\nAll Prototype B tests PASSED")
