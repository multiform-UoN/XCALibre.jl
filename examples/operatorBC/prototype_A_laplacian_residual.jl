# =============================================================================
# PROTOTYPE A: Matrix-Free Scalar Laplacian Residual Kernel
# =============================================================================
#
# PURPOSE
# -------
# Demonstrate a KernelAbstractions-native FV residual kernel for scalar
# diffusion that:
#   - allocates no sparse matrix
#   - runs cell-parallel (one thread per cell)
#   - has an allocation-free inner face loop
#   - is GPU-portable: swap backend = CUDABackend() / ROCmBackend()
#
# ARCHITECTURE OBSERVATION (verified by inspection)
# --------------------------------------------------
# In XCALibre's mesh topology, `cell.faces_range` contains ONLY interior
# face indices.  Boundary faces are handled in a separate BC kernel pass
# (see prototype_B_bc_residual.jl for the complementary piece).
# This clean separation means the interior assembly kernel needs no
# boundary-face guard — it is the correct GPU-safe design.
#
# COMPARISON WITH EXISTING PATH
# ------------------------------
# The existing explicit_residual! (`Discretise_2`, lines 453-490) is already
# a GPU kernel.  This prototype is a *lower-level* version that:
#   - bypasses @generated operator dispatch (for clarity / profiling)
#   - uses raw face geometry directly (gamma as a face field)
#   - makes the GPU-friendly structure explicit for documentation purposes
#
# LIMITATION
# ----------
# Computes interior residual only.  Boundary BC contributions are zero here
# and must be added by a separate BC residual kernel (Prototype B).
# The combination (Prototype A + B) equals the assembled-path residual.

using XCALibre
using KernelAbstractions
using Atomix
using StaticArrays
using LinearAlgebra
using Test

# =============================================================================
# KERNEL: direct scalar Laplacian residual over interior faces
# =============================================================================

@kernel function laplacian_residual_kernel!(
    r::AbstractArray{F},       # output: interior residual  r[i] = Σ_f D_f(φ_N-φ_P) - b[i]
    phi::AbstractArray{F},     # field values  [n_cells]
    gamma::AbstractArray{F},   # face diffusivity  [n_faces]
    source::AbstractArray{F},  # cell source term  [n_cells]  (positive = RHS)
    faces,
    cells,
    cell_faces,
    cell_neighbours,
    cell_nsign
) where F
    i = @index(Global)
    @inbounds begin
        cell = cells[i]
        R = zero(F)
        for fi in cell.faces_range
            fID  = cell_faces[fi]
            nID  = cell_neighbours[fi]
            ns   = cell_nsign[fi]
            face = faces[fID]
            (; area, delta, normal, e) = face
            # Over-relaxed diffusion area vector — matches Laplacian{Linear} in Discretise_1
            Sf    = ns * area * normal
            e_ns  = ns * e
            Ef    = ((Sf ⋅ Sf) / (Sf ⋅ e_ns)) * e_ns
            D_f   = gamma[fID] * norm(Ef) / delta
            # Sign convention: XCALibre uses -Laplacian on LHS, so the matrix has
            # A[i,i] = +ΣD_f and A[i,j] = -D_f.
            # r = A·φ - b gives r[i] = ΣD_f·(φ_P - φ_N) - b[i].
            R    += D_f * (phi[i] - phi[nID])
        end
        # r = A*phi - b  →  interior contribution (matches -Laplacian convention)
        r[i] = R - source[i]
    end
end

function laplacian_residual!(r, phi_vals, gamma_vals, source, mesh, backend, workgroup)
    (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh
    ndrange = length(mesh.cells)
    kernel! = laplacian_residual_kernel!(_setup(backend, workgroup, ndrange)...)
    kernel!(r, phi_vals, gamma_vals, source, faces, cells, cell_faces, cell_neighbours, cell_nsign)
    KernelAbstractions.synchronize(backend)
end

# =============================================================================
# VALIDATION
# =============================================================================

println("=" ^ 60)
println("PROTOTYPE A — Matrix-Free Laplacian Residual Kernel")
println("=" ^ 60)

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_file = joinpath(grids_dir, "laplace_unit_3by3.unv")
mesh = UNV2D_mesh(mesh_file)
backend  = CPU()
workgroup = 4
mesh_dev  = adapt(backend, mesh)

n_cells = length(mesh_dev.cells)
n_faces = length(mesh_dev.faces)

phi   = ScalarField(mesh_dev);    initialise!(phi,   1.0)
gamma = FaceScalarField(mesh_dev); initialise!(gamma, 1.0)

source = zeros(Float64, n_cells)
r_A    = zeros(Float64, n_cells)

# --- Test 1: uniform phi → all interior fluxes cancel → r = 0 ---
laplacian_residual!(r_A, phi.values, gamma.values, source, mesh_dev, backend, workgroup)

println("\nTest 1: uniform phi=1, gamma=1, source=0")
println("  max |r| = ", maximum(abs, r_A), "  (expected: 0)")
@test maximum(abs, r_A) < 1e-12

# --- Test 2: cross-check against existing explicit_residual! ---
# Both compute interior-only residual, so they must agree exactly.
# Note: phi=x does NOT give r=0 here because boundary face contributions
# (the Dirichlet BCs) are excluded — cell.faces_range has interior faces only.
# The full zero residual only holds when BC contributions are included (Prototype B).
using Accessors
for i in 1:n_cells
    phi.values[i] = sin(π * mesh_dev.cells[i].centre[1])
end

BCs_eqn = assign(region=mesh_dev, (C = [
    Dirichlet(:left_wall, 0.0), Dirichlet(:right_wall, 0.0),
    Zerogradient(:upper_wall), Zerogradient(:bottom_wall)
],))

L = ((-Laplacian{Linear}(gamma)) → BCs_eqn.C) → SolverSetup(
    solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0
)
eqn = L(phi)

config = Configuration(
    hardware=Hardware(backend=backend, workgroup=workgroup),
    runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0),
    schemes=(C=Schemes(laplacian=Linear),),
    solvers=(C=nothing,),
    boundaries=(C=BCs_eqn.C,)
)

r_existing = zeros(Float64, n_cells)
explicit_residual!(r_existing, eqn, phi, config)

laplacian_residual!(r_A, phi.values, gamma.values, source, mesh_dev, backend, workgroup)

println("\nTest 2: cross-check vs existing explicit_residual! (sin(πx) field)")
println("  Prototype A interior |r|    = ", maximum(abs, r_A))
println("  existing explicit_residual! = ", maximum(abs, r_existing))
println("  max |difference|            = ", maximum(abs, r_A .- r_existing))
# Both are interior-only (cell.faces_range excludes boundary faces).
# They should agree exactly for any field.
@test maximum(abs, r_A .- r_existing) < 1e-10

println("\nGPU COMPATIBILITY NOTE:")
println("  - No sparse matrix allocation (key GPU advantage)")
println("  - No heap allocations in kernel body")
println("  - Cell loop is embarrassingly parallel — no race conditions")
println("  - To run on GPU: change backend to CUDABackend() or ROCmBackend()")
println("    and ensure phi/gamma arrays are on the GPU (adapt will handle this)")
println("  - Limitation: does NOT include BC residual contributions")
println("    (boundary face residuals must be added separately — see prototype_B)")
println("\nAll Prototype A tests PASSED")
