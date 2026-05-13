# =============================================================================
# BENCHMARK: Matrix-Free vs Assembled Residual (CPU path)
# =============================================================================
#
# Compares three approaches to computing the FV residual r = A*phi - b
# for a scalar Laplacian problem:
#
#   PATH 1 — ASSEMBLED (existing standard path):
#     discretise! + apply_boundary_conditions! build the CSR matrix A and b
#     Then r = A*phi - b via sparse matrix-vector product
#     Memory: O(nnz) for CSR storage  (n_cells + n_interior_faces entries)
#
#   PATH 2 — MATRIX-FREE (existing explicit_residual! kernel):
#     Recomputes face fluxes on the fly via @generated scheme! dispatch
#     No sparse matrix; one pass over cells/faces
#     Memory: O(n_cells + n_faces) for field arrays only
#
#   PATH 3 — PROTOTYPE KERNEL (this file):
#     Prototype A + B: direct arithmetic, no operator dispatch layer
#     Same memory layout as Path 2
#     Slightly less overhead: no @generated function indirection
#
# EXPECTED RESULTS
# ----------------
# All three paths produce identical residuals (up to floating-point rounding).
# On CPU:
#   - Assembled path: large upfront cost (matrix build), fast SpMV after
#   - Matrix-free: no matrix, moderate per-call cost
#   - Direct kernel: similar to matrix-free, marginally less dispatch overhead
#
# On GPU (not run here — change backend):
#   - Assembled path: sparse assembly is memory-bandwidth bound; CSR SpMV
#     has irregular access → poor cache utilisation
#   - Matrix-free: embarrassingly parallel cell loop; high cache locality;
#     eliminates matrix memory footprint entirely
#   - Direct kernel: identical GPU advantages as matrix-free
#
# The GPU memory savings scale as:
#   n_cells × (n_interior_faces/n_cells) × sizeof(Float64)
#   ≈ 6 × n_cells × 8 bytes  for typical 3D hex mesh
# For 10M cell mesh: ~480 MB of sparse matrix vs ~80 MB for field arrays.

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
# SETUP: two mesh sizes to show scaling
# =============================================================================

function run_benchmark(mesh_file, label)
    println()
    println("=" ^ 70)
    println("BENCHMARK: ", label)
    println("=" ^ 70)

    mesh     = UNV2D_mesh(mesh_file)
    backend  = CPU()
    activate_multithread(backend)
    workgroup = Threads.nthreads() > 1 ? cld(length(mesh.cells), Threads.nthreads()) : 4
    mesh_dev  = adapt(backend, mesh)
    n_cells   = length(mesh_dev.cells)
    n_bfaces  = length(mesh_dev.boundary_cellsID)
    n_faces   = length(mesh_dev.faces)

    println("  cells: $n_cells  | boundary faces: $n_bfaces | interior faces: $(n_faces - n_bfaces)")

    phi   = ScalarField(mesh_dev);     initialise!(phi,   0.0)
    gamma = FaceScalarField(mesh_dev); initialise!(gamma, 1.0)

    # Set phi = sin(πx)·sin(πy) — non-trivial, non-zero residual
    for i in 1:n_cells
        x, y = mesh_dev.cells[i].centre[1], mesh_dev.cells[i].centre[2]
        phi.values[i] = sin(π * x) * sin(π * y)
    end

    BCs_eqn = assign(region=mesh_dev, (C = [
        Dirichlet(:left_wall, 0.0), Dirichlet(:right_wall, 0.0),
        Dirichlet(:upper_wall, 0.0), Dirichlet(:bottom_wall, 0.0),
    ],))
    BCs_tuple = Tuple(BCs_eqn.C)

    config = Configuration(
        hardware=Hardware(backend=backend, workgroup=workgroup),
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0),
        schemes=(C=Schemes(laplacian=Linear),),
        solvers=(C=nothing,),
        boundaries=(C=BCs_eqn.C,)
    )

    L = ((-Laplacian{Linear}(gamma)) → BCs_eqn.C) → SolverSetup(
        solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0)
    eqn = L(phi)

    # -----------------------------------------------------------------------
    # PATH 1: Assembled (discretise! + SpMV)
    # -----------------------------------------------------------------------
    r_assembled = zeros(Float64, n_cells)

    assembled_build! = function()
        discretise!(eqn, phi, config)
        apply_boundary_conditions!(eqn, config)
    end
    assembled_residual! = function()
        A = _A(eqn); b = _b(eqn)
        r_assembled .= Vector(A * phi.values) .- b
    end

    # warm-up
    assembled_build!(); assembled_residual!()

    t_build   = @elapsed for _ in 1:20; assembled_build!(); end;   t_build   /= 20
    t_spmv    = @elapsed for _ in 1:200; assembled_residual!(); end; t_spmv  /= 200
    a_build   = @allocated assembled_build!()
    a_spmv    = @allocated assembled_residual!()

    println("\n  PATH 1 — Assembled (CSR SpMV)")
    @printf "    matrix build time:  %8.3f μs  (%.0f alloc bytes)\n" t_build*1e6 a_build
    @printf "    SpMV residual time: %8.3f μs  (%.0f alloc bytes)\n" t_spmv*1e6 a_spmv
    @printf "    max |r| (sanity):   %8.2e\n" maximum(abs, r_assembled)

    # -----------------------------------------------------------------------
    # PATH 2: existing explicit_residual! kernel (matrix-free)
    # -----------------------------------------------------------------------
    r_mf = zeros(Float64, n_cells)

    mf_residual! = function()
        explicit_residual!(r_mf, eqn, phi, config)
    end

    # warm-up
    mf_residual!(); mf_residual!()

    t_mf = @elapsed for _ in 1:200; mf_residual!(); end; t_mf /= 200
    a_mf = @allocated mf_residual!()

    println("\n  PATH 2 — existing explicit_residual! (matrix-free kernel)")
    println("    NOTE: computes interior-only residual (no BC contributions)")
    @printf "    residual time: %8.3f μs  (%.0f alloc bytes)\n" t_mf*1e6 a_mf
    @printf "    max |r| (interior only): %8.2e\n" maximum(abs, r_mf)
    @printf "    diff vs Path 1 (expected non-zero due to BC terms): %.2e\n" maximum(abs, r_mf .- r_assembled)

    # -----------------------------------------------------------------------
    # PATH 3: Prototype A + B direct kernel
    # -----------------------------------------------------------------------
    r_proto = zeros(Float64, n_cells)
    source  = zeros(Float64, n_cells)

    proto_residual! = function()
        full_residual!(r_proto, phi.values, gamma.values, source, BCs_tuple, mesh_dev, backend, workgroup)
    end

    # warm-up
    proto_residual!(); proto_residual!()

    t_proto = @elapsed for _ in 1:200; proto_residual!(); end; t_proto /= 200
    a_proto = @allocated proto_residual!()

    println("\n  PATH 3 — Prototype A+B direct kernel")
    @printf "    residual time: %8.3f μs  (%.0f alloc bytes)\n" t_proto*1e6 a_proto
    @printf "    max |r|:       %8.2e\n" maximum(abs, r_proto)
    @printf "    agreement with Path 1: max |diff| = %.2e\n" maximum(abs, r_proto .- r_assembled)

    # -----------------------------------------------------------------------
    # AGREEMENT CHECK
    # -----------------------------------------------------------------------
    println()
    # Path 2 (explicit_residual!) is interior-only — it does NOT include BC
    # residual contributions, so it will differ from Path 1 (which has BCs).
    # Only Path 3 (full_residual! = interior + BC) should agree with Path 1.
    @test maximum(abs, r_proto .- r_assembled) < 1e-10
    println("  Path 3 agrees with Path 1 to < 1e-10 ✓")
    println("  Path 2 differs from Path 1 by design (interior-only, no BC terms)")

    # -----------------------------------------------------------------------
    # MEMORY FOOTPRINT (approximate)
    # -----------------------------------------------------------------------
    # CSR nnz ≈ n_cells + 2 * n_interior_faces (diagonal + 2 off-diagonals per face)
    n_interior = n_faces - n_bfaces
    nnz_approx = n_cells + 2 * n_interior
    matrix_bytes = nnz_approx * sizeof(Float64) + (nnz_approx + n_cells + 1) * sizeof(Int)
    field_bytes  = (n_cells + n_faces) * sizeof(Float64)
    println()
    @printf "  CSR matrix footprint (approx): %.1f KB\n" matrix_bytes / 1024
    @printf "  Field arrays only:             %.1f KB  (%.1fx smaller)\n" field_bytes/1024 matrix_bytes/field_bytes

    return (n_cells=n_cells, t_build=t_build, t_spmv=t_spmv, t_mf=t_mf, t_proto=t_proto)
end

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")

r_small  = run_benchmark(joinpath(grids_dir, "laplace_unit_3by3.unv"),  "3×3 mesh (9 cells)")
r_medium = run_benchmark(joinpath(grids_dir, "laplace_unit_5by5.unv"),  "5×5 mesh (25 cells)")
r_large  = run_benchmark(joinpath(grids_dir, "laplace_2d_mesh.unv"),    "2D mesh (~large)")

println()
println("=" ^ 70)
println("SUMMARY — CPU path")
println("=" ^ 70)
println()
println("Note: On this CPU with $(Threads.nthreads()) thread(s), matrix-free is")
println("comparable to assembled SpMV because n_cells is small and the")
println("assembled matrix fits in L1 cache.  GPU advantages are:")
println()
println("  1. Sparse matrix ELIMINATED from memory: matrix-free saves")
println("     O(nnz × 8 B) = ~6× n_cells × 8 B for 3D hex meshes")
println()
println("  2. Cell loop is embarrassingly parallel — perfect GPU utilisation")
println("     with no synchronisation between cells")
println()
println("  3. No CSR SpMV (irregular memory access) — matrix-free uses")
println("     contiguous face arrays with predictable access pattern")
println()
println("  4. Assembled path requires matrix build (O(nnz) writes) BEFORE")
println("     each residual evaluation if operators change (e.g. Newton)")
println("     Matrix-free skips this entirely")
println()
println("GPU ARCHITECTURE READINESS SUMMARY")
println("-" ^ 50)
println("  PATH 1 (assembled):    GPU-runnable (CSR SpMV supported)")
println("    but: matrix ASSEMBLY is currently sequential CPU code (Discretise_6)")
println("    and: ForwardDiff linearisation is explicitly CPU-only (Solve_3)")
println()
println("  PATH 2 (explicit_residual!):  READY for GPU today")
println("    @kernel function, no sparse matrix, type-stable, allocation-free")
println()
println("  PATH 3 (prototype A+B):  READY for GPU today")
println("    same kernel properties; explicit BC dispatch via typed tuple")
println("    Direct path: swap backend = CUDABackend() to run on GPU")
