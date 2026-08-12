using XCALibre
using Accessors
using KernelAbstractions
using Test
using SparseArrays

# Regression test asserting the two `apply_boundary_conditions!` call
# conventions (equation-owned BCs vs explicit positional BCs) are genuinely
# additive, ergonomic entry points over ONE shared implementation, not two
# competing implementations of BC semantics. See the note in
# `src/Discretise/Discretise_5_apply_bcs.jl`.
#
#   - equation-owned:  apply_boundary_conditions!(eqn, config; time, component)
#     (fork addition used by the PDE operator framework; BCs read from `eqn`
#     via `get_bcs(eqn)`)
#   - upstream legacy:  apply_boundary_conditions!(eqn, BCs, component, time, config)
#     (original upstream positional form, still used unchanged by
#     SIMPLE/PISO/CPISO, FilmModel and the LES k-equation solvers)
#
# Both must assemble to identical systems for identical BCs.

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "laplace_unit_3by3.unv"
mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file)

backend = CPU(); workgroup = 1024
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

BCs = assign(
    region = mesh_dev,
    (
        T = [
            Dirichlet(:left_wall, 50.0),
            Zerogradient(:right_wall),
            Dirichlet(:bottom_wall, 10.0),
            Zerogradient(:upper_wall)
        ],
    )
)

solvers = (
    T = SolverSetup(
        solver      = Cg(),
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 1.0,
    )
)

schemes = (
    T = Schemes(laplacian = Linear),
)

config = Configuration(solvers=solvers, schemes=schemes, runtime=Runtime(iterations=1, write_interval=1, time_step=1), hardware=hardware, boundaries=BCs)

T_owned = ScalarField(mesh_dev)
T_legacy = ScalarField(mesh_dev)
gamma = ConstantScalar(1.0)

# Two independently built equations, sharing the same BCs, gamma and mesh.
eqn_owned = (
    - Laplacian{schemes.T.laplacian}(gamma, T_owned)
    ==
    Source(ConstantScalar(0.0))
) → ScalarEquation(T_owned, config.boundaries.T)

eqn_legacy = (
    - Laplacian{schemes.T.laplacian}(gamma, T_legacy)
    ==
    Source(ConstantScalar(0.0))
) → ScalarEquation(T_legacy, config.boundaries.T)

discretise!(eqn_owned, T_owned, config)
discretise!(eqn_legacy, T_legacy, config)

# Entry point 1: equation-owned BCs (fork addition for the PDE operator framework)
apply_boundary_conditions!(eqn_owned, config; time=0.0)

# Entry point 2: legacy positional BCs (unchanged upstream API)
apply_boundary_conditions!(eqn_legacy, config.boundaries.T, nothing, 0.0, config)

@testset "BC entry points produce identical assembled systems" begin
    @test get_bcs(eqn_owned) == config.boundaries.T
    @test eqn_owned.equation.A.parent ≈ eqn_legacy.equation.A.parent
    @test eqn_owned.equation.b ≈ eqn_legacy.equation.b
end
