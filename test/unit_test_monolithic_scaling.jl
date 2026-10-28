using XCALibre
using Test
using LinearAlgebra
using SparseArrays

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "laplace_2d_mesh.unv"))
backend = CPU()
mesh_dev = adapt(backend, mesh)
walls = (:left_wall, :upper_wall, :right_wall, :bottom_wall)

function scaling_config(names, bcs; iterations=1)
    Configuration(
        hardware=Hardware(backend=backend, workgroup=1024),
        runtime=Runtime(iterations=iterations, time_step=1.0, write_interval=-1),
        schemes=NamedTuple{names}(Tuple(Schemes(laplacian=Linear) for _ in names)),
        solvers=NamedTuple{names}(Tuple(
            SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0)
            for _ in names)),
        boundaries=bcs,
    )
end

@testset "Monolithic active blocks and block-Jacobi" begin
    # One-way coupling: a does not see b, b sees a. Block (1, 2) is structurally zero.
    a = ScalarField(mesh_dev); initialise!(a, 0.0)
    b = ScalarField(mesh_dev); initialise!(b, 0.0)
    BCs = assign(
        region=mesh_dev,
        (
            a = [Dirichlet(:left_wall, 1.0), Dirichlet(:right_wall, 0.0),
                 Zerogradient(:upper_wall), Zerogradient(:bottom_wall)],
            b = [Dirichlet(name, 0.0) for name in walls],
        )
    )
    L_a = (-Laplacian{Linear}(ConstantScalar(1.0)) == Source(0.0)) → BCs.a
    L_b = (
        -Laplacian{Linear}(ConstantScalar(1.0))
        - Laplacian{Linear}(ConstantScalar(0.5), a)
        == Source(1.0)
    ) → BCs.b
    sys = MonolithicSystem([L_a(a), L_b(b)], [a, b])
    config = scaling_config((:a, :b), BCs)

    pattern = XCALibre.Solve._monolithic_block_pattern(sys)
    @test pattern == Bool[1 0; 1 1]

    A_csr, rhs = XCALibre.Solve.assemble_monolithic_system(sys, nothing, config)
    n = sys.n_cells
    full = XCALibre.Solve._build_monolithic_sparsity(2, n, mesh_dev, Float64)
    @test length(A_csr.nzval) * 4 == length(full.nzval) * 3
    x_direct = get_sparse_matrix(A_csr) \ rhs

    plain = solve_monolithic!(sys, config; diagnostics=true)
    x_plain = vcat(copy(a.values), copy(b.values))
    initialise!(a, 0.0); initialise!(b, 0.0)
    bj = solve_monolithic!(sys, config; preconditioner=:blockjacobi, diagnostics=true)
    x_bj = vcat(copy(a.values), copy(b.values))

    @test norm(x_plain - x_direct) <= 1e-7 * norm(x_direct)
    @test norm(x_bj - x_direct) <= 1e-7 * norm(x_direct)
    # block lower-triangular: M⁻¹A = I + nilpotent, so GMRES needs at most two iterations
    @test bj.linear[end].gmres_iterations <= 3
    @test bj.linear[end].gmres_iterations < plain.linear[end].gmres_iterations

    @test_throws ArgumentError solve_monolithic!(sys, config; preconditioner=:schur)
end

@testset "Monolithic Newton with equilibration and block-Jacobi" begin
    # -k lap(T) + k T² = 4k with T = 2 on the walls: solution T = 2, for any scale k
    k = 1e6
    T = ScalarField(mesh_dev); initialise!(T, 1.0)
    BCs = assign(region=mesh_dev, (T=[Dirichlet(name, 2.0) for name in walls],))
    L_T = (
        -Laplacian{Linear}(ConstantScalar(k)) +
        NonLinearSi(x -> k*x^2, x -> 2k*x) == Source(4k)
    ) → BCs.T
    sys = MonolithicSystem([L_T(T)], [T])
    config = scaling_config((:T,), BCs; iterations=12)

    result = newton_solve!(sys, config; tol=1e-3, equilibrate=true, preconditioner=:blockjacobi)
    @test result.converged
    @test all(isapprox.(T.values, 2.0; atol=1e-8))
    @test length(result.linear_iterations) == result.iterations - 1
    # one exact block: every correction solve is a single preconditioned iteration
    @test all(<=(2), result.linear_iterations)
end
