using XCALibre
using Test
using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using Krylov
using Random

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

    @test_throws ArgumentError solve_monolithic!(sys, config; preconditioner=:unknown)
    # neither block is weaker than its coupling: :schur refuses to guess
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

@testset "Schur preconditioner algebra" begin
    # [D Bᵀ; B C] with D diagonal: S = C - B D⁻¹ Bᵀ is the exact Schur complement, so the block
    # upper-triangular preconditioner leaves GMRES two iterations
    n = 30
    D = spdiagm(0 => 2.0 .+ (1:n) ./ n)
    B = 0.1 * sprand(MersenneTwister(1), n, n, 0.2) + 0.1I
    C = spdiagm(0 => fill(1e-3, n))
    A = [D B'; B C]
    A_csr = SparseMatrixCSR(A)
    P = MonolithicSchur(A_csr, 2, n, [2])
    z = collect(range(-1, 1; length=2n))
    S = C - B * (D \ B')
    Pz = vcat(D * z[1:n] + B' * z[n+1:end], S * z[n+1:end])
    y = similar(z)
    ldiv!(y, P, Pz)
    @test y ≈ z
    rhs = A * ones(2n)
    x, stats = Krylov.gmres(A, rhs; M=P, ldiv=true, atol=1e-12, rtol=1e-12)
    @test stats.solved
    @test stats.niter <= 2
    @test x ≈ ones(2n)
    @test XCALibre.Solve._auto_schur_blocks(A_csr, 2, n) == [2]
end

@testset "Schur preconditioner on weakly stabilised Stokes" begin
    u = ScalarField(mesh_dev); initialise!(u, 0.0)
    v = ScalarField(mesh_dev); initialise!(v, 0.0)
    p = ScalarField(mesh_dev); initialise!(p, 0.0)
    BCs = assign(
        region=mesh_dev,
        (
            u = [Dirichlet(:upper_wall, 1.0), Dirichlet(:left_wall, 0.0),
                 Dirichlet(:right_wall, 0.0), Dirichlet(:bottom_wall, 0.0)],
            v = [Dirichlet(name, 0.0) for name in walls],
            p = [Zerogradient(name) for name in walls],
        )
    )
    one_cst = ConstantScalar(1.0)
    L_u = (-Laplacian{Linear}(one_cst) + ScalarGrad{Linear,1}(one_cst, p) == Source(0.0)) → BCs.u
    L_v = (-Laplacian{Linear}(one_cst) + ScalarGrad{Linear,2}(one_cst, p) == Source(0.0)) → BCs.v
    L_p = (
        -Laplacian{Linear}(ConstantScalar(1e-4))
        + VectorDiv{Linear,1}(one_cst, u) + VectorDiv{Linear,2}(one_cst, v)
        == Source(0.0)
    ) → BCs.p
    sys = MonolithicSystem([L_u(u), L_v(v), L_p(p)], [u, v, p])
    config = scaling_config((:u, :v, :p), BCs)
    reference = (3, 0.0, 1)

    A_csr, rhs = XCALibre.Solve.assemble_monolithic_system(sys, nothing, config)
    XCALibre.Solve.apply_monolithic_reference!(A_csr, rhs, sys, 3, 0.0, 1)
    x_direct = get_sparse_matrix(A_csr) \ rhs
    @test XCALibre.Solve._auto_schur_blocks(A_csr, 3, sys.n_cells) == [3]

    its = Dict{String,Int}()
    for (name, pc) in [("blockjacobi", :blockjacobi), ("schur", SchurSplit(p)), ("auto", :schur)]
        foreach(f -> initialise!(f, 0.0), (u, v, p))
        d = solve_monolithic!(sys, config; reference, preconditioner=pc, diagnostics=true)
        x = vcat(u.values, v.values, p.values)
        @test d.linear[end].solved
        @test norm(x - x_direct) <= 1e-7 * norm(x_direct)
        its[name] = d.linear[end].gmres_iterations
    end
    @test its["schur"] == its["auto"]
    @test its["schur"] < its["blockjacobi"]
end
