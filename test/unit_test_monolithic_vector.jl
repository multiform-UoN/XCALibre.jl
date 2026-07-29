using XCALibre
using Test
using LinearAlgebra
using Accessors

@testset "Monolithic Vector-Scalar System" begin
    grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
    mesh = UNV2D_mesh(joinpath(grids_dir, "laplace_unit_3by3.unv"))
    backend = CPU(); workgroup = 1024
    mesh_dev = adapt(backend, mesh)

    # Fields: U (Vector), p (Scalar)
    U = VectorField(mesh_dev); initialise!(U, [1.0, 0.0, 0.0])
    p = ScalarField(mesh_dev); initialise!(p, 0.0)

    # Boundaries: Dirichlet on U, Zerogradient on p
    BCs = assign(
        region=mesh_dev,
        (
            U = [Dirichlet(:left_wall, [0.0, 0.0, 0.0]), Dirichlet(:right_wall, [0.0, 0.0, 0.0]), Zerogradient(:upper_wall), Zerogradient(:bottom_wall)],
            p = [Zerogradient(:left_wall), Zerogradient(:right_wall), Zerogradient(:upper_wall), Zerogradient(:bottom_wall)],
        )
    )

    # Define coupled system:
    # 1. -nu*laplacian(U) + grad(p) = 0
    # 2. div(U) = 0
    # (Simplified Stokes)

    nu = 0.01
    L_U = ( - Laplacian{Linear}(ConstantScalar(nu)) == Source(0.0) ) → BCs.U
    L_p = ( - Laplacian{Linear}(ConstantScalar(1.0)) == Source(0.0) ) → BCs.p

    U_eqn = L_U(U)
    p_eqn = L_p(p)

    # Monolithic construction should expand U to 2 block rows
    sys = MonolithicSystem([U_eqn, p_eqn], [U, p])
    @test sys.n_vars == 3

    config = Configuration(
        hardware=Hardware(backend=backend, workgroup=workgroup),
        runtime=Runtime(iterations=1, time_step=1.0, write_interval=-1),
        schemes=(U=Schemes(laplacian=Linear), p=Schemes(laplacian=Linear)),
        solvers=nothing,
        boundaries=BCs
    )

    # BCs are stored inside each sub-equation by the MonolithicSystem constructor
    # (via decompose). No manual bcs_list required.
    res = solve_monolithic!(sys, config)
    @test isfinite(res)

    println("Success: Monolithic Vector-Scalar system runs.")
end

@testset "Monolithic Newton regression" begin
    grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
    mesh = UNV2D_mesh(joinpath(grids_dir, "laplace_unit_3by3.unv"))
    backend = CPU()
    mesh_dev = adapt(backend, mesh)
    boundaries = (:left_wall, :upper_wall, :right_wall, :bottom_wall)

    function newton_config(name, bcs; iterations=12)
        Configuration(
            hardware=Hardware(backend=backend, workgroup=1024),
            runtime=Runtime(iterations=iterations, time_step=1.0, write_interval=-1),
            schemes=NamedTuple{(name,)}((Schemes(laplacian=Linear),)),
            solvers=nothing,
            boundaries=NamedTuple{(name,)}((bcs,)),
        )
    end

    # Fixed damping is applied to the Newton correction. For u²=4 from u=1,
    # the full first correction is 1.5 and damping=0.5 gives u=1.75.
    C = ScalarField(mesh_dev)
    initialise!(C, 1.0)
    C_bcs = assign(
        region=mesh_dev,
        (C=[Extrapolated(name) for name in boundaries],),
    ).C
    C_eqn = (NonLinearSi(x -> x^2, x -> 2x) == Source(4.0)) → C_bcs
    C_sys = MonolithicSystem([C_eqn(C)], [C])
    C_config = newton_config(:C, C_bcs; iterations=1)
    newton_solve!(C_sys, C_config; maxiter=1, damping=0.5)
    @test all(isapprox.(C.values, 1.75; atol=1e-12))

    # Nonzero Dirichlet data must be homogenised for the correction equation.
    T = ScalarField(mesh_dev)
    initialise!(T, 1.0)
    T_bcs = assign(
        region=mesh_dev,
        (T=[Dirichlet(name, 2.0) for name in boundaries],),
    ).T
    T_eqn = (
        -Laplacian{Linear}(ConstantScalar(1.0)) +
        NonLinearSi(x -> x^2, x -> 2x) == Source(4.0)
    ) → T_bcs
    T_sys = MonolithicSystem([T_eqn(T)], [T])
    T_config = newton_config(:T, T_bcs)
    T_result = newton_solve!(T_sys, T_config; tol=1e-10)
    @test T_result.converged
    @test all(isapprox.(T.values, 2.0; atol=1e-9))

    # A Newton reference row removes the null space and drives a pure-Neumann
    # Laplace field to the requested gauge value.
    p = ScalarField(mesh_dev)
    initialise!(p, 1.0)
    p_bcs = assign(
        region=mesh_dev,
        (p=[Zerogradient(name) for name in boundaries],),
    ).p
    p_eqn = (-Laplacian{Linear}(ConstantScalar(1.0)) == Source(0.0)) → p_bcs
    p_sys = MonolithicSystem([p_eqn(p)], [p])
    p_config = newton_config(:p, p_bcs; iterations=4)
    p_result = newton_solve!(p_sys, p_config; reference=(p, 0.0, 1), tol=1e-10)
    @test p_result.converged
    @test maximum(abs, p.values) <= 1e-9
end
