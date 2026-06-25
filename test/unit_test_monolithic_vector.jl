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
