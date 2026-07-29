# Nonlinear monolithic Newton with automatic VectorModel decomposition.
#
# Manufactured constant solution:
#   -gamma*laplacian(U) + U.^3 = f
#   U = [2, -1] and f = [8, -1].

using XCALibre
using Test

@testset "Nonlinear Monolithic Newton with Vector Decomposition" begin
    grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
    mesh_file = joinpath(grids_dir, "laplace_unit_3by3.unv")
    mesh = UNV2D_mesh(mesh_file)
    backend = CPU()
    mesh_dev = adapt(backend, mesh)

    exact = [2.0, -1.0, 0.0]
    U = VectorField(mesh_dev)
    initialise!(U, [1.0, -0.5, 0.0])
    forcing = VectorField(mesh_dev)
    initialise!(forcing, exact .^ 3)

    BCs = assign(
        region=mesh_dev,
        (U=[Dirichlet(boundary.name, exact) for boundary in mesh_dev.boundaries],),
    )

    L_U = (
        -Laplacian{Linear}(ConstantScalar(1.0)) +
        NonLinearSi(x -> x^3, x -> 3x^2) == Source(forcing)
    ) → BCs.U

    sys = MonolithicSystem([L_U(U)], [U])
    @test sys.n_vars == 2

    config = Configuration(
        hardware=Hardware(backend=backend, workgroup=1024),
        runtime=Runtime(iterations=12, write_interval=-1, time_step=1.0),
        schemes=(U=Schemes(laplacian=Linear),),
        solvers=nothing,
        boundaries=BCs,
    )

    result = newton_solve!(sys, config; tol=1e-10, verbose=true)
    @test result.converged
    @test all(isapprox.(U.x.values, exact[1]; atol=1e-9))
    @test all(isapprox.(U.y.values, exact[2]; atol=1e-9))
end
