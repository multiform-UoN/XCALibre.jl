using XCALibre
using Test
using LinearAlgebra

@testset "Matrix-Free vs Matrix-Based Residual" begin
    grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
    mesh_file = joinpath(grids_dir, "laplace_unit_3by3.unv")
    mesh = UNV2D_mesh(mesh_file)
    backend = CPU()
    mesh_dev = adapt(backend, mesh)

    phi = ScalarField(mesh_dev); initialise!(phi, 1.0)
    gamma = ConstantScalar(0.1)
    f = ConstantScalar(0.0)

    BCs = [
        Dirichlet(:left_wall, 100.0),
        Zerogradient(:right_wall),
        Dirichlet(:bottom_wall, 0.0),
        Zerogradient(:upper_wall)
    ]

    # Define PDE
    L = - Laplacian{Linear}(gamma) == Source(f)
    L_complete = L → BCs

    BC_mapped = assign(
        region = mesh_dev,
        (
            phi = BCs,
        )
    )

    # 1. Matrix-based residual
    config = Configuration(
        schemes = (phi=Schemes(laplacian=Linear),),
        solvers = (phi=SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0),),
        hardware = Hardware(backend=backend, workgroup=1023),
        runtime = Runtime(iterations=1, write_interval=1, time_step=1.0),
        boundaries = BC_mapped
    )
    eqn = L_complete(phi)
    discretise!(eqn, phi, config)
    apply_boundary_conditions!(eqn, config)

    r_mat = eqn.equation.A.parent * Vector(phi.values) - Vector(eqn.equation.b)

    # 2. Matrix-free residual
    r_free = similar(phi.values)
    explicit_residual!(r_free, eqn, phi, config)

    @test r_mat ≈ Vector(r_free) atol=1e-12
end
