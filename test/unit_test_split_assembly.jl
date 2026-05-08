using XCALibre
using Test
using LinearAlgebra

@testset "Split Assembly vs Full Discretise" begin
    grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
    mesh_file = joinpath(grids_dir, "laplace_unit_3by3.unv")
    mesh = UNV2D_mesh(mesh_file)
    backend = CPU()
    mesh_dev = adapt(backend, mesh)
    
    phi = ScalarField(mesh_dev); initialise!(phi, 1.0)
    gamma = ConstantScalar(0.1)
    f = ConstantScalar(1.0)
    
    BCs = [
        Dirichlet(:left_wall, 100.0),
        Zerogradient(:right_wall),
        Dirichlet(:bottom_wall, 0.0),
        Zerogradient(:upper_wall)
    ]
    
    BC_mapped = assign(
        region = mesh_dev,
        (
            phi = BCs,
        )
    )
    
    config = Configuration(
        schemes = (phi=Schemes(laplacian=Linear),),
        solvers = (phi=SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0),),
        hardware = Hardware(backend=backend, workgroup=1023),
        runtime = Runtime(iterations=1, write_interval=1, time_step=1.0),
        boundaries = BC_mapped
    )
    
    # 1. Full discretise!
    L = - Laplacian{Linear}(gamma) == Source(f)
    eqn1 = (L → BCs)(phi)
    discretise!(eqn1, phi, config)
    A_full = Vector(eqn1.equation.A.parent.nzval)
    b_full = Vector(eqn1.equation.b)
    
    # 2. Split assembly
    eqn2 = (L → BCs)(phi)
    assemble_matrix!(eqn2, config)
    assemble_rhs!(eqn2, eqn2.model.sources[1], config)
    
    A_split = Vector(eqn2.equation.A.parent.nzval)
    b_split = Vector(eqn2.equation.b)
    
    @test A_full ≈ A_split atol=1e-12
    @test b_full ≈ b_split atol=1e-12
end
