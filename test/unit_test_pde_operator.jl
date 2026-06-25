using XCALibre
using Test
using StaticArrays

@testset "PDEOperator and OperatorTemplate" begin
    # Load a real mesh to ensure all internal type detection works
    grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
    mesh_file = joinpath(grids_dir, "laplace_unit_3by3.unv")
    mesh = UNV2D_mesh(mesh_file)

    # 1. Test OperatorTemplate construction
    mu = ConstantScalar(1.0)
    L_temp = Laplacian{Linear}(mu)
    @test L_temp isa XCALibre.ModelFramework.OperatorTemplate

    # 2. Test PDEOperator construction via DSL
    f = ConstantScalar(0.0)
    L = - Laplacian{Linear}(mu) == Source(f)
    @test L isa PDEOperator
    @test length(L.templates) == 1
    @test L.templates[1].sign == -1
    @test length(L.sources) == 1

    # 3. Test PDEOperator with BCs and SolverSetup
    BCs = [Dirichlet(:inlet, 1.0)]
    setup = SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0)
    L_complete = (L → BCs) → setup
    @test L_complete.BCs == BCs
    @test L_complete.setup === setup

    # 4. Test binding to a field
    phi = ScalarField(mesh)
    eqn = L_complete(phi)
    @test eqn isa ModelEquation
    @test get_phi(eqn) === phi
    @test get_bcs(eqn) == BCs
    @test eqn.setup === setup
end
