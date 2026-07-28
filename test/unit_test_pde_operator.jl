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

    # 5. Scaling keeps source-field classification valid during generated
    # discretisation, and a composed time term is lowered first to satisfy the
    # legacy time-coefficient convention.
    all_BCs = [
        Dirichlet(:inlet, 1.0),
        Dirichlet(:outlet, 0.0),
        Zerogradient(:bottom),
        Zerogradient(:top),
    ]
    L_scaled = ((0.5 * L) → all_BCs) → setup
    scaled_eqn = L_scaled(phi)
    config = Configuration(
        solvers=(phi=setup,),
        schemes=(phi=Schemes(laplacian=Linear),),
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0),
        hardware=Hardware(backend=CPU(), workgroup=64),
        boundaries=(phi=Tuple(all_BCs),),
    )
    r = zeros(length(phi.values))
    residual!(r, scaled_eqn, config)
    @test all(isfinite, r)

    L_transient = L_scaled + Time{Euler}(ConstantScalar(1.0))
    @test first(L_transient.templates).type isa Time{Euler}
end
