using XCALibre
using Test

# A NonlinearOperator op(f(φ)), linearised about φ₀ = C, must assemble the same residual as the
# operator applied directly to F = f(C) with the mapped boundary values.
let
    grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
    mesh = UNV2D_mesh(joinpath(grids_dir, "laplace_unit_3by3.unv"))
    backend = CPU()
    hardware = Hardware(backend=backend, workgroup=1024)
    mesh_dev = adapt(backend, mesh)

    f_nl(x) = x^2
    df_nl(x) = 2x
    nl_map = NonlinearMap(f_nl, df_nl)

    bcs(g) = assign(region=mesh_dev, (C = [
        Dirichlet(:left_wall, g(0.7)), Zerogradient(:right_wall),
        Dirichlet(:bottom_wall, g(0.25)), Extrapolated(:upper_wall)],))
    BCs_phi = bcs(identity)
    BCs_f = bcs(f_nl)

    schemes = (C = Schemes(divergence=Linear, laplacian=Linear),)
    solvers = (C = SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),)
    config(BCs) = Configuration(solvers=solvers, schemes=schemes,
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0), hardware=hardware, boundaries=BCs)
    config_phi, config_f = config(BCs_phi), config(BCs_f)

    C = ScalarField(mesh_dev)
    F = ScalarField(mesh_dev)
    for i in eachindex(C.values)
        C.values[i] = 0.2 + 0.07i
        F.values[i] = f_nl(C.values[i])
    end
    gamma = ConstantScalar(1.0)
    mdotf = FaceScalarField(mesh_dev)
    for fID in eachindex(mdotf.values)
        mdotf.values[fID] = 0.4 + 0.03fID
    end

    residual(eqn, phi, BCs, config) = begin
        discretise!(eqn, phi, config)
        apply_boundary_conditions!(eqn, BCs, nothing, 0.0, config)
        eqn.equation.A.parent * Vector(phi.values) - Vector(eqn.equation.b)
    end

    function check(nonlinear_term, direct_term)
        nonlinear_eqn = (nonlinear_term(C) == Source(ConstantScalar(0.0))) → ScalarEquation(C, BCs_phi.C)
        BCs, linear_eqn = linearize_physics(BCs_phi.C, nonlinear_eqn, config_phi)
        @test nonlinear_eqn.model.terms[1] isa NonlinearOperator
        @test linear_eqn.model.terms[1] isa AffineOperator
        direct_eqn = (direct_term(F) == Source(ConstantScalar(0.0))) → ScalarEquation(F, BCs_f.C)
        @test residual(linear_eqn, C, BCs, config_phi) ≈ residual(direct_eqn, F, BCs_f.C, config_f) atol=1e-11 rtol=1e-11
    end

    @testset "Nonlinear operators: Newton linearisation" begin
        check(phi -> -Laplacian{Linear}(gamma, nl_map, phi), phi -> -Laplacian{Linear}(gamma, phi))
        check(phi -> Divergence{Linear}(mdotf, nl_map, phi), phi -> Divergence{Linear}(mdotf, phi))
        check(phi -> Divergence{Upwind}(mdotf, nl_map, phi), phi -> Divergence{Upwind}(mdotf, phi))
    end
end
