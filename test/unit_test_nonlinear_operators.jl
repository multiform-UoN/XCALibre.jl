using XCALibre
using KernelAbstractions
using Test

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "laplace_unit_3by3.unv"))

backend = CPU()
hardware = Hardware(backend=backend, workgroup=1024)
mesh_dev = adapt(backend, mesh)

f_nl(x) = x^2
df_nl(x) = 2x
nl_map = NonlinearMap(f_nl, df_nl)

BCs_phi = assign(
    region=mesh_dev,
    (
        C = [
            Dirichlet(:left_wall, 0.7),
            Zerogradient(:right_wall),
            Dirichlet(:bottom_wall, 0.25),
            Extrapolated(:upper_wall),
        ],
    ),
)

BCs_f = assign(
    region=mesh_dev,
    (
        C = [
            Dirichlet(:left_wall, f_nl(0.7)),
            Zerogradient(:right_wall),
            Dirichlet(:bottom_wall, f_nl(0.25)),
            Extrapolated(:upper_wall),
        ],
    ),
)

schemes = (C = Schemes(divergence=Linear, laplacian=Linear),)
solvers = (
    C = SolverSetup(
        solver=Cg(),
        preconditioner=Jacobi(),
        convergence=1e-10,
        relax=1.0,
    ),
)

config_phi = Configuration(
    solvers=solvers,
    schemes=schemes,
    runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0),
    hardware=hardware,
    boundaries=BCs_phi,
)

config_f = Configuration(
    solvers=solvers,
    schemes=schemes,
    runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0),
    hardware=hardware,
    boundaries=BCs_f,
)

C = ScalarField(mesh_dev)
F = ScalarField(mesh_dev)
for i in eachindex(C.values)
    C.values[i] = 0.2 + 0.07 * i
    F.values[i] = f_nl(C.values[i])
end

gamma = ConstantScalar(1.0)
mdotf = FaceScalarField(mesh_dev)
for fID in eachindex(mdotf.values)
    mdotf.values[fID] = 0.4 + 0.03 * fID
end

function assembled_residual(eqn, phi, BCs, config)
    discretise!(eqn, phi, config)
    apply_boundary_conditions!(eqn, config; time=0.0)
    return eqn.equation.A.parent * Vector(phi.values) - Vector(eqn.equation.b)
end

function assert_discrete_newton_match(nonlinear_term, direct_term)
    nonlinear_eqn = (
        nonlinear_term(C)
        ==
        Source(ConstantScalar(0.0))
    ) → ScalarEquation(C, BCs_phi.C)

    updated_bcs, linear_eqn = linearize_physics(BCs_phi.C, nonlinear_eqn)

    @test nonlinear_eqn.model.terms[1] isa NonlinearOperator
    @test linear_eqn.model.terms[1] isa AffineOperator
    @test length(linear_eqn.model.sources) == length(nonlinear_eqn.model.sources)
    @test typeof(linear_eqn.model).parameters[1] == length(linear_eqn.model.terms)
    @test typeof(linear_eqn.model).parameters[2] == length(linear_eqn.model.sources)

    direct_eqn = (
        direct_term(F)
        ==
        Source(ConstantScalar(0.0))
    ) → ScalarEquation(F, BCs_f.C)

    r_linear = assembled_residual(linear_eqn, C, updated_bcs, config_phi)
    r_direct = assembled_residual(direct_eqn, F, BCs_f.C, config_f)

    @test r_linear ≈ r_direct atol=1e-11 rtol=1e-11
end

@testset "Affine nonlinear differential operators" begin
    assert_discrete_newton_match(
        phi -> -Laplacian{schemes.C.laplacian}(gamma, nl_map, phi),
        phi -> -Laplacian{schemes.C.laplacian}(gamma, phi),
    )

    assert_discrete_newton_match(
        phi -> Divergence{Linear}(mdotf, nl_map, phi),
        phi -> Divergence{Linear}(mdotf, phi),
    )

    assert_discrete_newton_match(
        phi -> Divergence{Upwind}(mdotf, nl_map, phi),
        phi -> Divergence{Upwind}(mdotf, phi),
    )
end

@testset "NonLinearSi source parameters" begin
    source_template = (
        NonLinearSi(nl_map, C)
        ==
        Source(ConstantScalar(0.0))
    ) → ScalarEquation(C, BCs_phi.C)

    _, source_linear = linearize_physics(BCs_phi.C, source_template)

    @test source_template.model.terms[1].type isa NonLinearSi
    @test source_linear.model.terms[1].type isa Si
    @test length(source_linear.model.sources) == 2
    @test typeof(source_linear.model).parameters[1] == length(source_linear.model.terms)
    @test typeof(source_linear.model).parameters[2] == length(source_linear.model.sources)

    discretise!(source_linear, C, config_phi)
    A = source_linear.equation.A.parent
    b = source_linear.equation.b

    for i in eachindex(C.values)
        volume = mesh_dev.cells[i].volume
        expected_diag = df_nl(C.values[i]) * volume
        expected_rhs = -(f_nl(C.values[i]) - df_nl(C.values[i]) * C.values[i]) * volume
        @test A[i, i] ≈ expected_diag
        @test b[i] ≈ expected_rhs
    end
end
