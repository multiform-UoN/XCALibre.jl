# =============================================================================
# PENDING — requires Phase 3A: newton_solve!(::MonolithicSystem, ...)
# =============================================================================
# This file shows the intended API for monolithic Newton with VectorModel
# decomposition. It will not run until Phase 3A (monolithic Newton) is
# implemented in Solve_2_monolithic.jl.
#
# See /plans/vector-decompose-monolithic-newton.md for the implementation plan.
# =============================================================================

using XCALibre
using Test
using LinearAlgebra
using Accessors

@testset "Monolithic Newton with Vector Decomposition" begin
    grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
    mesh_file = joinpath(grids_dir, "laplace_unit_3by3.unv")
    mesh = UNV2D_mesh(mesh_file)
    backend = CPU()
    mesh_dev = adapt(backend, mesh)

    # 1. Setup Fields
    p = ScalarField(mesh_dev); initialise!(p, 0.0)
    U = VectorField(mesh_dev); initialise!(U, [0.0, 0.0, 0.0])

    # 2. Setup Equations
    # Simple coupled system:
    # p + laplacian(1, U_x) = x^2 (nonlinear source)
    # laplacian(1, p) + div(U) = 0

    # We will use NonLinearSi for p to test nonlinearity
    # We will use GradDiv to couple U to p
    # Actually, a simpler test:
    # U_x - U_y^2 = 0
    # U_y + U_x^2 = 1

    # For FVM, let's just do a diffusion-reaction system:
    # - Laplacian(p) + p^2 = U_x
    # - Laplacian(U) + U = grad(p)

    # Let's do a simple 1D-like coupling
    # eq1: -Laplacian(p) + NonLinearSi(x -> x^2)(p) - ScalarGrad(1, U.x) = 0

    # Actually, let's just test that the API runs without crashing on a linear system first!

    gamma = ConstantScalar(1.0)

    L_p = ((
        - Laplacian{Linear}(gamma)
        + Si(ConstantScalar(1.0))
    ) → [Extrapolated(:all)]) → SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0)

    L_U = ((
        - Laplacian{Linear}(gamma)
        + Si(ConstantScalar(1.0))
    ) → [Extrapolated(:all)]) → SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0)

    p_eqn = L_p(p)
    U_eqn = L_U(U)

    sys = MonolithicSystem([p_eqn, U_eqn], [p, U])
    @test sys.n_vars == 3

    config = Configuration(
        hardware = Hardware(backend=backend, workgroup=1024),
        runtime = Runtime(iterations=1, write_interval=1, time_step=1.0),
        schemes = (
            p = Schemes(laplacian=Linear),
            U = Schemes(laplacian=Linear)
        ),
        solvers = (p=nothing, U=nothing),
        boundaries = (p=[], U=[])
    )

    bcs_list = [get_bcs(eqn) for eqn in sys.equations]

    res = newton_solve!(sys, bcs_list, config; maxiter=2)
    @test res.iterations > 0

    println("Success")
end
