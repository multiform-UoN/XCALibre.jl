using XCALibre
using Test
using LinearAlgebra
using Accessors
using Statistics

@testset "Newton AD Backends" begin
    grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
    mesh = UNV2D_mesh(joinpath(grids_dir, "laplace_unit_3by3.unv"))
    backend = CPU(); workgroup = 1024
    mesh_dev = adapt(backend, mesh)
    
    phi = ScalarField(mesh_dev)
    
    # Equation: -Laplacian(phi) + phi^2 = 0.75
    # BCs: Dirichlet(left)=0, Dirichlet(right)=1
    
    L = -Laplacian{Linear}(ConstantScalar(1.0)) + NonLinearSi(x -> x^2) == Source(0.75)
    
    BCs = assign(region=mesh_dev, (
        phi = [
            Dirichlet(:left_wall, 0.0),
            Dirichlet(:right_wall, 1.0),
            Zerogradient(:upper_wall),
            Zerogradient(:bottom_wall)
        ],
    ))
    
    setup = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0)
    
    L_full = (L → BCs.phi) → setup
    eqn = L_full(phi)
    @reset eqn.preconditioner = set_preconditioner(setup.preconditioner, eqn)
    @reset eqn.solver = XCALibre._workspace(setup.solver, _b(eqn))
    
    config = Configuration(
        hardware=Hardware(backend=backend, workgroup=workgroup),
        runtime=Runtime(iterations=20, time_step=1.0, write_interval=-1),
        schemes=(phi=Schemes(laplacian=Linear),),
        solvers=(phi=nothing,),
        boundaries=BCs
    )
    
    println("\n--- Testing ForwardDiff ---")
    initialise!(phi, 0.5)
    res_fd = newton_solve!(eqn, config; ad_backend=:forwarddiff, verbose=true)
    @test res_fd.converged
    
    println("\n--- Testing Enzyme ---")
    initialise!(phi, 0.5)
    res_ez = newton_solve!(eqn, config; ad_backend=:enzyme, verbose=true)
    @test res_ez.converged
    
    println("\nAD Backends Test Success")
end
