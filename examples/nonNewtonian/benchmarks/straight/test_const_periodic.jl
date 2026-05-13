using XCALibre
using LinearAlgebra
using Printf
using SparseArrays
using StaticArrays

include("../benchmark_utils.jl")

function small_periodic_test()
    # Create 4x4 mesh
    # quad40.unv scaled is 25x25. 40 cells. 
    # Let's just use the 40x40 mesh but smaller iterations.
    
    mesh_cpu, _ = get_straight_mesh()
    mesh_per = XCALibre.Mesh.construct_periodic_topology(mesh_cpu, :inlet, :outlet, [25.0, 0.0, 0.0])
    mesh_dev = adapt(CPU(), mesh_per)

    u = ScalarField(mesh_dev); initialise!(u, 1.0) # Constant field test
    v = ScalarField(mesh_dev); initialise!(v, 0.0)
    p = ScalarField(mesh_dev); initialise!(p, 1.0)

    BCs = assign(
        region = mesh_dev,
        (
            u = [Dirichlet(:top, 1.0), Dirichlet(:bottom, 1.0), Empty(:inlet), Empty(:outlet)],
            v = [Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0), Empty(:inlet), Empty(:outlet)],
            p = [Zerogradient(:top), Zerogradient(:bottom), Empty(:inlet), Empty(:outlet)],
        )
    )

    solvers = (
        u = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
        v = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
        p = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    )
    config = Configuration(solvers=solvers, schemes=(u=Schemes(), v=Schemes(), p=Schemes()),
                           runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0), hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=BCs)

    # Constant u=1, p=1 should be a solution for f=0
    L_u = ((- Laplacian{XCALibre.Linear}(ConstantScalar(1.0)) + ScalarGrad{XCALibre.Linear,1}(ConstantScalar(1.0), p) == Source(0.0)) → BCs.u) → solvers.u
    L_v = ((- Laplacian{XCALibre.Linear}(ConstantScalar(1.0)) + ScalarGrad{XCALibre.Linear,2}(ConstantScalar(1.0), p) == Source(0.0)) → BCs.v) → solvers.v
    L_p = ((- Laplacian{XCALibre.Linear}(ConstantScalar(0.1)) + VectorDiv{XCALibre.Linear,1}(ConstantScalar(1.0), u) + VectorDiv{XCALibre.Linear,2}(ConstantScalar(1.0), v) == Source(0.0)) → BCs.p) → solvers.p

    sys = MonolithicSystem([L_u(u), L_v(v), L_p(p)], [u, v, p])
    A_csr, b_mono = assemble_monolithic_system(sys, (BCs.u, BCs.v, BCs.p), config)
    
    # Residual of u=1, p=1
    x = zeros(3 * length(mesh_dev.cells))
    n = length(mesh_dev.cells)
    x[1:n] .= 1.0 # u
    x[2n+1:3n] .= 1.0 # p
    
    res_vec = A_csr * x - b_mono
    u_res = res_vec[1:n]
    v_res = res_vec[n+1:2n]
    p_res = res_vec[2n+1:3n]
    
    @info "Residual Breakdown" u_norm=norm(u_res) v_norm=norm(v_res) p_norm=norm(p_res)
    @info "Sample u res" res=u_res[1:5]
    @info "Sample p res" res=p_res[1:5]
end

small_periodic_test()
