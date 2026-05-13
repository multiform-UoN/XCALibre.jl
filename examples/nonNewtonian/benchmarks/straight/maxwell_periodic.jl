# =============================================================================
# Incompressible Maxwell: Straight Channel (Periodic/Body Force Driven)
# =============================================================================

include("../benchmark_utils.jl")

mesh_cpu, _ = get_straight_mesh()
mesh_per = make_periodic_topology(mesh_cpu, :inlet, :outlet, [25.0, 0.0, 0.0])
mesh_dev = adapt(CPU(), mesh_per)

u = ScalarField(mesh_dev); initialise!(u, 0.0)
v = ScalarField(mesh_dev); initialise!(v, 0.0)
p = ScalarField(mesh_dev); initialise!(p, 0.0)
txx = ScalarField(mesh_dev); initialise!(txx, 0.0)
tyy = ScalarField(mesh_dev); initialise!(tyy, 0.0)
txy = ScalarField(mesh_dev); initialise!(txy, 0.0)

BCs = assign(
    region = mesh_dev,
    (
        u = [Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0), Empty(:inlet), Empty(:outlet)],
        v = [Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0), Empty(:inlet), Empty(:outlet)],
        p = [Zerogradient(:top), Zerogradient(:bottom), Empty(:inlet), Empty(:outlet)],
        txx = [Zerogradient(:top), Zerogradient(:bottom), Empty(:inlet), Empty(:outlet)],
        tyy = [Zerogradient(:top), Zerogradient(:bottom), Empty(:inlet), Empty(:outlet)],
        txy = [Zerogradient(:top), Zerogradient(:bottom), Empty(:inlet), Empty(:outlet)],
    )
)

solvers = (
    u = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    v = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    p = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    txx = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    tyy = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    txy = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
)
config = Configuration(solvers=solvers, schemes=(u=Schemes(), v=Schemes(), p=Schemes(), txx=Schemes(), tyy=Schemes(), txy=Schemes()),
                       runtime=Runtime(iterations=5, write_interval=-1, time_step=0.01), hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=BCs)

one_cst = ConstantScalar(1.0)
mu_s_cst = ConstantScalar(1e-6)
tau_rc_cst = ConstantScalar(0.1)
inv_lam = ConstantScalar(1.0/1.0)
two_mu_lam = ConstantScalar(2.0*1.0/1.0)
mu_lam_cst = ConstantScalar(1.0/1.0)

for step in 1:config.runtime.iterations
    L_u = (( - Laplacian{XCALibre.Linear}(mu_s_cst) - ScalarGrad{XCALibre.Linear,1}(one_cst, txx) - ScalarGrad{XCALibre.Linear,2}(one_cst, txy) + ScalarGrad{XCALibre.Linear,1}(one_cst, p) == Source(1.0) ) → BCs.u) → solvers.u
    L_v = (( - Laplacian{XCALibre.Linear}(mu_s_cst) - ScalarGrad{XCALibre.Linear,1}(one_cst, txy) - ScalarGrad{Linear,2}(one_cst, tyy) + ScalarGrad{XCALibre.Linear,2}(one_cst, p) == Source(0.0) ) → BCs.v) → solvers.v
    L_p = (( - Laplacian{XCALibre.Linear}(tau_rc_cst) + VectorDiv{XCALibre.Linear,1}(one_cst, u) + VectorDiv{XCALibre.Linear,2}(one_cst, v) == Source(0.0) ) → BCs.p) → solvers.p

    L_txx = (( Si(inv_lam) - ScalarGrad{XCALibre.Linear,1}(two_mu_lam, u) == Source(0.0) ) → BCs.txx) → solvers.txx
    L_tyy = (( Si(inv_lam) - ScalarGrad{XCALibre.Linear,2}(two_mu_lam, v) == Source(0.0) ) → BCs.tyy) → solvers.tyy
    L_txy = (( Si(inv_lam) - ScalarGrad{XCALibre.Linear,2}(mu_lam_cst, u) - ScalarGrad{XCALibre.Linear,1}(mu_lam_cst, v) == Source(0.0) ) → BCs.txy) → solvers.txy

    sys = MonolithicSystem([L_u(u), L_v(v), L_p(p), L_txx(txx), L_tyy(tyy), L_txy(txy)], [u, v, p, txx, tyy, txy])
    A_csr, b_mono = assemble_monolithic_system(sys, (BCs.u, BCs.v, BCs.p, BCs.txx, BCs.tyy, BCs.txy), config)
    A_julia = get_sparse_matrix(A_csr)

    # Pin pressure
    p_row = 2 * length(mesh_dev.cells) + 1
    A_julia[p_row, :] .= 0.0
    A_julia[p_row, p_row] = 1.0
    b_mono[p_row] = 0.0

    x = A_julia \ b_mono
    XCALibre.Solve.update_fields!(sys, x)
    res = norm(A_julia*x - b_mono)

    if step == config.runtime.iterations
        report_results("Maxwell Straight Periodic", res, u, p, txx)
    end
end
