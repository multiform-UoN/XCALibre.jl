# =============================================================================
# Incompressible Stokes: L-Bend Channel (Neumann/Body Force Driven)
# =============================================================================

include("../benchmark_utils.jl")

mesh, mesh_dev = get_bend_mesh()
u = ScalarField(mesh_dev); initialise!(u, 0.0)
v = ScalarField(mesh_dev); initialise!(v, 0.0)
p = ScalarField(mesh_dev); initialise!(p, 0.0)

BCs = assign(
    region = mesh_dev,
    (
        u = [Dirichlet(:walls, 0.0), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)],
        v = [Dirichlet(:walls, 0.0), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)],
        p = [Zerogradient(:walls), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)],
    )
)

solvers = (
    u = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    v = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    p = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
)
config = Configuration(solvers=solvers, schemes=(u=Schemes(), v=Schemes(), p=Schemes()),
                       runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0), hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=BCs)

mu_cst = ConstantScalar(1.0)
one_cst = ConstantScalar(1.0)
tau_rc_cst = ConstantScalar(0.1)

L_u = ((- Laplacian{XCALibre.Linear}(mu_cst) + ScalarGrad{XCALibre.Linear,1}(one_cst, p) == Source(1.0)) → BCs.u) → solvers.u
L_v = ((- Laplacian{XCALibre.Linear}(mu_cst) + ScalarGrad{XCALibre.Linear,2}(one_cst, p) == Source(1.0)) → BCs.v) → solvers.v
L_p = ((- Laplacian{XCALibre.Linear}(tau_rc_cst) + VectorDiv{XCALibre.Linear,1}(one_cst, u) + VectorDiv{XCALibre.Linear,2}(one_cst, v) == Source(0.0)) → BCs.p) → solvers.p

sys = MonolithicSystem([L_u(u), L_v(v), L_p(p)], [u, v, p])
res = XCALibre.Solve.solve_monolithic!(sys, (BCs.u, BCs.v, BCs.p), config; reference=(3, 0.0, 1))

report_results("Stokes Bend Neumann", res, u, p)
