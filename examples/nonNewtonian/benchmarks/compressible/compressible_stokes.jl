# =============================================================================
# Weakly Compressible Stokes: Straight Channel
# =============================================================================

include("../benchmark_utils.jl")

mesh, mesh_dev = get_straight_mesh()
u = ScalarField(mesh_dev); initialise!(u, 0.0)
v = ScalarField(mesh_dev); initialise!(v, 0.0)
p = ScalarField(mesh_dev); initialise!(p, 0.0)

BCs = assign(
    region = mesh_dev,
    (
        u = [Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0), Zerogradient(:inlet), Zerogradient(:outlet)],
        v = [Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0), Zerogradient(:inlet), Zerogradient(:outlet)],
        p = [Zerogradient(:top), Zerogradient(:bottom), Dirichlet(:inlet, 1.0), Dirichlet(:outlet, 0.0)],
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
beta_cst = ConstantScalar(0.01) # Compressibility factor (1/BulkModulus)

# Momentum: -μ∇²u + ∂p/∂x = 0 (Pressure driven)
L_u = ((- Laplacian{XCALibre.Linear}(mu_cst) + ScalarGrad{XCALibre.Linear,1}(one_cst, p) == Source(0.0)) → BCs.u) → solvers.u
L_v = ((- Laplacian{XCALibre.Linear}(mu_cst) + ScalarGrad{XCALibre.Linear,2}(one_cst, p) == Source(0.0)) → BCs.v) → solvers.v

# Continuity: β p + ∇·u - τ_rc ∇²p = 0
L_p = ((Si(beta_cst) - Laplacian{XCALibre.Linear}(tau_rc_cst) + VectorDiv{XCALibre.Linear,1}(one_cst, u) + VectorDiv{XCALibre.Linear,2}(one_cst, v) == Source(0.0)) → BCs.p) → solvers.p

sys = MonolithicSystem([L_u(u), L_v(v), L_p(p)], [u, v, p])
res = XCALibre.Solve.solve_monolithic!(sys, (BCs.u, BCs.v, BCs.p), config)

report_results("Compressible Stokes Pressure", res, u, p)
