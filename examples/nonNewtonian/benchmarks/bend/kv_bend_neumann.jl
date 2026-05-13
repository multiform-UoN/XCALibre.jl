# =============================================================================
# Incompressible Kelvin-Voigt: L-Bend Channel (Neumann/Body Force Driven)
# =============================================================================

include("../benchmark_utils.jl")

mesh, mesh_dev = get_bend_mesh()
u = ScalarField(mesh_dev); initialise!(u, 0.0)
v = ScalarField(mesh_dev); initialise!(v, 0.0)
p = ScalarField(mesh_dev); initialise!(p, 0.0)
txx = ScalarField(mesh_dev); initialise!(txx, 0.0)
tyy = ScalarField(mesh_dev); initialise!(tyy, 0.0)
txy = ScalarField(mesh_dev); initialise!(txy, 0.0)

gxx = ScalarField(mesh_dev); initialise!(gxx, 0.0)
gyy = ScalarField(mesh_dev); initialise!(gyy, 0.0)
gxy = ScalarField(mesh_dev); initialise!(gxy, 0.0)

BCs = assign(
    region = mesh_dev,
    (
        u = [Dirichlet(:walls, 0.0), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)],
        v = [Dirichlet(:walls, 0.0), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)],
        p = [Zerogradient(:walls), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)],
        txx = [Zerogradient(:walls), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)],
        tyy = [Zerogradient(:walls), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)],
        txy = [Zerogradient(:walls), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)],
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
mu_s_val = 1.0; mu_p_val = 1.0; G_val = 1.0; dt = 0.01
mu_s_cst = ConstantScalar(mu_s_val)
mu_eff_val = mu_p_val + G_val * dt
mu_eff_cst = ConstantScalar(mu_eff_val)
two_mu_eff = ConstantScalar(2.0 * mu_eff_val)
tau_rc_cst = ConstantScalar(0.1)

∇u = Grad{Gauss}(u); uf = FaceScalarField(mesh_dev)
∇v = Grad{Gauss}(v); vf = FaceScalarField(mesh_dev)

for step in 1:config.runtime.iterations
    L_u = (( - Laplacian{XCALibre.Linear}(mu_s_cst) - ScalarGrad{XCALibre.Linear,1}(one_cst, txx) - ScalarGrad{XCALibre.Linear,2}(one_cst, txy) + ScalarGrad{XCALibre.Linear,1}(one_cst, p) == Source(1.0) ) → BCs.u) → solvers.u
    L_v = (( - Laplacian{XCALibre.Linear}(mu_s_cst) - ScalarGrad{XCALibre.Linear,1}(one_cst, txy) - ScalarGrad{Linear,2}(one_cst, tyy) + ScalarGrad{XCALibre.Linear,2}(one_cst, p) == Source(1.0) ) → BCs.v) → solvers.v
    L_p = (( - Laplacian{XCALibre.Linear}(tau_rc_cst) + VectorDiv{XCALibre.Linear,1}(one_cst, u) + VectorDiv{XCALibre.Linear,2}(one_cst, v) == Source(0.0) ) → BCs.p) → solvers.p

    L_txx = (( Si(one_cst) - ScalarGrad{XCALibre.Linear,1}(two_mu_eff, u) == Source(XCALibre.ModelFramework.ScaledFlux(gxx, G_val)) ) → BCs.txx) → solvers.txx
    L_tyy = (( Si(one_cst) - ScalarGrad{XCALibre.Linear,2}(two_mu_eff, v) == Source(XCALibre.ModelFramework.ScaledFlux(gyy, G_val)) ) → BCs.tyy) → solvers.tyy
    L_txy = (( Si(one_cst) - ScalarGrad{XCALibre.Linear,2}(mu_eff_cst, u) - ScalarGrad{XCALibre.Linear,1}(mu_eff_cst, v) == Source(XCALibre.ModelFramework.ScaledFlux(gxy, G_val)) ) → BCs.txy) → solvers.txy

    sys = MonolithicSystem([L_u(u), L_v(v), L_p(p), L_txx(txx), L_tyy(tyy), L_txy(txy)], [u, v, p, txx, tyy, txy])
    res = XCALibre.Solve.solve_monolithic!(sys, (BCs.u, BCs.v, BCs.p, BCs.txx, BCs.tyy, BCs.txy), config; reference=(3, 0.0, 1))

    grad!(∇u, uf, u, BCs.u, nothing, config)
    grad!(∇v, vf, v, BCs.v, nothing, config)
    
    gxx.values .+= 2.0 .* ∇u.result.x.values .* dt
    gyy.values .+= 2.0 .* ∇v.result.y.values .* dt
    gxy.values .+= (∇u.result.y.values .+ ∇v.result.x.values) .* dt

    if step == config.runtime.iterations
        report_results("KV Bend Neumann", res, u, p, txx)
    end
end
