# =============================================================================
# Incompressible Oldroyd-B: L-Bend Channel (Neumann/Body Force Driven)
# =============================================================================

include("../benchmark_utils.jl")

mesh, mesh_dev = get_bend_mesh()
u = ScalarField(mesh_dev); initialise!(u, 0.0)
v = ScalarField(mesh_dev); initialise!(v, 0.0)
p = ScalarField(mesh_dev); initialise!(p, 0.0)
txx = ScalarField(mesh_dev); initialise!(txx, 0.0)
tyy = ScalarField(mesh_dev); initialise!(tyy, 0.0)
txy = ScalarField(mesh_dev); initialise!(txy, 0.0)

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
                       runtime=Runtime(iterations=10, write_interval=-1, time_step=0.01), hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=BCs)

one_cst = ConstantScalar(1.0)
mu_s_cst = ConstantScalar(1.0)
tau_rc_cst = ConstantScalar(0.1)
inv_lam = ConstantScalar(1.0/1.0)
two_mu_lam = ConstantScalar(2.0*1.0/1.0)
mu_lam_cst = ConstantScalar(1.0/1.0)

mass_flux = FaceScalarField(mesh_dev)

for step in 1:config.runtime.iterations
    for fID in 1:length(mesh_dev.faces)
        face = mesh_dev.faces[fID]
        if fID > length(mesh_dev.boundary_cellsID)
            c1 = face.ownerCells[1]; c2 = face.ownerCells[2]
            uf_val = 0.5 * (u.values[c1] + u.values[c2])
            vf_val = 0.5 * (v.values[c1] + v.values[c2])
            mass_flux.values[fID] = (uf_val * face.normal[1] + vf_val * face.normal[2]) * face.area
        else
            c1 = face.ownerCells[1]
            mass_flux.values[fID] = (u.values[c1] * face.normal[1] + v.values[c1] * face.normal[2]) * face.area
        end
    end

    L_u = (( - Laplacian{XCALibre.Linear}(mu_s_cst) - ScalarGrad{XCALibre.Linear,1}(one_cst, txx) - ScalarGrad{XCALibre.Linear,2}(one_cst, txy) + ScalarGrad{XCALibre.Linear,1}(one_cst, p) == Source(1.0) ) → BCs.u) → solvers.u
    L_v = (( - Laplacian{XCALibre.Linear}(mu_s_cst) - ScalarGrad{XCALibre.Linear,1}(one_cst, txy) - ScalarGrad{Linear,2}(one_cst, tyy) + ScalarGrad{XCALibre.Linear,2}(one_cst, p) == Source(1.0) ) → BCs.v) → solvers.v
    L_p = (( - Laplacian{XCALibre.Linear}(tau_rc_cst) + VectorDiv{XCALibre.Linear,1}(one_cst, u) + VectorDiv{XCALibre.Linear,2}(one_cst, v) == Source(0.0) ) → BCs.p) → solvers.p

    L_txx = (( Si(inv_lam) + Divergence{XCALibre.Upwind}(mass_flux) - ScalarGrad{XCALibre.Linear,1}(two_mu_lam, u) == Source(0.0) ) → BCs.txx) → solvers.txx
    L_tyy = (( Si(inv_lam) + Divergence{XCALibre.Upwind}(mass_flux) - ScalarGrad{XCALibre.Linear,2}(two_mu_lam, v) == Source(0.0) ) → BCs.tyy) → solvers.tyy
    L_txy = (( Si(inv_lam) + Divergence{XCALibre.Upwind}(mass_flux) - ScalarGrad{XCALibre.Linear,2}(mu_lam_cst, u) - ScalarGrad{XCALibre.Linear,1}(mu_lam_cst, v) == Source(0.0) ) → BCs.txy) → solvers.txy

    sys = MonolithicSystem([L_u(u), L_v(v), L_p(p), L_txx(txx), L_tyy(tyy), L_txy(txy)], [u, v, p, txx, tyy, txy])
    res = XCALibre.Solve.solve_monolithic!(sys, (BCs.u, BCs.v, BCs.p, BCs.txx, BCs.tyy, BCs.txy), config; reference=(3, 0.0, 1))

    if step == config.runtime.iterations
        report_results("Oldroyd Bend Neumann", res, u, p, txx)
    end
end
