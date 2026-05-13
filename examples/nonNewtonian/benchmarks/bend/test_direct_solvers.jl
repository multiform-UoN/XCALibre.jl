using XCALibre
using Accessors
using LinearAlgebra
using Printf
using Statistics
using SparseArrays
using SuiteSparse

include("../benchmark_utils.jl")

mesh, mesh_dev = get_bend_mesh()

function run_direct(name, mu_s_val, mu_p_val, lam_val)
    @info "Running Direct Solve: $name"
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
                           runtime=Runtime(iterations=1, write_interval=-1, time_step=0.01), hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=BCs)

    one_cst = ConstantScalar(1.0)
    mu_s_cst = ConstantScalar(mu_s_val)
    tau_rc_cst = ConstantScalar(0.1)
    inv_lam = ConstantScalar(1.0/lam_val)
    two_mu_lam = ConstantScalar(2.0*mu_p_val/lam_val)
    mu_lam_cst = ConstantScalar(mu_p_val/lam_val)

    # Simplified Maxwell (No advection, straight substitution for direct solve verification)
    L_u = (( - Laplacian{XCALibre.Linear}(mu_s_cst) - ScalarGrad{XCALibre.Linear,1}(one_cst, txx) - ScalarGrad{XCALibre.Linear,2}(one_cst, txy) + ScalarGrad{XCALibre.Linear,1}(one_cst, p) == Source(1.0) ) → BCs.u) → solvers.u
    L_v = (( - Laplacian{XCALibre.Linear}(mu_s_cst) - ScalarGrad{XCALibre.Linear,1}(one_cst, txy) - ScalarGrad{Linear,2}(one_cst, tyy) + ScalarGrad{XCALibre.Linear,2}(one_cst, p) == Source(1.0) ) → BCs.v) → solvers.v
    L_p = (( - Laplacian{XCALibre.Linear}(tau_rc_cst) + VectorDiv{XCALibre.Linear,1}(one_cst, u) + VectorDiv{XCALibre.Linear,2}(one_cst, v) == Source(0.0) ) → BCs.p) → solvers.p

    L_txx = (( Si(inv_lam) - ScalarGrad{XCALibre.Linear,1}(two_mu_lam, u) == Source(0.0) ) → BCs.txx) → solvers.txx
    L_tyy = (( Si(inv_lam) - ScalarGrad{XCALibre.Linear,2}(two_mu_lam, v) == Source(0.0) ) → BCs.tyy) → solvers.tyy
    L_txy = (( Si(inv_lam) - ScalarGrad{XCALibre.Linear,2}(mu_lam_cst, u) - ScalarGrad{XCALibre.Linear,1}(mu_lam_cst, v) == Source(0.0) ) → BCs.txy) → solvers.txy

    sys = MonolithicSystem([L_u(u), L_v(v), L_p(p), L_txx(txx), L_tyy(tyy), L_txy(txy)], [u, v, p, txx, tyy, txy])
    
    A_csr, b_mono = assemble_monolithic_system(sys, (BCs.u, BCs.v, BCs.p, BCs.txx, BCs.tyy, BCs.txy), config)
    
    # Exact Pinning of pressure at center of bend
    n_cells = length(mesh_dev.cells)
    ref_idx = n_cells ÷ 2
    mono_row = 2 * n_cells + ref_idx 
    
    A_julia = get_sparse_matrix(A_csr)
    A_julia[mono_row, :] .= 0.0
    A_julia[mono_row, mono_row] = 1.0
    b_mono[mono_row] = 0.0
    
    t0 = time()
    x = A_julia \ b_mono
    t1 = time()
    
    XCALibre.Solve.update_fields!(sys, x)
    
    res = norm(A_julia * x - b_mono)
    u_max = maximum(abs.(u.values))
    tau_max = maximum(abs, vcat(txx.values, tyy.values, txy.values))
    
    @printf("[%s] Time: %.3fs | Residual: %.2e | max|u|: %.4e | max|tau|: %.4e\n", name, t1-t0, res, u_max, tau_max)
end

run_direct("Stokes-like (mu_s=1, mu_p=0)", 1.0, 0.0, 1.0)
run_direct("Oldroyd-B-like (mu_s=1, mu_p=1)", 1.0, 1.0, 1.0)
run_direct("Maxwell (mu_s=1e-6, mu_p=1)", 1e-6, 1.0, 1.0)

