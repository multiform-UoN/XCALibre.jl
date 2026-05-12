# =============================================================================
# Oldroyd-B Viscoelastic Solver — Monolithic 5-field (u, v, txx, tyy, txy)
# =============================================================================

using XCALibre
using Accessors
using LinearAlgebra
using Printf
using Statistics

# ── Mesh ──────────────────────────────────────────────────────────────────────
grids_dir = pkgdir(XCALibre, "examples", "0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.025)
backend   = CPU(); workgroup = 1024
hardware  = Hardware(backend=backend, workgroup=workgroup)
mesh_dev  = adapt(backend, mesh)

# ── Parameters ────────────────────────────────────────────────────────────────
eta_s   = 0.5
eta_p   = 0.5
lambda_p = 0.1
penalty = 1e6

@info "Oldroyd-B: eta_s=$eta_s, eta_p=$eta_p, lambda_p=$lambda_p, Wi=$(lambda_p*1.0/1.0)"

# ── Fields ────────────────────────────────────────────────────────────────────
u   = ScalarField(mesh_dev); initialise!(u, 0.0)
v   = ScalarField(mesh_dev); initialise!(v, 0.0)
txx = ScalarField(mesh_dev); initialise!(txx, 0.0)
tyy = ScalarField(mesh_dev); initialise!(tyy, 0.0)
txy = ScalarField(mesh_dev); initialise!(txy, 0.0)

# Helper fields for linearization
two_grad_ux = ScalarField(mesh_dev)
two_grad_uy = ScalarField(mesh_dev)
two_grad_vx = ScalarField(mesh_dev)
two_grad_vy = ScalarField(mesh_dev)
grad_div_u  = ScalarField(mesh_dev)
grad_uy_neg = ScalarField(mesh_dev)
grad_vx_neg = ScalarField(mesh_dev)
mass_flux = FaceScalarField(mesh_dev)

# ── Solver / config ───────────────────────────────────────────────────────────
solvers = (
    u = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    v = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    txx = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    tyy = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    txy = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
)
schemes = (u = Schemes(laplacian=Linear),
           v = Schemes(laplacian=Linear),
           txx = Schemes(laplacian=Linear, divergence=Upwind),
           tyy = Schemes(laplacian=Linear, divergence=Upwind),
           txy = Schemes(laplacian=Linear, divergence=Upwind))
runtime = Runtime(iterations=20, write_interval=-1, time_step=1.0)

BCs = assign(
    region = mesh_dev,
    (
        u = [Dirichlet(:top, 1.0), Dirichlet(:bottom, 0.0), Dirichlet(:inlet, 0.0), Dirichlet(:outlet, 0.0)],
        v = [Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0), Dirichlet(:inlet, 0.0), Dirichlet(:outlet, 0.0)],
        txx = [Zerogradient(:top), Zerogradient(:bottom), Zerogradient(:inlet), Zerogradient(:outlet)],
        tyy = [Zerogradient(:top), Zerogradient(:bottom), Zerogradient(:inlet), Zerogradient(:outlet)],
        txy = [Zerogradient(:top), Zerogradient(:bottom), Zerogradient(:inlet), Zerogradient(:outlet)],
    )
)
config = Configuration(solvers=solvers, schemes=schemes,
                       runtime=runtime, hardware=hardware, boundaries=BCs)

# ── Constants ─────────────────────────────────────────────────────────────────
eta_s_cst = ConstantScalar(eta_s)
penalty_cst = ConstantScalar(penalty + eta_s)
one_cst   = ConstantScalar(1.0)

# ── Gradient Operators for Linearisation ──────────────────────────────────────
∇u = Grad{Gauss}(u); uf = FaceScalarField(mesh_dev)
∇v = Grad{Gauss}(v); vf = FaceScalarField(mesh_dev)

@info "Starting Monolithic Oldroyd-B solve..."

for iter in 1:runtime.iterations
    grad!(∇u, uf, u, BCs.u, nothing, config)
    grad!(∇v, vf, v, BCs.v, nothing, config)
    @. two_grad_ux.values = -2.0 * ∇u.result.x.values
    @. two_grad_uy.values = -2.0 * ∇u.result.y.values
    @. two_grad_vx.values = -2.0 * ∇v.result.x.values
    @. two_grad_vy.values = -2.0 * ∇v.result.y.values
    @. grad_div_u.values  = -(∇u.result.x.values + ∇v.result.y.values)
    @. grad_uy_neg.values = -∇u.result.y.values
    @. grad_vx_neg.values = -∇v.result.x.values

    n_bfaces = length(mesh_dev.boundary_cellsID)
    for fID in 1:length(mesh_dev.faces)
        face = mesh_dev.faces[fID]
        if fID > n_bfaces
            c1 = face.ownerCells[1]; c2 = face.ownerCells[2]
            uf_val = 0.5 * (u.values[c1] + u.values[c2])
            vf_val = 0.5 * (v.values[c1] + v.values[c2])
            mass_flux.values[fID] = (uf_val * face.normal[1] + vf_val * face.normal[2]) * face.area
        else
            c1 = face.ownerCells[1]
            mass_flux.values[fID] = (u.values[c1] * face.normal[1] + v.values[c1] * face.normal[2]) * face.area
        end
    end

    coeff_self = ConstantScalar(1.0/lambda_p)
    strain_xx_coeff = ConstantScalar(2.0*eta_p/lambda_p)
    strain_yy_coeff = ConstantScalar(2.0*eta_p/lambda_p)
    strain_xy_coeff = ConstantScalar(eta_p/lambda_p)

    L_u = ((
        - Laplacian{Linear}(eta_s_cst)
        - GradDiv{Linear,1,1}(penalty_cst)
        - GradDiv{Linear,1,2}(penalty_cst, v)
        - ScalarGrad{Linear,1}(one_cst, txx)
        - ScalarGrad{Linear,2}(one_cst, txy)
        == Source(0.0)
    ) → BCs.u) → solvers.u

    L_v = ((
        - Laplacian{Linear}(eta_s_cst)
        - GradDiv{Linear,2,1}(penalty_cst, u)
        - GradDiv{Linear,2,2}(penalty_cst)
        - ScalarGrad{Linear,1}(one_cst, txy)
        - ScalarGrad{Linear,2}(one_cst, tyy)
        == Source(0.0)
    ) → BCs.v) → solvers.v

    L_txx = ((
        Si(coeff_self)
        + Divergence{Upwind}(mass_flux)
        + Si(two_grad_ux)
        + Si(two_grad_uy, txy)
        - ScalarGrad{Linear,1}(strain_xx_coeff, u)
        == Source(0.0)
    ) → BCs.txx) → solvers.txx

    L_tyy = ((
        Si(coeff_self)
        + Divergence{Upwind}(mass_flux)
        + Si(two_grad_vy)
        + Si(two_grad_vx, txy)
        - ScalarGrad{Linear,2}(strain_yy_coeff, v)
        == Source(0.0)
    ) → BCs.tyy) → solvers.tyy

    L_txy = ((
        Si(coeff_self)
        + Divergence{Upwind}(mass_flux)
        + Si(grad_div_u)
        + Si(grad_uy_neg, tyy)
        + Si(grad_vx_neg, txx)
        - ScalarGrad{Linear,2}(strain_xy_coeff, u)
        - ScalarGrad{Linear,1}(strain_xy_coeff, v)
        == Source(0.0)
    ) → BCs.txy) → solvers.txy

    sys = MonolithicSystem([L_u(u), L_v(v), L_txx(txx), L_tyy(tyy), L_txy(txy)], [u, v, txx, tyy, txy])
    res = solve_monolithic!(sys, (BCs.u, BCs.v, BCs.txx, BCs.tyy, BCs.txy), config)

    @printf("Newton Iteration %d, Linear Residual: %.2e, max|u|: %.4f\n", iter, res, maximum(abs.(u.values)))
    if res < 1e-8 && iter > 1; break; end
end
@info "Oldroyd-B solve complete."
