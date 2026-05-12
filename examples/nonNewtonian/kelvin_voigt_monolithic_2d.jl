# =============================================================================
# Kelvin-Voigt Viscoelastic Solver — Monolithic 5-field (u, v, txx, tyy, txy)
# =============================================================================
#
# Governing equations (Transient, Incompressible via Penalty):
#
#   Momentum:
#       - ∇·(2η_s ε(u)) - ∇·τ_p - ∇(λ_penalty ∇·u) = f
#
#   Constitutive (Kelvin-Voigt):
#       τ_p = 2η_p D + G γ
#       dγ/dt = 2D  => γ^{n+1} = γ^n + 2D^{n+1} Δt
#
#   Effective τ_p for monolithic coupling:
#       τ_p = 2(η_p + G Δt) D + G γ^n
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

# ── Parameters (All set to 1.0) ────────────────────────────────────────────────
eta_s   = 1.0        # Solvent viscosity
eta_p   = 1.0        # Polymeric viscosity
G       = 1.0        # Elastic modulus
dt      = 0.01       # Time step
penalty = 1e6        # Incompressibility penalty

@info "Kelvin-Voigt: eta_s=$eta_s, eta_p=$eta_p, G=$G, dt=$dt"

# ── Fields ────────────────────────────────────────────────────────────────────
u   = ScalarField(mesh_dev); initialise!(u, 0.0)
v   = ScalarField(mesh_dev); initialise!(v, 0.0)
txx = ScalarField(mesh_dev); initialise!(txx, 0.0)
tyy = ScalarField(mesh_dev); initialise!(tyy, 0.0)
txy = ScalarField(mesh_dev); initialise!(txy, 0.0)

# Cumulative strain fields (gamma)
gxx = ScalarField(mesh_dev); initialise!(gxx, 0.0)
gyy = ScalarField(mesh_dev); initialise!(gyy, 0.0)
gxy = ScalarField(mesh_dev); initialise!(gxy, 0.0)

# Helper fields for gradients
grad_ux = ScalarField(mesh_dev)
grad_uy = ScalarField(mesh_dev)
grad_vx = ScalarField(mesh_dev)
grad_vy = ScalarField(mesh_dev)

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
           txx = Schemes(laplacian=Linear),
           tyy = Schemes(laplacian=Linear),
           txy = Schemes(laplacian=Linear))
runtime = Runtime(iterations=10, write_interval=-1, time_step=dt)

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
eta_s_cst   = ConstantScalar(eta_s)
penalty_cst = ConstantScalar(penalty + eta_s)
one_cst     = ConstantScalar(1.0)
eta_eff_cst = ConstantScalar(eta_p + G * dt) # Effective viscosity for monolithic coupling

# ── Gradient Operators ────────────────────────────────────────────────────────
∇u = Grad{Gauss}(u); uf = FaceScalarField(mesh_dev)
∇v = Grad{Gauss}(v); vf = FaceScalarField(mesh_dev)

@info "Starting Kelvin-Voigt time loop..."

for step in 1:runtime.iterations
    # 1. Update gradients of current velocity
    grad!(∇u, uf, u, BCs.u, nothing, config)
    grad!(∇v, vf, v, BCs.v, nothing, config)
    
    # 2. Define equations
    # Momentum (U-eqn)
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

    # Stress (tau-eqn): txx = 2*eta_eff*D_xx + G*gxx_old
    # Note: 2*D_xx = 2*du/dx
    L_txx = ((
        Si(one_cst)                                    # +txx
        - ScalarGrad{Linear,1}(ConstantScalar(2.0 * (eta_p + G*dt)), u) # -2*eta_eff*du/dx
        == Source(ConstantScalar(G) * gxx)             # +G*gxx_old
    ) → BCs.txx) → solvers.txx

    L_tyy = ((
        Si(one_cst)
        - ScalarGrad{Linear,2}(ConstantScalar(2.0 * (eta_p + G*dt)), v)
        == Source(ConstantScalar(G) * gyy)
    ) → BCs.tyy) → solvers.tyy

    L_txy = ((
        Si(one_cst)
        - ScalarGrad{Linear,2}(ConstantScalar(eta_p + G*dt), u)
        - ScalarGrad{Linear,1}(ConstantScalar(eta_p + G*dt), v)
        == Source(ConstantScalar(G) * gxy)
    ) → BCs.txy) → solvers.txy

    # 3. Solve monolithic system
    sys = MonolithicSystem([L_u(u), L_v(v), L_txx(txx), L_tyy(tyy), L_txy(txy)], [u, v, txx, tyy, txy])
    res = solve_monolithic!(sys, (BCs.u, BCs.v, BCs.txx, BCs.tyy, BCs.txy), config)

    # 4. Update strain (gamma) for next step
    # g^{n+1} = g^n + 2*D^{n+1}*dt
    grad!(∇u, uf, u, BCs.u, nothing, config)
    grad!(∇v, vf, v, BCs.v, nothing, config)
    @. gxx.values += 2.0 * ∇u.result.x.values * dt
    @. gyy.values += 2.0 * ∇v.result.y.values * dt
    @. gxy.values += (∇u.result.y.values + ∇v.result.x.values) * dt

    @printf("Step %d, Residual: %.2e, max|u|: %.4f\n", step, res, maximum(abs.(u.values)))
end

save_output(u, "u", 0.0, config)
save_output(txx, "txx", 0.0, config)
@info "Kelvin-Voigt solve complete."
