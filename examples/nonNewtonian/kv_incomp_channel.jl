# =============================================================================
# Incompressible Kelvin-Voigt Benchmark — XCALibre.jl
# =============================================================================
#
# This example mimics the 'incompressible_KelvinVoigt' OpenFOAM benchmark.
# It solves the standard mixed-form Stokes equations coupled to a 
# Kelvin-Voigt viscoelastic constitutive law.
#
# Equations:
#   - Momentum: -∇·(2η_s ε(u)) - ∇·τ_p + ∇p = f
#   - Constitutive: τ_p = 2η_p D + G γ
#
# Integration:
#   γ^{n+1} = γ^n + 2D^{n+1} Δt
#   => τ_p = 2(η_p + G Δt) D + G γ^n
#
# Geometry: 2D Straight Channel (quad40.unv, 40x40 mesh, 1m x 1m)
# Parameters: mu_s = 1.0, mu_p = 1.0, G = 1.0, dt = 0.01
# BCs: Neumann (zeroGradient) at inlet/outlet.
# =============================================================================

using XCALibre
using Accessors
using LinearAlgebra
using Printf
using Statistics

# ── 1. Mesh and Hardware ──────────────────────────────────────────────────────
grids_dir = pkgdir(XCALibre, "examples", "0_GRIDS")
mesh_path = joinpath(grids_dir, "quad40.unv")
mesh      = UNV2D_mesh(mesh_path, scale=0.025)

backend   = CPU(); workgroup = 1024
hardware  = Hardware(backend=backend, workgroup=workgroup)
mesh_dev  = adapt(backend, mesh)

# ── 2. Material and Forcing Parameters ────────────────────────────────────────
mu_s_val    = 1.0       # Solvent viscosity (Unitary)
mu_p_val    = 1.0       # Polymeric viscosity
G_val       = 1.0       # Elastic modulus
dt          = 0.01      # Time step
force_x     = 1.0       # Unitary forcing
tau_rc_val  = 0.1       # Rhie-Chow stabilization

@info "Incomp KV Benchmark: mu_s=$mu_s_val, mu_p=$mu_p_val, G=$G_val, dt=$dt"

# ── 3. Fields ─────────────────────────────────────────────────────────────────
u   = ScalarField(mesh_dev); initialise!(u, 0.0)
v   = ScalarField(mesh_dev); initialise!(v, 0.0)
p   = ScalarField(mesh_dev); initialise!(p, 0.0)
txx = ScalarField(mesh_dev); initialise!(txx, 0.0)
tyy = ScalarField(mesh_dev); initialise!(tyy, 0.0)
txy = ScalarField(mesh_dev); initialise!(txy, 0.0)

# Cumulative strain fields (gamma)
gxx = ScalarField(mesh_dev); initialise!(gxx, 0.0)
gyy = ScalarField(mesh_dev); initialise!(gyy, 0.0)
gxy = ScalarField(mesh_dev); initialise!(gxy, 0.0)

# ── 4. Boundary Conditions ───────────────────────────────────────────────────
BCs = assign(
    region = mesh_dev,
    (
        u = [Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0), Zerogradient(:inlet), Zerogradient(:outlet)],
        v = [Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0), Zerogradient(:inlet), Zerogradient(:outlet)],
        p = [Zerogradient(:top), Zerogradient(:bottom), Zerogradient(:inlet), Zerogradient(:outlet)],
        txx = [Zerogradient(:top), Zerogradient(:bottom), Zerogradient(:inlet), Zerogradient(:outlet)],
        tyy = [Zerogradient(:top), Zerogradient(:bottom), Zerogradient(:inlet), Zerogradient(:outlet)],
        txy = [Zerogradient(:top), Zerogradient(:bottom), Zerogradient(:inlet), Zerogradient(:outlet)],
    )
)

# ── 5. Solver Setup ───────────────────────────────────────────────────────────
solvers = (
    u = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    v = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    p = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    txx = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    tyy = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    txy = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
)
schemes = (u=Schemes(), v=Schemes(), p=Schemes(), txx=Schemes(), tyy=Schemes(), txy=Schemes())
runtime = Runtime(iterations=10, write_interval=-1, time_step=dt)

config = Configuration(solvers=solvers, schemes=schemes,
                       runtime=runtime, hardware=hardware, boundaries=BCs)

# ── 6. Monolithic Equations ───────────────────────────────────────────────────
one_cst     = ConstantScalar(1.0)
mu_s_cst    = ConstantScalar(mu_s_val)
mu_eff_cst  = ConstantScalar(mu_p_val + G_val * dt) # η_eff for stress coupling
two_mu_eff  = ConstantScalar(2.0 * (mu_p_val + G_val * dt))
tau_rc_cst  = ConstantScalar(tau_rc_val)

∇u = Grad{Gauss}(u); uf = FaceScalarField(mesh_dev)
∇v = Grad{Gauss}(v); vf = FaceScalarField(mesh_dev)

@info "Starting Kelvin-Voigt loop..."

for step in 1:runtime.iterations
    # 1. Momentum + Stress + Pressure (6-field system)
    L_u = ((
        - Laplacian{Linear}(mu_s_cst)
        - ScalarGrad{Linear,1}(one_cst, txx)
        - ScalarGrad{Linear,2}(one_cst, txy)
        + ScalarGrad{Linear,1}(one_cst, p)
        == Source(force_x)
    ) → BCs.u) → solvers.u

    L_v = ((
        - Laplacian{Linear}(mu_s_cst)
        - ScalarGrad{Linear,1}(one_cst, txy)
        - ScalarGrad{Linear,2}(one_cst, tyy)
        + ScalarGrad{Linear,2}(one_cst, p)
        == Source(0.0)
    ) → BCs.v) → solvers.v

    L_p = ((
        - Laplacian{Linear}(tau_rc_cst)
        + VectorDiv{Linear,1}(one_cst, u)
        + VectorDiv{Linear,2}(one_cst, v)
        == Source(0.0)
    ) → BCs.p) → solvers.p

    L_txx = ((
        Si(one_cst)
        - ScalarGrad{Linear,1}(two_mu_eff, u)
        == Source(ConstantScalar(G_val) * gxx)
    ) → BCs.txx) → solvers.txx

    L_tyy = ((
        Si(one_cst)
        - ScalarGrad{Linear,2}(two_mu_eff, v)
        == Source(ConstantScalar(G_val) * gyy)
    ) → BCs.tyy) → solvers.tyy

    L_txy = ((
        Si(one_cst)
        - ScalarGrad{Linear,2}(mu_eff_cst, u)
        - ScalarGrad{Linear,1}(mu_eff_cst, v)
        == Source(ConstantScalar(G_val) * gxy)
    ) → BCs.txy) → solvers.txy

    sys = MonolithicSystem([L_u(u), L_v(v), L_p(p), L_txx(txx), L_tyy(tyy), L_txy(txy)], [u, v, p, txx, tyy, txy])
    setReference!(sys.equations[3], 0.0, 1, config)
    res = solve_monolithic!(sys, (BCs.u, BCs.v, BCs.p, BCs.txx, BCs.tyy, BCs.txy), config)

    # 2. Update strain (gamma)
    grad!(∇u, uf, u, BCs.u, nothing, config)
    grad!(∇v, vf, v, BCs.v, nothing, config)
    @. gxx.values += 2.0 * ∇u.result.x.values * dt
    @. gyy.values += 2.0 * ∇v.result.y.values * dt
    @. gxy.values += (∇u.result.y.values + ∇v.result.x.values) * dt

    @printf("Step %d, Residual: %.2e, max|u|: %.4f\n", step, res, maximum(abs.(u.values)))
end

save_output(u, "kv_incomp_u", 0.0, config)
@info "Benchmark kv_incomp finished."
