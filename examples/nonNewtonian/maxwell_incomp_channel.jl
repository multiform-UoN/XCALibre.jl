# =============================================================================
# Incompressible Maxwell Benchmark — XCALibre.jl
# =============================================================================
#
# This example mimics the 'GMRT_Maxwell_incompFoam' OpenFOAM benchmark.
# It solves the standard mixed-form Stokes equations coupled to a 
# Maxwell viscoelastic constitutive law.
#
# Geometry: 2D Straight Channel (quad40.unv, 40x40 mesh, 1m x 1m)
# Parameters: mu_p = 1.0, lambda_p = 1.0, mu_s = 1e-6, dt = 0.01
# =============================================================================

using XCALibre
using Accessors
using LinearAlgebra
using Printf
using Statistics

# ── 1. Mesh and Hardware ──────────────────────────────────────────────────────
grids_dir = pkgdir(XCALibre, "examples", "0_GRIDS")
mesh      = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.025)
backend   = CPU(); workgroup = 1024
hardware  = Hardware(backend=backend, workgroup=workgroup)
mesh_dev  = adapt(backend, mesh)

# ── 2. Material and Forcing Parameters ────────────────────────────────────────
mu_s_val    = 1e-6      # Minimal solvent viscosity
mu_p_val    = 1.0       # Polymeric viscosity
lambda_p    = 1.0       # Relaxation time
dt          = 0.01      # Time step
force_x     = 1.0       # Unitary forcing
tau_rc_val  = 0.1       # Rhie-Chow stabilization

@info "Incomp Maxwell Benchmark: mu_p=$mu_p_val, lambda_p=$lambda_p, dt=$dt"

# ── 3. Fields ─────────────────────────────────────────────────────────────────
u   = ScalarField(mesh_dev); initialise!(u, 0.0)
v   = ScalarField(mesh_dev); initialise!(v, 0.0)
p   = ScalarField(mesh_dev); initialise!(p, 0.0)
txx = ScalarField(mesh_dev); initialise!(txx, 0.0)
tyy = ScalarField(mesh_dev); initialise!(tyy, 0.0)
txy = ScalarField(mesh_dev); initialise!(txy, 0.0)

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
config = Configuration(solvers=solvers, schemes=(u=Schemes(), v=Schemes(), p=Schemes(), txx=Schemes(), tyy=Schemes(), txy=Schemes()),
                       runtime=Runtime(iterations=10, write_interval=-1, time_step=dt), hardware=hardware, boundaries=BCs)

# ── 6. Monolithic Equations ───────────────────────────────────────────────────
one_cst     = ConstantScalar(1.0)
mu_s_cst    = ConstantScalar(mu_s_val)
tau_rc_cst  = ConstantScalar(tau_rc_val)
inv_lam     = ConstantScalar(1.0/lambda_p)
mu_lam      = ConstantScalar(mu_p_val/lambda_p)

∇u = Grad{Gauss}(u); uf = FaceScalarField(mesh_dev)
∇v = Grad{Gauss}(v); vf = FaceScalarField(mesh_dev)
mass_flux = FaceScalarField(mesh_dev)

@info "Starting Incomp Maxwell loop (Steady-State coupling)..."

for step in 1:runtime.iterations
    # Update mass flux for convection
    # (Simplified linear interpolation)
    for fID in 1:length(mesh_dev.faces)
        face = mesh_dev.faces[fID]
        if !face.boundary
            c1 = face.ownerCells[1]; c2 = face.ownerCells[2]
            uf_val = 0.5 * (u.values[c1] + u.values[c2])
            vf_val = 0.5 * (v.values[c1] + v.values[c2])
            mass_flux.values[fID] = (uf_val * face.normal[1] + vf_val * face.normal[2]) * face.area
        else
            c1 = face.ownerCells[1]
            mass_flux.values[fID] = (u.values[c1] * face.normal[1] + v.values[c1] * face.normal[2]) * face.area
        end
    end

    # Momentum
    L_u = (( - Laplacian{Linear}(mu_s_cst) - ScalarGrad{Linear,1}(one_cst, txx) - ScalarGrad{Linear,2}(one_cst, txy) + ScalarGrad{Linear,1}(one_cst, p) == Source(force_x) ) → BCs.u) → solvers.u
    L_v = (( - Laplacian{Linear}(mu_s_cst) - ScalarGrad{Linear,1}(one_cst, txy) - ScalarGrad{Linear,2}(one_cst, tyy) + ScalarGrad{Linear,2}(one_cst, p) == Source(0.0) ) → BCs.v) → solvers.v
    L_p = (( - Laplacian{Linear}(tau_rc_cst) + VectorDiv{Linear,1}(one_cst, u) + VectorDiv{Linear,2}(one_cst, v) == Source(0.0) ) → BCs.p) → solvers.p

    # Constitutive (Simplified steady: τ + λ(u·∇)τ = 2ηD)
    L_txx = (( Si(inv_lam) + Divergence{Upwind}(mass_flux) - ScalarGrad{Linear,1}(ConstantScalar(2.0*mu_lam), u) == Source(0.0) ) → BCs.txx) → solvers.txx
    L_tyy = (( Si(inv_lam) + Divergence{Upwind}(mass_flux) - ScalarGrad{Linear,2}(ConstantScalar(2.0*mu_lam), v) == Source(0.0) ) → BCs.tyy) → solvers.tyy
    L_txy = (( Si(inv_lam) + Divergence{Upwind}(mass_flux) - ScalarGrad{Linear,2}(mu_lam, u) - ScalarGrad{Linear,1}(mu_lam, v) == Source(0.0) ) → BCs.txy) → solvers.txy

    sys = MonolithicSystem([L_u(u), L_v(v), L_p(p), L_txx(txx), L_tyy(tyy), L_txy(txy)], [u, v, p, txx, tyy, txy])
    setReference!(sys.equations[3], 0.0, 1, config)
    res = solve_monolithic!(sys, (BCs.u, BCs.v, BCs.p, BCs.txx, BCs.tyy, BCs.txy), config)

    @printf("Step %d, Residual: %.2e, max|u|: %.4f\n", step, res, maximum(abs.(u.values)))
end

save_output(u, "maxwell_incomp_u", 0.0, config)
@info "Benchmark maxwell_incomp finished."
