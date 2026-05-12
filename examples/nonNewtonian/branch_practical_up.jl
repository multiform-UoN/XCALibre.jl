# =============================================================================
# Practical Branch Benchmark (Robust) — XCALibre.jl
# =============================================================================

using XCALibre
using Accessors
using LinearAlgebra
using Printf
using Statistics

# ── 1. Parse Arguments ──
MODEL    = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :Stokes
BC_TYPE  = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :Neumann
GEOMETRY = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :Channel

@info "Practical Branch Benchmark" MODEL BC_TYPE GEOMETRY

# ── 2. Mesh Selection ──
grids_dir = pkgdir(XCALibre, "examples", "0_GRIDS")
if GEOMETRY == :Channel
    mesh      = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.025)
    force_vec = [1.0, 0.0, 0.0]
elseif GEOMETRY == :Bend
    mesh_dir  = "/Volumes/OpenFOAM/mixed_viscoelasticity/openfoam_cases/stokes3plus3_bend/viscoelasticChannelBend_stokes_compressibleSolid_KelvinVoigt/constant/polyMesh"
    mesh      = FOAM3D_mesh(mesh_dir, scale=1.0)
    force_vec = [1.0, 1.0, 0.0]
end

backend  = CPU(); workgroup = 1024
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# ── 3. Material Parameters ──
mu_s = 1.0; mu_p = 1.0; G = 1.0; dt = 0.01; tau_rc = 0.1

# ── 4. Fields ──
u = ScalarField(mesh_dev); initialise!(u, 0.0)
v = ScalarField(mesh_dev); initialise!(v, 0.0)
p = ScalarField(mesh_dev); initialise!(p, 0.0)
gxx = ScalarField(mesh_dev); initialise!(gxx, 0.0)
gyy = ScalarField(mesh_dev); initialise!(gyy, 0.0)
gxy = ScalarField(mesh_dev); initialise!(gxy, 0.0)
src_u = ScalarField(mesh_dev); initialise!(src_u, force_vec[1])
src_v = ScalarField(mesh_dev); initialise!(src_v, force_vec[2])

# ── 5. BCs ──
if GEOMETRY == :Channel
    u_bcs = [Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0), Zerogradient(:inlet), Zerogradient(:outlet)]
    v_bcs = [Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0), Zerogradient(:inlet), Zerogradient(:outlet)]
    p_bcs = [Zerogradient(:top), Zerogradient(:bottom), Zerogradient(:inlet), Zerogradient(:outlet)]
elseif GEOMETRY == :Bend
    u_bcs = [Dirichlet(:walls, 0.0), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)]
    v_bcs = [Dirichlet(:walls, 0.0), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)]
    p_bcs = [Zerogradient(:walls), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)]
end
BCs = assign(region=mesh_dev, (u=u_bcs, v=v_bcs, p=p_bcs))
u_bcs, v_bcs, p_bcs = BCs.u, BCs.v, BCs.p

# ── 6. Solver Setup ──
solvers = (
    u = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0),
    v = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0),
    p = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0),
)
nsteps = 20
config = Configuration(solvers=solvers, schemes=(u=Schemes(), v=Schemes(), p=Schemes()),
                       runtime=Runtime(iterations=1, write_interval=-1, time_step=dt), hardware=hardware,
                       boundaries=(u=u_bcs, v=v_bcs, p=p_bcs))

# ── 7. Equations (KV Elimination) ──
mu_eff = (MODEL == :KelvinVoigt) ? (mu_s + mu_p + G*dt) : mu_s
mu_cst = ConstantScalar(mu_eff); one_cst = ConstantScalar(1.0); tau_rc_cst = ConstantScalar(tau_rc)
graddiv_cst = ConstantScalar(0.0) # practical (u,p,tau) branch uses extra stress, not total-stress grad-div

L_u = ((- Laplacian{Linear}(mu_cst) - GradDiv{Linear,1,1}(graddiv_cst) - GradDiv{Linear,1,2}(graddiv_cst, v) + ScalarGrad{Linear,1}(one_cst, p)) == Source(src_u))
L_v = ((- Laplacian{Linear}(mu_cst) - GradDiv{Linear,2,1}(graddiv_cst, u) - GradDiv{Linear,2,2}(graddiv_cst) + ScalarGrad{Linear,2}(one_cst, p)) == Source(src_v))
L_p = ((- Laplacian{Linear}(tau_rc_cst) + VectorDiv{Linear,1}(one_cst, u) + VectorDiv{Linear,2}(one_cst, v)) == Source(0.0))

sys = MonolithicSystem([L_u(u), L_v(v), L_p(p)], [u, v, p])
if GEOMETRY == :Channel; setReference!(sys.equations[3], 0.0, 1, config); end

# ── 8. Loop ──
@info "Starting Loop..."
for step in 1:nsteps
    res = solve_monolithic!(sys, (u_bcs, v_bcs, p_bcs), config; reference=(3, 0.0, 1))
    if MODEL == :KelvinVoigt
        ∇u = Grad{Gauss}(u); uf = FaceScalarField(mesh_dev); ∇v = Grad{Gauss}(v); vf = FaceScalarField(mesh_dev)
        grad!(∇u, uf, u, u_bcs, nothing, config); grad!(∇v, vf, v, v_bcs, nothing, config)
        @. gxx.values += 2.0 * ∇u.result.x.values * dt
        @. gyy.values += 2.0 * ∇v.result.y.values * dt
        @. gxy.values += (∇u.result.y.values + ∇v.result.x.values) * dt
        # (∇·Gγ term could be added to src_u/v here)
    end
    @printf("Step %d, Residual: %.2e, max|u|: %.4f\n", step, res, maximum(abs.(u.values)))
    if res < 1e-10 && step > 1; break; end
end
@info "Benchmark Complete."
