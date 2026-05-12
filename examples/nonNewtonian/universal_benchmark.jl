# =============================================================================
# Universal Non-Newtonian Benchmark (V3) — XCALibre.jl
# =============================================================================

using XCALibre
using Accessors
using LinearAlgebra
using Printf
using Statistics

# ── 1. Parse Arguments ──
MODEL    = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :OldroydB
REGIME   = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :Compressible
BC_TYPE  = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :Neumann
GEOMETRY = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :Channel

@info "Non-Newtonian Benchmark" MODEL REGIME BC_TYPE GEOMETRY

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
mu_s = 1.0; mu_p = 1.0; lambda_p = 1.0; G = 1.0; penalty = 1e6; dt = 0.01; tau_rc = 0.1

# ── 4. Fields ──
u = ScalarField(mesh_dev); initialise!(u, 0.0)
v = ScalarField(mesh_dev); initialise!(v, 0.0)
p = ScalarField(mesh_dev); initialise!(p, 0.0)
txx = ScalarField(mesh_dev); initialise!(txx, 0.0)
tyy = ScalarField(mesh_dev); initialise!(tyy, 0.0)
txy = ScalarField(mesh_dev); initialise!(txy, 0.0)
gxx = ScalarField(mesh_dev); initialise!(gxx, 0.0)
gyy = ScalarField(mesh_dev); initialise!(gyy, 0.0)
gxy = ScalarField(mesh_dev); initialise!(gxy, 0.0)

# ── 5. BCs ──
if GEOMETRY == :Channel
    if BC_TYPE == :Neumann
        u_bcs = [Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0), Zerogradient(:inlet), Zerogradient(:outlet)]
        v_bcs = [Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0), Zerogradient(:inlet), Zerogradient(:outlet)]
        p_bcs = [Zerogradient(:top), Zerogradient(:bottom), Zerogradient(:inlet), Zerogradient(:outlet)]
        t_bcs = [Zerogradient(:top), Zerogradient(:bottom), Zerogradient(:inlet), Zerogradient(:outlet)]
    else
        u_bcs = [Periodic(:inlet, :outlet), Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0)]
        v_bcs = [Periodic(:inlet, :outlet), Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0)]
        p_bcs = [Periodic(:inlet, :outlet), Zerogradient(:top), Zerogradient(:bottom)]
        t_bcs = [Periodic(:inlet, :outlet), Zerogradient(:top), Zerogradient(:bottom)]
    end
elseif GEOMETRY == :Bend
    u_bcs = [Dirichlet(:walls, 0.0), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)]
    v_bcs = [Dirichlet(:walls, 0.0), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)]
    p_bcs = [Zerogradient(:walls), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)]
    t_bcs = [Zerogradient(:walls), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)]
end

# ── 6. Solver Setup ──
solvers = (
    u = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0),
    v = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0),
    p = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0),
    txx = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0),
    tyy = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0),
    txy = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0),
)
config = Configuration(solvers=solvers, schemes=(u=Schemes(), v=Schemes(), p=Schemes(), txx=Schemes(), tyy=Schemes(), txy=Schemes()),
                       runtime=Runtime(iterations=10, write_interval=-1, time_step=dt), hardware=hardware, 
                       boundaries=(u=u_bcs, v=v_bcs, p=p_bcs, txx=t_bcs, tyy=t_bcs, txy=t_bcs))
runtime = config.runtime

# ── 7. Build Equations ──
mu_s_cst = ConstantScalar(mu_s); mu_p_cst = ConstantScalar(mu_p); one_cst = ConstantScalar(1.0)
penalty_cst = ConstantScalar(penalty + mu_s); tau_rc_cst = ConstantScalar(tau_rc)

L_u_op = (- Laplacian{Linear}(mu_s_cst) - GradDiv{Linear,1,1}(REGIME == :Compressible ? penalty_cst : mu_s_cst))
L_v_op = (- Laplacian{Linear}(mu_s_cst) - GradDiv{Linear,2,2}(REGIME == :Compressible ? penalty_cst : mu_s_cst))

if REGIME == :Incompressible
    L_u_op = L_u_op + ScalarGrad{Linear,1}(one_cst, p)
    L_v_op = L_v_op + ScalarGrad{Linear,2}(one_cst, p)
end

if MODEL != :Stokes
    L_u_op = L_u_op - ScalarGrad{Linear,1}(one_cst, txx)
    L_u_op = L_u_op - ScalarGrad{Linear,2}(one_cst, txy)
    L_v_op = L_v_op - ScalarGrad{Linear,1}(one_cst, txy)
    L_v_op = L_v_op - ScalarGrad{Linear,2}(one_cst, tyy)
end

L_u = (L_u_op == Source(force_vec[1]))
L_v = (L_v_op == Source(force_vec[2]))

if REGIME == :Incompressible
    L_p_op = (- Laplacian{Linear}(tau_rc_cst) + VectorDiv{Linear,1}(one_cst, u) + VectorDiv{Linear,2}(one_cst, v))
    L_p = (L_p_op == Source(0.0))
end

if MODEL == :OldroydB
    L_txx = ((Si(one_cst) - ScalarGrad{Linear,1}(ConstantScalar(2.0*mu_p), u)) == Source(0.0))
    L_tyy = ((Si(one_cst) - ScalarGrad{Linear,2}(ConstantScalar(2.0*mu_p), v)) == Source(0.0))
    L_txy = ((Si(one_cst) - ScalarGrad{Linear,2}(mu_p_cst, u) - ScalarGrad{Linear,1}(mu_p_cst, v)) == Source(0.0))
elseif MODEL == :KelvinVoigt
    mu_eff = mu_p + G*dt
    L_txx = ((Si(one_cst) - ScalarGrad{Linear,1}(ConstantScalar(2.0*mu_eff), u)) == Source(0.0))
    L_tyy = ((Si(one_cst) - ScalarGrad{Linear,2}(ConstantScalar(2.0*mu_eff), v)) == Source(0.0))
    L_txy = ((Si(one_cst) - ScalarGrad{Linear,2}(ConstantScalar(mu_eff), u) - ScalarGrad{Linear,1}(ConstantScalar(mu_eff), v)) == Source(0.0))
end

# ── 8. Solve ──
eqns = [L_u(u), L_v(v)]; phis = [u, v]; bcs_list = (u_bcs, v_bcs)
if REGIME == :Incompressible; push!(eqns, L_p(p)); push!(phis, p); bcs_list = (bcs_list..., p_bcs); end
if MODEL != :Stokes; append!(eqns, [L_txx(txx), L_tyy(tyy), L_txy(txy)]); append!(phis, [txx, tyy, txy]); bcs_list = (bcs_list..., t_bcs, t_bcs, t_bcs); end

sys = MonolithicSystem(eqns, phis)
if REGIME == :Incompressible; setReference!(sys.equations[3], 0.0, 1, config); end

for step in 1:runtime.iterations
    if MODEL == :KelvinVoigt
        idx = REGIME == :Incompressible ? 4 : 3
        @. sys.equations[idx].model.sources[1].field.values = G * gxx.values
        @. sys.equations[idx+1].model.sources[1].field.values = G * gyy.values
        @. sys.equations[idx+2].model.sources[1].field.values = G * gxy.values
    end
    res = solve_monolithic!(sys, bcs_list, config)
    if MODEL == :KelvinVoigt
        ∇u = Grad{Gauss}(u); uf = FaceScalarField(mesh_dev); ∇v = Grad{Gauss}(v); vf = FaceScalarField(mesh_dev)
        grad!(∇u, uf, u, u_bcs, nothing, config); grad!(∇v, vf, v, v_bcs, nothing, config)
        @. gxx.values += 2.0 * ∇u.result.x.values * dt
        @. gyy.values += 2.0 * ∇v.result.y.values * dt
        @. gxy.values += (∇u.result.y.values + ∇v.result.x.values) * dt
    end
    @printf("Step %d, Residual: %.2e, max|u|: %.4f\n", step, res, maximum(abs.(u.values)))
    if res < 1e-10 && step > 1; break; end
end
@info "Benchmark Complete."
