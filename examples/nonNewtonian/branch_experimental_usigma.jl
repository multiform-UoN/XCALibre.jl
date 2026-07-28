# =============================================================================
# Experimental Branch: Total Stress Formulation (u, σ)
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

@info "Experimental Branch Benchmark" MODEL BC_TYPE GEOMETRY

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
mu_val     = 1.0
trace_stab = 1e-6
dt         = 0.01

# ── 4. Fields ──
u   = ScalarField(mesh_dev); initialise!(u, 0.0)
v   = ScalarField(mesh_dev); initialise!(v, 0.0)
sxx = ScalarField(mesh_dev); initialise!(sxx, 0.0)
syy = ScalarField(mesh_dev); initialise!(syy, 0.0)
sxy = ScalarField(mesh_dev); initialise!(sxy, 0.0)

# ── 5. BCs ──
if GEOMETRY == :Channel
    u_bcs = [Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0), Zerogradient(:inlet), Zerogradient(:outlet)]
    v_bcs = [Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0), Zerogradient(:inlet), Zerogradient(:outlet)]
    s_bcs = [Zerogradient(:top), Zerogradient(:bottom), Zerogradient(:inlet), Zerogradient(:outlet)]
elseif GEOMETRY == :Bend
    u_bcs = [Dirichlet(:walls, 0.0), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)]
    v_bcs = [Dirichlet(:walls, 0.0), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)]
    s_bcs = [Zerogradient(:walls), Zerogradient(:inlet), Zerogradient(:outlet), Symmetry(:frontAndBack)]
end

# ── 6. Solver Setup ──
solvers = (
    u   = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0, itmax=100),
    v   = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0, itmax=100),
    sxx = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0, itmax=100),
    syy = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0, itmax=100),
    sxy = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0, itmax=100),
)
config = Configuration(solvers=solvers, schemes=(u=Schemes(), v=Schemes(), sxx=Schemes(), syy=Schemes(), sxy=Schemes()),
                       runtime=Runtime(iterations=1, write_interval=-1, time_step=dt), hardware=hardware,
                       boundaries=(u=u_bcs, v=v_bcs, sxx=s_bcs, syy=s_bcs, sxy=s_bcs))

# ── 7. Equations ──
one_cst = ConstantScalar(1.0); mu_cst = ConstantScalar(mu_val); stab_cst = ConstantScalar(trace_stab)

L_u = (ScalarGrad{Linear,1}(one_cst, sxx) + ScalarGrad{Linear,2}(one_cst, sxy) == Source(force_vec[1]))
L_v = (ScalarGrad{Linear,1}(one_cst, sxy) + ScalarGrad{Linear,2}(one_cst, syy) == Source(force_vec[2]))

# dev(σ) = 2μ ε(u)
# sxx - 0.5(sxx+syy) = 2μ ∂u/∂x
# syy - 0.5(sxx+syy) = 2μ ∂v/∂y
# sxy = μ(∂u/∂y + ∂v/∂x)

# Incompressibility
L_trace_op = (VectorDiv{Linear,1}(one_cst, u) + VectorDiv{Linear,2}(one_cst, v))
L_trace_op = L_trace_op + Si(stab_cst, sxx) + Si(stab_cst, syy)
L_trace = (L_trace_op == Source(0.0))

L_sxx_op = (Si(ConstantScalar(0.5)) - Si(ConstantScalar(0.5), syy) - ScalarGrad{Linear,1}(ConstantScalar(2.0*mu_val), u))
L_sxx = (L_sxx_op == Source(0.0))

L_sxy_op = (Si(one_cst) - ScalarGrad{Linear,2}(mu_cst, u) - ScalarGrad{Linear,1}(mu_cst, v))
L_sxy = (L_sxy_op == Source(0.0))

# ── 8. Assemble and Solve ──
eqns = [
    L_u     → ScalarEquation(u, u_bcs),
    L_v     → ScalarEquation(v, v_bcs),
    (L_sxx → s_bcs)(sxx),
    L_trace → ScalarEquation(syy, s_bcs),
    (L_sxy → s_bcs)(sxy),
]
phis = [u, v, sxx, syy, sxy]
bcs_list = (u_bcs, v_bcs, s_bcs, s_bcs, s_bcs)

sys = MonolithicSystem(eqns, phis)

@info "Solving Experimental Branch system..."
res = solve_monolithic!(sys, bcs_list, config)
@info "Residual: $res, max|u|: $(maximum(abs.(u.values)))"
@info "Benchmark Complete."
