# =============================================================================
# Incompressible Stokes Benchmark — XCALibre.jl
# =============================================================================
#
# This example mimics the 'stokes3plus3_zeroGradient' OpenFOAM benchmark.
# It solves the standard mixed-form Stokes equations using a monolithic
# block-coupled approach with Rhie-Chow pressure stabilization.
#
# Equations:
#   - Momentum: -∇·(2μ ε(u)) + ∇p = f
#   - Continuity: ∇·u = 0
#
# Geometry: 2D Straight Channel (quad40.unv, 40x40 mesh, 1m x 1m)
# Parameters: mu = 1.0, f = (1.0, 0.0)
# BCs: 
#   - Walls (top/bottom): u = (0,0) [No-slip]
#   - Inlet/Outlet: zeroGradient (Neumann) for u and p.
# =============================================================================

using XCALibre
using Accessors
using LinearAlgebra
using Printf
using Statistics

# ── 1. Mesh and Hardware ──────────────────────────────────────────────────────
# Using the quad40.unv grid which is a 40x40 structured quad mesh.
# Scaling it to 1.0m x 1.0m.
grids_dir = pkgdir(XCALibre, "examples", "0_GRIDS")
mesh_path = joinpath(grids_dir, "quad40.unv")
mesh      = UNV2D_mesh(mesh_path, scale=0.025)

# Hardware selection: CPU for stability and ease of debugging.
backend   = CPU(); workgroup = 1024
hardware  = Hardware(backend=backend, workgroup=workgroup)
mesh_dev  = adapt(backend, mesh)

# ── 2. Material and Forcing Parameters ────────────────────────────────────────
mu_val  = 1.0       # Viscosity (Unitary)
force_x = 1.0       # Unitary x-direction forcing
force_y = 0.0       # Zero y-direction forcing
tau_rc  = 0.1       # Rhie-Chow stabilization parameter

@info "Stokes Benchmark: mu=$mu_val, force=($force_x, $force_y)"

# ── 3. Fields ─────────────────────────────────────────────────────────────────
# Initializing velocity (u, v) and pressure (p) fields to zero.
u = ScalarField(mesh_dev); initialise!(u, 0.0)
v = ScalarField(mesh_dev); initialise!(v, 0.0)
p = ScalarField(mesh_dev); initialise!(p, 0.0)

# ── 4. Boundary Conditions ───────────────────────────────────────────────────
# Mimicking the zeroGradient setup from stokes3plus3_zeroGradient OpenFOAM case.
# top/bottom are walls, inlet/outlet are patches.
BCs = assign(
    region = mesh_dev,
    (
        u = [Dirichlet(:top,    0.0), # No-slip
             Dirichlet(:bottom, 0.0),
             Zerogradient(:inlet),
             Zerogradient(:outlet)],
        v = [Dirichlet(:top,    0.0),
             Dirichlet(:bottom, 0.0),
             Zerogradient(:inlet),
             Zerogradient(:outlet)],
        p = [Zerogradient(:top),
             Zerogradient(:bottom),
             Zerogradient(:inlet),
             Zerogradient(:outlet)],
    )
)

# ── 5. Solver and Numerical Schemes ───────────────────────────────────────────
# Using Gmres for the monolithic linear system with Jacobi preconditioning.
solvers = (
    u = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    v = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    p = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
)
schemes = (u = Schemes(laplacian=Linear),
           v = Schemes(laplacian=Linear),
           p = Schemes(laplacian=Linear))
runtime = Runtime(iterations=1, write_interval=-1, time_step=1.0)

config = Configuration(solvers=solvers, schemes=schemes,
                       runtime=runtime, hardware=hardware, boundaries=BCs)

# ── 6. Monolithic Equation Definitions ────────────────────────────────────────
# Using the PDEOperator DSL to define the coupled blocks.
mu_cst     = ConstantScalar(mu_val)
one_cst    = ConstantScalar(1.0)
tau_rc_cst = ConstantScalar(tau_rc)

# Momentum equations (u and v)
# -μ∇²u + ∂p/∂x = f_x
L_u = ((
    - Laplacian{Linear}(mu_cst)
    + ScalarGrad{Linear,1}(one_cst, p)
    == Source(force_x)
) → BCs.u) → solvers.u

L_v = ((
    - Laplacian{Linear}(mu_cst)
    + ScalarGrad{Linear,2}(one_cst, p)
    == Source(force_y)
) → BCs.v) → solvers.v

# Continuity equation with Rhie-Chow stabilization
# ∇·u - τ_rc ∇²p = 0
L_p = ((
    - Laplacian{Linear}(tau_rc_cst) # Stabilization term
    + VectorDiv{Linear,1}(one_cst, u)
    + VectorDiv{Linear,2}(one_cst, v)
    == Source(0.0)
) → BCs.p) → solvers.p

# Combining into a monolithic system
u_eqn = L_u(u); v_eqn = L_v(v); p_eqn = L_p(p)
sys = MonolithicSystem([u_eqn, v_eqn, p_eqn], [u, v, p])

# ── 7. Solve ──────────────────────────────────────────────────────────────────
@info "Solving Incompressible Stokes system..."
# Setting a reference pressure since all boundaries are zeroGradient
setReference!(p_eqn, 0.0, 1, config)
res = solve_monolithic!(sys, (BCs.u, BCs.v, BCs.p), config)
@info "Solve complete. Residual: $res"

# ── 8. Post-process and Results ───────────────────────────────────────────────
max_u = maximum(abs.(u.values))
@info "Peak Velocity max|u| = $max_u"

# Exporting results to VTK for verification
save_output(u, "stokes_incomp_u", 0.0, config)
save_output(v, "stokes_incomp_v", 0.0, config)
save_output(p, "stokes_incomp_p", 0.0, config)
@info "Benchmark stokes_incomp finished."
