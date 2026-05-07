# =============================================================================
# 2-D Linear Elastic Solver — uniaxial bar stretching (monolithic, 2-field)
# =============================================================================
#
# Governing equation:  ∇·σ = 0,  σ = λ(∇·U)I + μ(∇U + ∇Uᵀ)
#
# For constant Lamé parameters the PDE expands to:
#
#   μ ∇²U + (μ+λ) ∇(∇·U) = 0
#
# Discretisation strategy (monolithic, fully implicit, 2-field):
#
#   [Laplacian(μ,u) + GradDiv{1,1}(α,u)   GradDiv{1,2}(α,v)       ] {u}   {0}
#   [GradDiv{2,1}(α,u)   Laplacian(μ,v) + GradDiv{2,2}(α,v)       ] {v} = {0}
#
# where α = μ+λ.  GradDiv{I,J} contributes the face coefficient:
#   α_f · face.e[J] · face.area · face.normal[I] / face.delta
#
# This assembles the full Cauchy-stress stiffness block-coupled:
#   - Laplacian: ∇·(μ ∇U_i) in each diagonal block (non-orthogonal corrected)
#   - GradDiv:   (μ+λ) ∇(∇·U) across all blocks, including off-diagonal coupling
#
# Problem: 6×6 mesh (laplace_2d_mesh.unv, boundaries at x=±3, y=0/6)
#   left_wall  (x=-3): u=0, v=0   (fixed support)
#   right_wall (x=+3): u=δ, v=0   (prescribed stretch)
#   upper_wall/bottom_wall: Zerogradient u, Dirichlet v=0  (rollers)
#
# Analytical solution: u(x)=δ·(x+3)/6, v=0
#   P-wave mod. M = 2μ+λ  →  σ_xx = M·δ/L
# =============================================================================

using XCALibre
using Statistics
using Printf

# ── Material parameters ───────────────────────────────────────────────────────
E       = 1.0
nu      = 0.3
mu_val  = E / (2*(1 + nu))
lam_val = E*nu / ((1 + nu)*(1 - 2*nu))   # plane-strain Lamé
alpha_val = mu_val + lam_val              # coefficient for GradDiv (μ+λ)
delta   = 0.01                            # prescribed x-displacement at right wall

# ── Mesh ─────────────────────────────────────────────────────────────────────
grids_dir = pkgdir(XCALibre, "examples", "0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "laplace_2d_mesh.unv"), scale=1.0)

backend  = CPU(); workgroup = 1024
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# ── Fields ────────────────────────────────────────────────────────────────────
u = ScalarField(mesh_dev); initialise!(u, 0.0)
v = ScalarField(mesh_dev); initialise!(v, 0.0)

# ── Boundary conditions ───────────────────────────────────────────────────────
# Note: use Zerogradient (not Extrapolated) for u at top/bottom so that the
# zero-gradient condition is imposed directly in a single solve pass.
BCs = assign(
    (
        u = [Dirichlet(:left_wall,  0.0),
             Dirichlet(:right_wall, delta),
             Zerogradient(:upper_wall),
             Zerogradient(:bottom_wall)],
        v = [Dirichlet(:left_wall,  0.0),
             Dirichlet(:right_wall, 0.0),
             Dirichlet(:upper_wall, 0.0),
             Dirichlet(:bottom_wall, 0.0)],
    ),
    region = mesh_dev
)

# ── Solver setup ──────────────────────────────────────────────────────────────
schemes = (u = Schemes(laplacian=Linear), v = Schemes(laplacian=Linear))
runtime = Runtime(iterations=1, write_interval=1, time_step=1)
solvers = (u = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                           convergence=1e-10, relax=1.0),
           v = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                           convergence=1e-10, relax=1.0))
config = Configuration(schemes=schemes, solvers=solvers, runtime=runtime,
                       hardware=hardware, boundaries=BCs)

# ── Constant flux fields ──────────────────────────────────────────────────────
mu_flux    = ConstantScalar(mu_val)
alpha_flux = ConstantScalar(alpha_val)

# ── Equation system ───────────────────────────────────────────────────────────
u_eqn = (
    - Laplacian{Linear}(mu_flux,    u)
    - GradDiv{Linear,1,1}(alpha_flux, u)
    - GradDiv{Linear,1,2}(alpha_flux, v)
    == Source(ConstantScalar(0.0))
) → ScalarEquation(u, BCs.u)

v_eqn = (
    - Laplacian{Linear}(mu_flux,    v)
    - GradDiv{Linear,2,1}(alpha_flux, u)
    - GradDiv{Linear,2,2}(alpha_flux, v)
    == Source(ConstantScalar(0.0))
) → ScalarEquation(v, BCs.v)

# ── Monolithic solve ──────────────────────────────────────────────────────────
@info "Building monolithic 2-field linear elastic system..."
sys = MonolithicSystem([u_eqn, v_eqn], [u, v])

@info "Solving..."
res = solve_monolithic!(sys, (BCs.u, BCs.v), config)

# ── Verification ──────────────────────────────────────────────────────────────
xs    = [c.centre[1] for c in mesh_dev.cells]
x_L   = -3.0; x_R = 3.0; L = x_R - x_L       # actual boundary positions
u_analytical = delta .* (xs .- x_L) ./ L
u_err = maximum(abs.(u.values .- u_analytical))

M_wave  = 2*mu_val + lam_val
u_mean  = mean(u.values)
v_mean  = mean(v.values)

println()
println("Linear Elastic Results — 2-D uniaxial stretch (monolithic, fully implicit)")
println("  E=$E, ν=$nu  →  μ=$(round(mu_val,digits=4)), λ=$(round(lam_val,digits=4))")
println("  α = μ+λ = $(round(alpha_val,digits=4))   M = 2μ+λ = $(round(M_wave,digits=4))")
println("  Applied δ=$delta at right_wall, domain L=$L")
println()
@printf("  BiCGSTAB residual     : %.2e\n", res)
@printf("  Max |u - u_analytical|: %.2e\n", u_err)
@printf("  Mean u                : %.6f  (analytical %.6f)\n", u_mean, mean(u_analytical))
@printf("  Mean v                : %.2e  (analytical 0)\n", v_mean)
println()
@info "Linear elastic 2-D (monolithic) example completed!"
