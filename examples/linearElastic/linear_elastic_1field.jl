# =============================================================================
# 1-Field Linear Elastic Solver — 1-D bar stretching (single ScalarEquation)
# =============================================================================
#
# Problem: axis-aligned bar, x-displacement u only, v forced to zero.
#
# Governing PDE for u (v=0 substituted):
#
#   μ ∇²u + (μ+λ) ∂²u/∂x² = (2μ+λ) ∂²u/∂x² = 0
#
# Discretised as a single scalar equation:
#
#   Laplacian(μ, u) + GradDiv{Linear,1,1}(α, u) = 0
#
# where α = μ+λ.  The diagonal block (1,1) gives face coefficient:
#   α_f * e_x * Sf_x / delta
# which on an aligned mesh is simply α_f * area / delta,
# adding the P-wave stiffening:  μ + α = μ + (μ+λ) = 2μ+λ = M_wave
#
# Mesh: quad40.unv  (1000×1000 domain, 40×40 cells, x∈[0,1000])
#   inlet  (x=0):  u = 0       (fixed)
#   outlet (x=L):  u = delta   (prescribed stretch)
#   bottom/top:    Zerogradient (traction-free rollers — no u gradient normal)
#
# Analytical solution: u(x) = delta * x / L   (linear profile)
# =============================================================================

using XCALibre
using Accessors
using Statistics
using Printf

# ── Material parameters ───────────────────────────────────────────────────────
E       = 1.0
nu      = 0.3
mu_val  = E / (2*(1 + nu))
lam_val = E*nu / ((1 + nu)*(1 - 2*nu))   # plane-strain Lamé
alpha_val = mu_val + lam_val              # GradDiv coefficient  (α = μ+λ)
M_wave  = 2*mu_val + lam_val             # P-wave modulus (total axial stiffness)
delta   = 0.01                           # prescribed x-displacement at outlet

# ── Mesh ─────────────────────────────────────────────────────────────────────
grids_dir = pkgdir(XCALibre, "examples", "0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=1.0)

backend  = CPU(); workgroup = 1024
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# ── Field ─────────────────────────────────────────────────────────────────────
u = ScalarField(mesh_dev); initialise!(u, 0.0)

# ── Boundary conditions ───────────────────────────────────────────────────────
BCs = assign(
    (
        u = [Dirichlet(:inlet,  0.0),
             Dirichlet(:outlet, delta),
             Zerogradient(:bottom),
             Zerogradient(:top)],
    ),
    region = mesh_dev
)

# ── Solver setup ──────────────────────────────────────────────────────────────
schemes = (u = Schemes(laplacian=Linear),)
runtime = Runtime(iterations=1, write_interval=1, time_step=1)
solvers = (u = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                           convergence=1e-12, relax=1.0),)
config = Configuration(schemes=schemes, solvers=solvers, runtime=runtime,
                       hardware=hardware, boundaries=BCs)

# ── Constant flux fields ──────────────────────────────────────────────────────
mu_flux    = ConstantScalar(mu_val)
alpha_flux = ConstantScalar(alpha_val)

# ── Equation ──────────────────────────────────────────────────────────────────
# Laplacian contributes μ * area/delta per face.
# GradDiv{1,1} contributes α * e_x * Sf_x / delta = α * area/delta on x-aligned faces.
# Total axial stiffness = μ + α = 2μ+λ = M_wave  ✓
u_eqn = (
    - Laplacian{Linear}(mu_flux,    u)
    - GradDiv{Linear,1,1}(alpha_flux, u)
    == Source(ConstantScalar(0.0))
) → ScalarEquation(u, BCs.u)

# ── Solve ─────────────────────────────────────────────────────────────────────
@info "Solving 1-field linear elastic equation..."
@reset u_eqn.preconditioner = set_preconditioner(solvers.u.preconditioner, u_eqn)
@reset u_eqn.solver = XCALibre._workspace(solvers.u.solver, XCALibre._b(u_eqn))

# Iterate until convergence (the linear system is solved exactly when res < convergence)
res = let r = Inf
    for i in 1:50
        r = solve_equation!(u_eqn, u, BCs.u, solvers.u, config)
        r < solvers.u.convergence && break
    end
    r
end

# ── Verification ──────────────────────────────────────────────────────────────
xs = [c.centre[1] for c in mesh_dev.cells]
x_L = 0.0; x_R = 1000.0; L = x_R - x_L
u_analytical = delta .* (xs .- x_L) ./ L
u_err  = maximum(abs.(u.values .- u_analytical))
u_mean = mean(u.values)

println()
println("Linear Elastic Results — 1-field bar stretch")
println("  E=$E, ν=$nu  →  μ=$(round(mu_val,digits=4)), λ=$(round(lam_val,digits=4))")
println("  M = 2μ+λ = $(round(M_wave,digits=4))   (Laplacian μ + GradDiv{1,1} α)")
println("  Applied δ=$delta at outlet, domain L=$L")
println()
@printf("  Solver residual           : %.2e\n", res)
@printf("  Max |u - u_analytical|    : %.2e\n", u_err)
@printf("  Mean u                    : %.6f  (analytical %.6f)\n", u_mean, mean(u_analytical))
println()
@info "1-field linear elastic example completed!"
