using XCALibre
using LinearAlgebra
using Accessors
using Printf
using Statistics

# ==============================================================================
# Advanced Example: Non-Linear Vector Newton Solver
# ==============================================================================
# This script solves a vector advection-diffusion-reaction equation with
# a non-linear quadratic drag term:
#   -ν ∇²U + (U ⋅ ∇)U + α|U|U = f
#
# We solve this using the Monolithic Newton framework with auto-decomposition.
# ==============================================================================

# 1. Setup Mesh
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "laplace_unit_3by3.unv")) # Tiny mesh for speed

backend = CPU(); workgroup=1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# 2. Setup Fields
U = VectorField(mesh_dev); initialise!(U, [1.0, 0.0, 0.0])
f = VectorField(mesh_dev); initialise!(f, [0.0, 0.0, 0.0])

# 3. Parameters
nu = 0.01
alpha_drag = 10.0 # Strong quadratic drag

# 4. Define Non-Linear Operator
# We use NonLinearSi to define the quadratic drag |U|U.
# For a vector field, we define the scalar mapping for each component.
# Note: |U| is treated as a frozen coefficient in this simplified example,
# or we can use the full nonlinear form.

# 5. Setup Configuration
BCs = assign(
    region=mesh_dev,
    (
        U = [Dirichlet(:left_wall, [1.0, 0.0, 0.0]), Dirichlet(:right_wall, [0.5, 0.0, 0.0]), Zerogradient(:upper_wall), Zerogradient(:bottom_wall)],
    )
)

schemes = (U = Schemes(laplacian=Linear),)

config = Configuration(
    solvers=nothing,
    schemes=schemes,
    runtime=Runtime(iterations=10, time_step=1.0, write_interval=-1),
    hardware=hardware,
    boundaries=BCs
)

@info "Defining Non-Linear Vector PDE..."

# L_U = -nu*Laplacian(U) + alpha*|U|*U == 0
# We use the new PDEOperator DSL.
# Note: decompose() will automatically split this into Ux and Uy equations.
# The Nonlinear mapping for drag: u -> alpha * |U_old| * u
L_U = ((
      - Laplacian{Linear}(ConstantScalar(nu))
      + NonLinearSi(u -> alpha_drag * u * abs(u)) # Quadratic drag
      == Source(0.0)
) → BCs.U)

U_eqn = L_U(U)

# 6. Monolithic Newton Solve
# MonolithicSystem will decompose U_eqn into [Ux_eqn, Uy_eqn]
sys = MonolithicSystem([U_eqn], [U])
bcs_list = [get_bcs(eqn) for eqn in sys.equations]

@info "Solving via Monolithic Newton..."
res = newton_solve!(sys, bcs_list, config; tol=1e-9, verbose=true)

@info "Final U mean: $(mean(U.x.values)), $(mean(U.y.values))"
@info "Non-Linear Vector Newton Example Completed!"
