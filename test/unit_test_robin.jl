using XCALibre
using Accessors
using KernelAbstractions
using Test
using SparseArrays

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "laplace_unit_3by3.unv"
mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file)

backend = CPU(); workgroup = 1024
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# Case 1: Robin matching Dirichlet
# a=1, b=0, value=50.0 => Dirichlet(50.0)
BCs_robin = assign(
    region = mesh_dev,
    (
        T = [
            Robin(:left_wall, a=1.0, b=0.0, value=50.0),
            Robin(:right_wall, a=0.0, b=1.0, value=0.0), # ZeroGradient
            Robin(:bottom_wall, a=1.0, b=0.0, value=10.0),
            Robin(:upper_wall, a=0.0, b=1.0, value=0.0)  # ZeroGradient
        ],
    )
)

BCs_dirichlet = assign(
    region = mesh_dev,
    (
        T = [
            Dirichlet(:left_wall, 50.0),
            Zerogradient(:right_wall),
            Dirichlet(:bottom_wall, 10.0),
            Zerogradient(:upper_wall)
        ],
    )
)

solvers = (
    T = SolverSetup(
        solver      = Cg(),
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 1.0,
    )
)

schemes = (
    T = Schemes(laplacian = Linear),
)

config_robin = Configuration(solvers=solvers, schemes=schemes, runtime=Runtime(iterations=1, write_interval=1, time_step=1), hardware=hardware, boundaries=BCs_robin)
config_dirichlet = Configuration(solvers=solvers, schemes=schemes, runtime=Runtime(iterations=1, write_interval=1, time_step=1), hardware=hardware, boundaries=BCs_dirichlet)

T_robin = ScalarField(mesh_dev)
T_dirichlet = ScalarField(mesh_dev)
gamma = ConstantScalar(1.0)

# Build equations
T_eqn_robin = (
    - Laplacian{schemes.T.laplacian}(gamma, T_robin)
    ==
    Source(ConstantScalar(0.0))
) → ScalarEquation(T_robin, config_robin.boundaries.T)

T_eqn_dirichlet = (
    - Laplacian{schemes.T.laplacian}(gamma, T_dirichlet)
    ==
    Source(ConstantScalar(0.0))
) → ScalarEquation(T_dirichlet, config_dirichlet.boundaries.T)

# Discretise and apply BCs
discretise!(T_eqn_robin, T_robin, config_robin)
apply_boundary_conditions!(T_eqn_robin, config_robin; time=0.0)

discretise!(T_eqn_dirichlet, T_dirichlet, config_dirichlet)
apply_boundary_conditions!(T_eqn_dirichlet, config_dirichlet; time=0.0)

# Compare matrices and RHS
@test T_eqn_robin.equation.A.parent ≈ T_eqn_dirichlet.equation.A.parent
@test T_eqn_robin.equation.b ≈ T_eqn_dirichlet.equation.b

# Case 2: Robin with non-zero a and b (Mixed)
# a*T + b*grad(T).n = value
# Let's say a=1, b=1, value=100
# denom = a*delta + b = delta + 1
# delta for 3x3 unit mesh is 1/6 (distance from center (1/6) to face (0))?
# Unit mesh is 1x1. 3x3 cells. Cell width = 1/3.
# Distance from center to face = 1/6.
# denom = 1/6 + 1 = 7/6.
# For -Laplacian, the interior diagonal is positive and the Robin diagonal
# contribution follows the same sign convention as Dirichlet.
# AP = (1 * area * 1) / (7/6) = 6/7 * area.
# BP = (1 * area * 100) / (7/6) = 600/7 * area.

BCs_mixed = assign(
    region = mesh_dev,
    (
        T = [
            Robin(:left_wall, a=1.0, b=1.0, value=100.0),
            Zerogradient(:right_wall),
            Zerogradient(:bottom_wall),
            Zerogradient(:upper_wall)
        ],
    )
)
config_mixed = Configuration(solvers=solvers, schemes=schemes, runtime=Runtime(iterations=1, write_interval=1, time_step=1), hardware=hardware, boundaries=BCs_mixed)
T_eqn_mixed = (
    - Laplacian{schemes.T.laplacian}(gamma, T_robin)
    ==
    Source(ConstantScalar(0.0))
) → ScalarEquation(T_robin, config_mixed.boundaries.T)

discretise!(T_eqn_mixed, T_robin, config_mixed)
apply_boundary_conditions!(T_eqn_mixed, config_mixed; time=0.0)

# Check cell 1 (left bottom)
# Neighbors: 2 (right), 4 (top)
# Boundary faces: 1 (left), 4 (bottom)
# Left face (ID 1) is boundary face on :left_wall
# A[1,1] should have contribution from Robin
area = 1/3 # face area for unit mesh
delta = 1/6
expected_AP = (1.0 * area * 1.0) / (1.0 * delta + 1.0)
# A[1,1] also has contributions from internal faces (right, top) and other boundary faces (bottom)
# Internal face: Gamma*area/delta = 1*(1/3)/(1/3) = 1.
# Bottom face is Zerogradient: contribution 0.
# So A[1,1] = 1 (right) + 1 (top) + expected_AP
expected_A11 = 1.0 + 1.0 + expected_AP
@test T_eqn_mixed.equation.A.parent[1,1] ≈ expected_A11

# Case 3: NonLinearRobin lowering uses an explicit derivative when provided.
C_nl = ScalarField(mesh_dev)
initialise!(C_nl, 2.0)
f_wall(c) = c^2
df_wall(c) = 2c

BCs_nonlinear = assign(
    region = mesh_dev,
    (
        T = [
            NonLinearRobin(:left_wall, f_wall, df_wall),
            Zerogradient(:right_wall),
            Zerogradient(:bottom_wall),
            Zerogradient(:upper_wall)
        ],
    )
)

updated_bcs = update_nonlinear_robin(BCs_nonlinear.T, C_nl)
@test updated_bcs[1] isa Robin
@test updated_bcs[1].value.a ≈ -4.0
@test updated_bcs[1].value.b ≈ 1.0
@test updated_bcs[1].value.value ≈ -4.0

# Direct boundary-module lowering should fail fast without an analytic derivative.
BCs_no_derivative = assign(
    region = mesh_dev,
    (
        T = [
            NonLinearRobin(:left_wall, f_wall),
            Zerogradient(:right_wall),
            Zerogradient(:bottom_wall),
            Zerogradient(:upper_wall)
        ],
    )
)
@test_throws ErrorException update_nonlinear_robin(BCs_no_derivative.T, C_nl)

# The solver-level CPU Newton path can still supply its selected AD backend.
updated_bcs_ad = linearize_bcs(BCs_no_derivative.T, C_nl; ad_backend=:forwarddiff)
@test updated_bcs_ad[1] isa Robin
@test updated_bcs_ad[1].value.a ≈ -4.0
