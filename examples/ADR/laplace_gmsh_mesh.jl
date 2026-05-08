# =============================================================================
# Laplace equation on a Gmsh-generated mesh
# =============================================================================
#
# Demonstrates creating a mesh with Gmsh.jl directly inside a Julia script —
# no pre-computed mesh file needed.
#
# Problem:  -∇²C = 0  on [0,L]×[0,L]
#   inlet  (x=0): C = C₀  (Dirichlet)
#   outlet (x=L): C = 0   (Dirichlet)
#   top / bottom: ∂C/∂n = 0  (Zerogradient)
#
# Exact solution: C(x) = C₀ (1 - x/L)  (linear)
#
# Install Gmsh.jl once with: ] add Gmsh
# =============================================================================

using XCALibre
using LinearAlgebra
using Printf
import gmsh          # Gmsh.jl — provides gmsh.* API

# ── Parameters ────────────────────────────────────────────────────────────────
L  = 1.0     # domain side length [m]
n  = 30      # cells per side
C0 = 1.0     # inlet concentration

# ── Build mesh with Gmsh ──────────────────────────────────────────────────────
# Create a structured n×n quad mesh on [0,L]×[0,L] entirely in code.
# Boundary Physical Groups are exported as named patches in the UNV file.
# We use Mesh.SaveAll = 1 so quad cells are included in the file even though
# no Physical Group is defined for the 2D domain (avoids confusing the reader).

gmsh.initialize()
gmsh.model.add("square")
gmsh.option.setNumber("General.Terminal", 0)   # suppress Gmsh console output
gmsh.option.setNumber("Mesh.SaveAll", 1)       # write all elements, not only physical groups

# Corner points
gmsh.model.geo.addPoint(0, 0, 0, L/n, 1)   # bottom-left
gmsh.model.geo.addPoint(L, 0, 0, L/n, 2)   # bottom-right
gmsh.model.geo.addPoint(L, L, 0, L/n, 3)   # top-right
gmsh.model.geo.addPoint(0, L, 0, L/n, 4)   # top-left

# Boundary curves
gmsh.model.geo.addLine(1, 2, 1)   # bottom  (y = 0)
gmsh.model.geo.addLine(2, 3, 2)   # outlet  (x = L)
gmsh.model.geo.addLine(3, 4, 3)   # top     (y = L)
gmsh.model.geo.addLine(4, 1, 4)   # inlet   (x = 0)

# Surface
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.geo.addPlaneSurface([1], 1)

# Structured (transfinite) discretisation → uniform quads
gmsh.model.geo.mesh.setTransfiniteCurve(1, n + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(2, n + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(3, n + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(4, n + 1)
gmsh.model.geo.mesh.setTransfiniteSurface(1)    # structured surface mapping

gmsh.model.geo.synchronize()

# Named boundary patches — tag returned by addPhysicalGroup is used by setPhysicalName
# Patch names here become the :symbols used in assign(BCs) below
tI = gmsh.model.addPhysicalGroup(1, [4]); gmsh.model.setPhysicalName(1, tI, "inlet")
tO = gmsh.model.addPhysicalGroup(1, [2]); gmsh.model.setPhysicalName(1, tO, "outlet")
tB = gmsh.model.addPhysicalGroup(1, [1]); gmsh.model.setPhysicalName(1, tB, "bottom")
tT = gmsh.model.addPhysicalGroup(1, [3]); gmsh.model.setPhysicalName(1, tT, "top")

# Generate and recombine to all-quad mesh
gmsh.model.mesh.generate(2)
gmsh.model.mesh.recombine()

# Write to a temporary UNV file and load into XCALibre
mesh_file = tempname() * ".unv"
gmsh.write(mesh_file)
gmsh.finalize()

@info "Gmsh mesh written to $mesh_file"

# ── XCALibre setup ────────────────────────────────────────────────────────────
mesh      = UNV2D_mesh(mesh_file)
backend   = CPU(); workgroup = 1024
hardware  = Hardware(backend=backend, workgroup=workgroup)
mesh_dev  = adapt(backend, mesh)
n_cells   = length(mesh_dev.cells)

C = ScalarField(mesh_dev)
initialise!(C, 0.0)

D = ConstantScalar(1.0)   # diffusion coefficient (any positive value works for Laplace)

solvers = (
    C = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                    convergence=1e-12, relax=1.0),
)
schemes = (C = Schemes(laplacian=Linear),)
runtime = Runtime(iterations=1, write_interval=-1, time_step=1.0)

BCs = assign(
    region = mesh_dev,
    (
        C = [
            Dirichlet(:inlet,  C0),
            Dirichlet(:outlet, 0.0),
            Zerogradient(:bottom),
            Zerogradient(:top),
        ],
    )
)
config = Configuration(solvers=solvers, schemes=schemes,
                       runtime=runtime, hardware=hardware, boundaries=BCs)

# ── Solve -∇²C = 0 via PDEOperator DSL ───────────────────────────────────────
# Define the abstract PDE operator. Note that this doesn't depend on 'C'.
# It encapsulates the math (Laplacian), the boundaries, and the solver settings.
L_C = (
    - Laplacian{Linear}(D)
    == Source(0.0)
) → BCs.C → solvers.C

# Bind the abstract operator to the field 'C' to create a solvable ModelEquation.
C_eqn = L_C(C)

# Setup numerical workspace
@reset C_eqn.preconditioner = set_preconditioner(solvers.C.preconditioner, C_eqn)
@reset C_eqn.solver = XCALibre._workspace(solvers.C.solver, XCALibre._b(C_eqn))

@info "Solving Laplace equation..."
for iter in 1:200
    res = solve_equation!(C_eqn, config)
    res < solvers.C.convergence && break
end

# ── Verification against exact solution C(x) = C₀ (1 - x/L) ─────────────────
xs     = [mesh_dev.cells[i].centre[1] for i in 1:n_cells]
C_anal = [C0 * (1 - x/L)             for x in xs]
err    = maximum(abs.(C.values .- C_anal))

println()
println("Laplace on Gmsh mesh — verification")
@printf("  Domain: [0, %.1f]²,  %d×%d quads\n", L, n, n)
@printf("  Max |C_FVM - C_exact| = %.2e  (C₀ = %.1f)\n", err, C0)
@printf("  Relative L∞ error     = %.2e\n", err / C0)
println()
@info "Done."
