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
using XCALibre.UNV2
using LinearAlgebra
using Accessors
using Printf
using Gmsh

# ── Parameters ────────────────────────────────────────────────────────────────
L  = 1.0     # domain side length [m]
n  = 30      # cells per side
C0 = 1.0     # inlet concentration

# ── Build mesh with Gmsh ──────────────────────────────────────────────────────
gmsh.initialize()
gmsh.model.add("square")
gmsh.option.setNumber("General.Terminal", 0)

# Corner points
gmsh.model.geo.addPoint(0, 0, 0, L/n, 1)
gmsh.model.geo.addPoint(L, 0, 0, L/n, 2)
gmsh.model.geo.addPoint(L, L, 0, L/n, 3)
gmsh.model.geo.addPoint(0, L, 0, L/n, 4)

# Boundary curves
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)

# Surface
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.geo.addPlaneSurface([1], 1)

# Structured discretisation
gmsh.model.geo.mesh.setTransfiniteCurve(1, n + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(2, n + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(3, n + 1)
gmsh.model.geo.mesh.setTransfiniteCurve(4, n + 1)
gmsh.model.geo.mesh.setTransfiniteSurface(1)

gmsh.model.geo.synchronize()

# Named boundary patches
tI = gmsh.model.addPhysicalGroup(1, [4]); gmsh.model.setPhysicalName(1, tI, "inlet")
tO = gmsh.model.addPhysicalGroup(1, [2]); gmsh.model.setPhysicalName(1, tO, "outlet")
tB = gmsh.model.addPhysicalGroup(1, [1]); gmsh.model.setPhysicalName(1, tB, "bottom")
tT = gmsh.model.addPhysicalGroup(1, [3]); gmsh.model.setPhysicalName(1, tT, "top")

gmsh.model.mesh.generate(2)
gmsh.model.mesh.recombine()

# Direct conversion to XCALibre Mesh2
mesh = Gmsh2D_mesh()
gmsh.finalize()

# ── XCALibre setup ────────────────────────────────────────────────────────────
backend   = CPU(); workgroup = 1024
hardware  = Hardware(backend=backend, workgroup=workgroup)
mesh_dev  = adapt(backend, mesh)
n_cells   = length(mesh_dev.cells)

C = ScalarField(mesh_dev); initialise!(C, 0.0)
D = ConstantScalar(1.0)

solvers = (
    C = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0),
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
L_C = ( (
    - Laplacian{Linear}(D)
    == Source(0.0)
) → BCs.C ) → solvers.C

C_eqn = L_C(C)
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
