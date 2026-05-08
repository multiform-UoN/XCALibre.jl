# ==============================================================================
# Example: Direct Gmsh.jl Integration (Poisson Problem)
# ==============================================================================
# This script demonstrates a complete, self-contained CAD-to-Solution workflow:
#
# 1. GEOMETRY & MESH: Uses Gmsh.jl to programmatically define a unit square,
#    classify boundary edges into physical groups, and generate a mesh.
#
# 2. XCALIBRE LOADING: Seamlessly loads the Gmsh-generated mesh into XCALibre.
#
# 3. OPERATOR PARADIGM: Uses the new abstract `PDEOperator` DSL to define
#    the Poisson equation (-∇²φ = f) independently of the field data.
#
# 4. SOLUTION: Binds the operator to the field 'phi' and solves the resulting BVP.
# ==============================================================================

using XCALibre
using Gmsh
using Test

# ... (rest of the file)

function generate_mesh_gmsh(filename)
    gmsh.initialize()
    gmsh.model.add("simple_box")
    
    # Define a 1x1 square
    lc = 0.1
    gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
    gmsh.model.geo.addPoint(1, 0, 0, lc, 2)
    gmsh.model.geo.addPoint(1, 1, 0, lc, 3)
    gmsh.model.geo.addPoint(0, 1, 0, lc, 4)
    
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    
    # Define physical groups for boundaries
    gmsh.model.addPhysicalGroup(1, [4], 101, "left")
    gmsh.model.addPhysicalGroup(1, [2], 102, "right")
    gmsh.model.addPhysicalGroup(1, [1], 103, "bottom")
    gmsh.model.addPhysicalGroup(1, [3], 104, "top")
    gmsh.model.addPhysicalGroup(2, [1], 201, "domain")
    
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    
    # Export to UNV (robustly supported by XCALibre)
    gmsh.write(filename)
    gmsh.finalize()
end

# 1. Generate Mesh
msh_file = "gmsh_box.unv"
@info "Generating mesh with Gmsh.jl..."
generate_mesh_gmsh(msh_file)

# 2. Load Mesh into XCALibre
@info "Loading mesh into XCALibre..."
mesh = UNV2D_mesh(msh_file)
backend = CPU()
mesh_dev = adapt(backend, mesh)

# 3. Setup Physics (Simple Poisson Problem)
@info "Setting up Poisson problem using PDEOperator paradigm..."

phi = ScalarField(mesh_dev); initialise!(phi, 0.0)
k = ConstantScalar(1.0)
f = ConstantScalar(1.0)

BCs = assign(
    region = mesh_dev,
    (
        phi = [
            Dirichlet(:left, 0.0),
            Dirichlet(:right, 1.0),
            Zerogradient(:bottom),
            Zerogradient(:top)
        ],
    )
)

solvers = (
    phi = SolverSetup(
        solver = Cg(),
        preconditioner = Jacobi(),
        convergence = 1e-10,
        relax = 1.0
    ),
)

# 4. Define Operator and Solve
# Using the new abstract PDE paradigm
L_phi = (
    - Laplacian{Linear}(k) == Source(f)
) → BCs.phi → solvers.phi

phi_eqn = L_phi(phi)

@info "Solving..."
res = solve_equation!(phi_eqn, Configuration(solvers=solvers, hardware=Hardware(backend=backend)))

@info "Final Residual: $res"
@test res < 1e-10

# Cleanup
rm(msh_file)
@info "Gmsh integration example completed successfully!"
