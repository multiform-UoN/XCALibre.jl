# =============================================================================
# TUTORIAL: Gmsh Integration — Direct API vs File Import
# =============================================================================
#
# This tutorial demonstrates the two ways to bring complex geometries into 
# XCALibre.jl using Gmsh.
#
#   Mode A: Direct API Integration (Bypasses intermediate files)
#   Mode B: Import from existing mesh (e.g. UNV format)
#
# =============================================================================

using XCALibre
using LinearAlgebra
using Accessors
using Printf
using Gmsh

# ── PROBLEM: Poisson Equation ───────────────────────────────────────────────
# -∇²phi = 1  on a disk
# phi = 0 on boundary
# Analytical: phi = (R² - r²)/4
# ─────────────────────────────────────────────────────────────────────────────

radius = 1.0
lc = 0.1

function run_sim(mesh, mode_name)
    backend = CPU(); workgroup = 1024
    mesh_dev = adapt(backend, mesh)
    
    phi = ScalarField(mesh_dev); initialise!(phi, 0.0)
    
    # Setup BCs
    BCs = assign(region=mesh_dev, (
        phi = [Dirichlet(b.name, 0.0) for b in mesh.boundaries],
    ))
    
    # Define PDE
    L_phi = ( - Laplacian{Linear}(ConstantScalar(1.0)) == Source(1.0) ) → BCs.phi
    
    setup = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0)
    config = Configuration(hardware=Hardware(backend=backend, workgroup=workgroup),
                           runtime=Runtime(iterations=1, time_step=1.0, write_interval=-1),
                           schemes=(phi=Schemes(laplacian=Linear),),
                           solvers=(phi=nothing,), boundaries=BCs)
    
    eqn = (L_phi → setup)(phi)
    
    # Explicitly initialize workspace
    @reset eqn.preconditioner = set_preconditioner(setup.preconditioner, eqn)
    @reset eqn.solver = XCALibre._workspace(setup.solver, _b(eqn))
    
    @info "Solving Poisson on $mode_name mesh..."
    for i in 1:200
        res = solve_equation!(eqn, config)
        res < setup.convergence && break
    end
    
    # Verify center value
    c_idx = argmin([norm(c.centre) for c in mesh_dev.cells])
    phi_center = phi.values[c_idx]
    analytical = radius^2 / 4.0
    @printf("  Mode %s: Center phi = %.4f (Analytical %.4f)\n", mode_name, phi_center, analytical)
end

# ── MODE A: Direct Gmsh Integration ──────────────────────────────────────────

gmsh.initialize()
gmsh.model.add("disk")
gmsh.option.setNumber("General.Terminal", 0)

# Create disk geometry
gmsh.model.occ.addDisk(0, 0, 0, radius, radius)
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(2, [1], 101); gmsh.model.setPhysicalName(2, 101, "domain")
gmsh.model.addPhysicalGroup(1, [1], 201); gmsh.model.setPhysicalName(1, 201, "boundary")

gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
gmsh.model.mesh.generate(2)

mesh_direct = Gmsh2D_mesh() # Direct conversion!
gmsh.finalize()

run_sim(mesh_direct, "DIRECT API")

# ── MODE B: Import from Existing Mesh (UNV) ───────────────────────────────────

# We use a pre-computed mesh from the examples folder
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_file = joinpath(grids_dir, "laplace_unit_3by3.unv")

if isfile(mesh_file)
    @info "Loading pre-computed UNV mesh: $mesh_file"
    mesh_file_import = UNV2D_mesh(mesh_file)
    run_sim(mesh_file_import, "FILE IMPORT") 
else
    @warn "Pre-computed mesh not found, skipping Mode B."
end

@info "Gmsh Integration Tutorial Completed!"
