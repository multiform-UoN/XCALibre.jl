using XCALibre
using LinearAlgebra
using Accessors
using Printf
using Statistics

# ==============================================================================
# TUTORIAL: Pore-Scale Homogenisation in XCALibre.jl
# ==============================================================================
# This tutorial demonstrates how to calculate the permeability tensor of a
# porous medium (a single sphere in a unit cell) using XCALibre.
#
# Methodology:
# 1. Load an OpenFOAM mesh (generated via blockMesh + snappyHexMesh).
# 2. Solve the Stokes Cell Problem in each principal direction (X, Y, Z).
# 3. Apply a unit macroscopic pressure gradient as a source term.
# 4. Calculate the volume-averaged velocity to obtain the Permeability Tensor K.
#
# Comparison with OpenFOAM:
# This script uses the exact same mesh and boundary conditions as the
# OpenFOAM 'singleSphere' tutorial, allowing for direct performance comparison.
# ==============================================================================

# 1. LOAD MESH
# ==============================================================================
# We generate the domain geometry using Gmsh.jl.
# A cubic unit cell with a spherical inclusion is created and meshed.

using Gmsh

function build_3d_unit_cell(L, R, h)
    gmsh.initialize()
    gmsh.model.add("unit_cell_3d")
    gmsh.option.setNumber("General.Terminal", 0)

    # Boolean: box minus sphere
    box_tag    = gmsh.model.occ.addBox(-L, -L, -L, 2L, 2L, 2L)
    sphere_tag = gmsh.model.occ.addSphere(0, 0, 0, R)
    out_tags, _ = gmsh.model.occ.cut([(3, box_tag)], [(3, sphere_tag)])
    gmsh.model.occ.synchronize()

    # Classify surfaces by centre-of-mass
    left_s  = Int[]; right_s  = Int[]
    bottom_s = Int[]; top_s   = Int[]
    front_s = Int[]; back_s   = Int[]
    sphere_s = Int[]

    for surf in gmsh.model.getEntities(2)
        tag = surf[2]
        com = gmsh.model.occ.getCenterOfMass(2, tag)
        if     isapprox(com[1], -L; atol=1e-3); push!(left_s,   tag)
        elseif isapprox(com[1],  L; atol=1e-3); push!(right_s,  tag)
        elseif isapprox(com[2], -L; atol=1e-3); push!(bottom_s, tag)
        elseif isapprox(com[2],  L; atol=1e-3); push!(top_s,    tag)
        elseif isapprox(com[3], -L; atol=1e-3); push!(front_s,  tag)
        elseif isapprox(com[3],  L; atol=1e-3); push!(back_s,   tag)
        else                                     push!(sphere_s, tag)
        end
    end

    # Periodic mesh: slave ← master with translation 2L
    Lc = 2L
    tx = [1.,0,0, Lc, 0,1,0,0, 0,0,1,0, 0,0,0,1]
    ty = [1.,0,0,0, 0,1,0, Lc, 0,0,1,0, 0,0,0,1]
    tz = [1.,0,0,0, 0,1,0,0, 0,0,1, Lc, 0,0,0,1]
    gmsh.model.mesh.setPeriodic(2, right_s,  left_s,   tx)
    gmsh.model.mesh.setPeriodic(2, top_s,    bottom_s, ty)
    gmsh.model.mesh.setPeriodic(2, back_s,   front_s,  tz)

    # Named Physical Groups
    tL  = gmsh.model.addPhysicalGroup(2, left_s);   gmsh.model.setPhysicalName(2, tL,  "left")
    tR  = gmsh.model.addPhysicalGroup(2, right_s);  gmsh.model.setPhysicalName(2, tR,  "right")
    tBo = gmsh.model.addPhysicalGroup(2, bottom_s); gmsh.model.setPhysicalName(2, tBo, "bottom")
    tTo = gmsh.model.addPhysicalGroup(2, top_s);    gmsh.model.setPhysicalName(2, tTo, "top")
    tFr = gmsh.model.addPhysicalGroup(2, front_s);  gmsh.model.setPhysicalName(2, tFr, "front")
    tBa = gmsh.model.addPhysicalGroup(2, back_s);   gmsh.model.setPhysicalName(2, tBa, "back")
    tSp = gmsh.model.addPhysicalGroup(2, sphere_s); gmsh.model.setPhysicalName(2, tSp, "sphere")
    tV  = gmsh.model.addPhysicalGroup(3, [out_tags[1][2]]); gmsh.model.setPhysicalName(3, tV, "fluid")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.model.mesh.generate(3)

    mesh = Gmsh3D_mesh()
    gmsh.finalize()
    return mesh
end

@info "Generating Pore-Scale Mesh (Single Sphere) with Gmsh..."
const L         = 1.0
const R         = 0.5
const mesh_size = 0.4
mesh = build_3d_unit_cell(L, R, mesh_size)

# Hardware selection (CPU with multithreading)
backend = CPU(); workgroup=1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# 2. Define Physics & Boundary Conditions
# ------------------------------------------------------------------------------
# We apply periodic BCs on outer walls and Wall BC on the sphere.
periodic_x = construct_periodic(mesh, backend, :left,   :right)
periodic_y = construct_periodic(mesh, backend, :bottom, :top)
periodic_z = construct_periodic(mesh, backend, :front,  :back)

BCs = assign(
    region=mesh_dev,
    (
        U = [
            periodic_x..., periodic_y..., periodic_z...,
            Wall(:sphere, [0.0, 0.0, 0.0])
        ],
        p = [
            periodic_x..., periodic_y..., periodic_z...,
            Wall(:sphere)
        ]
    )
)

# Schemes: Simple Upwind for divergence, Linear for Laplacian (Standard Stokes)
schemes = (U = Schemes(divergence=Upwind, time=SteadyState), p = Schemes(divergence=Linear, time=SteadyState))

# Solvers: Iterative solvers with Jacobi preconditioning
solvers = (
    U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-6, relax=0.7),
    p = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-6, relax=0.3, itmax=500, rtol=1e-3)
)

config = Configuration(solvers=solvers, schemes=schemes, runtime=Runtime(iterations=500, write_interval=-1, time_step=1.0), hardware=hardware, boundaries=BCs)

# 3. Solver Implementation (Stokes Cell Problem)
# ------------------------------------------------------------------------------
# This function solves for a single direction e_j
function solve_stokes_direction(direction_vector, mesh_dev, BCs, solvers, schemes, config)
    U = VectorField(mesh_dev); initialise!(U, [0.0, 0.0, 0.0])
    p = ScalarField(mesh_dev); initialise!(p, 0.0)
    Uf = FaceVectorField(mesh_dev)
    pf = FaceScalarField(mesh_dev)

    ∇p = Grad{schemes.p.gradient}(p)
    mdotf = FaceScalarField(mesh_dev)
    rDf = FaceScalarField(mesh_dev); initialise!(rDf, 1.0)
    nueff = FaceScalarField(mesh_dev); initialise!(nueff, 1.0) # nu = 1.0
    divHv = ScalarField(mesh_dev)

    # Apply macroscopic gradient as an implicit source in momentum
    macro_grad = VectorField(mesh_dev); initialise!(macro_grad, direction_vector)

    L_U = ((
          Time{schemes.U.time}()
        - Laplacian{schemes.U.laplacian}(nueff)
        ==
        - Source(∇p.result) + Source(macro_grad)
    ) → BCs.U) → solvers.U

    L_p = ((
        - Laplacian{schemes.p.laplacian}(rDf) == - Source(divHv)
    ) → BCs.p) → solvers.p

    U_eqn = L_U(U)
    p_eqn = L_p(p)

    @reset U_eqn.preconditioner = set_preconditioner(solvers.U.preconditioner, U_eqn)
    @reset p_eqn.preconditioner = set_preconditioner(solvers.p.preconditioner, p_eqn)
    @reset U_eqn.solver = XCALibre._workspace(solvers.U.solver, XCALibre._b(U_eqn, XDir()))
    @reset p_eqn.solver = XCALibre._workspace(solvers.p.solver, XCALibre._b(p_eqn))

    Hv = VectorField(mesh_dev)
    rD = ScalarField(mesh_dev)
    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    for iter in 1:config.runtime.iterations
        rx, ry, rz = solve_equation!(U_eqn, config)

        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rDf, rD, config)
        remove_pressure_source!(U_eqn, ∇p, config)
        H!(Hv, U, U_eqn, config)
        interpolate!(Uf, Hv, config)
        flux!(mdotf, Uf, config)
        XCALibre.div!(divHv, mdotf, config)

        rp = solve_equation!(p_eqn, config; ref=0.0)
        grad!(∇p, pf, p, BCs.p, 0.0, config)
        XCALibre.Solvers.correct_mass_flux!(mdotf, p_eqn, config)
        correct_velocity!(U, Hv, ∇p, rD, config)

        if rx < solvers.U.convergence && rp < solvers.p.convergence
            @printf("  Converged at iteration %d (Res U: %.2e, p: %.2e)\n", iter, rx, rp)
            break
        end
    end

    # Volume average velocity for permeability
    return volume_average(U)
end

# 4. Main Execution (Tensor Calculation)
# ------------------------------------------------------------------------------
dim = 3
K = zeros(dim, dim)
@info "Calculating Permeability Tensor K..."

for j in 1:dim
    e_j = zeros(3); e_j[j] = 1.0
    @info "Solving for direction $j ($e_j)..."
    v_avg = solve_stokes_direction(e_j, mesh_dev, BCs, solvers, schemes, config)
    for i in 1:dim
        K[i, j] = v_avg[i]
    end
end

# 5. Output Results
# ------------------------------------------------------------------------------
println("\n" * "="^40)
println("PERMEABILITY TENSOR K:")
display(K)
println("\nSymmetry Residual: ", norm(K - K') / norm(K))
println("="^40)

@info "Tutorial Completed Successfully!"
