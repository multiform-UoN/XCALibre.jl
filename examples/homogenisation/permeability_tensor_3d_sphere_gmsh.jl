# ==============================================================================
# 3D Periodic Homogenisation — Cubic Unit Cell with Spherical Inclusion
# ==============================================================================
# This example solves the 3D Stokes cell problems required to compute the
# effective permeability tensor K for a cubic unit cell containing a sphere.
#
# MATHEMATICAL FORMULATION:
# We solve the following steady Stokes system on the fluid domain Y_f:
#   -Δw^(j) + ∇π^(j) = e_j,   ∇·w^(j) = 0
# where e_j is a unit body force in direction j (x, y, or z).
#
# PERIODIC BOUNDARY CONDITIONS:
# For the macroscopic Darcy law to be valid, the microscopic velocity w and
# pressure π must be periodic on the unit cell boundaries.
#   w(x + 2L e_i) = w(x)
#
# GMSH MESH GENERATION:
# - A cubic box of side 2L is created and a sphere of radius R is subtracted.
# - XCALibre's `Gmsh3D_mesh` directly converts the Gmsh API data into Mesh3 format.
#
# PERMEABILITY TENSOR:
# The components of K are found by volume integration:
#   K_ij = (1/|Y|) ∫_{Y_f} w_i^(j) dΩ, where |Y| = (2L)³
# ==============================================================================

using XCALibre
using XCALibre.UNV3
using LinearAlgebra
using KernelAbstractions
using Printf
using Gmsh
using Accessors

# ── Parameters ────────────────────────────────────────────────────────────────
const L         = 1.0    # half-cell side (cell is [-L, L]³, side = 2L)
const R         = 0.5    # sphere radius
const mesh_size = 0.4    # target element size (refine to 0.2 for accuracy)
const ν         = 1.0    # kinematic viscosity

# ── Gmsh mesh generation ─────────────────────────────────────────────────────
"""
    build_3d_unit_cell(L, R, h) → mesh

Generate a tet mesh on [-L,L]³ minus a sphere of radius R.
Uses Gmsh API and converts directly to XCALibre Mesh3 format.
"""
function build_3d_unit_cell(L, R, h)
    gmsh.initialize()
    gmsh.model.add("unit_cell_3d")
    gmsh.option.setNumber("General.Terminal", 0)

    # ── Boolean: box minus sphere ────────────────────────────────────────────
    box_tag    = gmsh.model.occ.addBox(-L, -L, -L, 2L, 2L, 2L)
    sphere_tag = gmsh.model.occ.addSphere(0, 0, 0, R)
    out_tags, _ = gmsh.model.occ.cut([(3, box_tag)], [(3, sphere_tag)])
    gmsh.model.occ.synchronize()

    # ── Classify surfaces by centre-of-mass ──────────────────────────────────
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

    # ── Periodic mesh: slave ← master with translation 2L ───────────────────
    Lc = 2L
    tx = [1.,0,0, Lc, 0,1,0,0, 0,0,1,0, 0,0,0,1]
    ty = [1.,0,0,0, 0,1,0, Lc, 0,0,1,0, 0,0,0,1]
    tz = [1.,0,0,0, 0,1,0,0, 0,0,1, Lc, 0,0,0,1]
    gmsh.model.mesh.setPeriodic(2, right_s,  left_s,   tx)
    gmsh.model.mesh.setPeriodic(2, top_s,    bottom_s, ty)
    gmsh.model.mesh.setPeriodic(2, back_s,   front_s,  tz)

    # ── Named Physical Groups ─────────────────────────────────────────────────
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

    # ── Direct conversion ────────────────────────────────────────────────────
    mesh = Gmsh3D_mesh()
    gmsh.finalize()
    return mesh
end

# ── Stokes cell-problem solver ────────────────────────────────────────────────
function solve_cell_problem_3d(model, config, e_j; pref=0.0)
    (; solvers, schemes, runtime, hardware, boundaries) = config
    (; U, p, Uf, pf) = model.momentum
    mesh    = model.domain
    backend = hardware.backend

    initialise!(U, [0.0, 0.0, 0.0])
    initialise!(p, 0.0)

    ∇p    = Grad{schemes.p.gradient}(p)
    mdotf = FaceScalarField(mesh)
    rDf   = FaceScalarField(mesh)
    initialise!(rDf, 1.0)
    nueff = FaceScalarField(mesh)
    divHv = ScalarField(mesh)

    macro_grad = VectorField(mesh)
    initialise!(macro_grad, e_j)

    # Use the new PDEOperator paradigm
    L_U = ((
          Time{schemes.U.time}()
        + Divergence{schemes.U.divergence}(mdotf)
        - Laplacian{schemes.U.laplacian}(nueff)
        ==
        - Source(∇p.result) + Source(macro_grad)
    ) → boundaries.U) → solvers.U

    L_p = ((
        - Laplacian{schemes.p.laplacian}(rDf) == - Source(divHv)
    ) → boundaries.p) → solvers.p

    U_eqn = L_U(U)
    p_eqn = L_p(p)

    @reset U_eqn.preconditioner = set_preconditioner(solvers.U.preconditioner, U_eqn)
    @reset p_eqn.preconditioner = set_preconditioner(solvers.p.preconditioner, p_eqn)
    @reset U_eqn.solver = XCALibre._workspace(solvers.U.solver, XCALibre._b(U_eqn, XDir()))
    @reset p_eqn.solver = XCALibre._workspace(solvers.p.solver, XCALibre._b(p_eqn))

    turbulenceModel, config = initialise(model.turbulence, model, mdotf, p_eqn, config)

    (; nu) = model.fluid
    n_cells = length(mesh.cells)
    gradU  = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    S = StrainRate(gradU, gradUT, U, Uf)
    Hv   = VectorField(mesh)
    rD   = ScalarField(mesh)
    prev = KernelAbstractions.zeros(backend, _get_float(mesh), n_cells)

    time = 0.0
    interpolate!(Uf, U, config)
    XCALibre.correct_boundaries!(Uf, U, boundaries.U, time, config)
    flux!(mdotf, Uf, config)
    grad!(∇p, pf, p, boundaries.p, time, config)
    update_nueff!(nueff, nu, model.turbulence, config)

    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    for iter in 1:runtime.iterations
        rx, ry, rz = solve_equation!(U_eqn, config)
        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rDf, rD, config)
        remove_pressure_source!(U_eqn, ∇p, config)
        H!(Hv, U, U_eqn, config)
        interpolate!(Uf, Hv, config)
        XCALibre.correct_boundaries!(Uf, Hv, boundaries.U, time, config)
        flux!(mdotf, Uf, config)
        XCALibre.div!(divHv, mdotf, config)
        prev .= p.values
        rp = solve_equation!(p_eqn, config; ref=pref)
        explicit_relaxation!(p, prev, solvers.p.relax, config)
        grad!(∇p, pf, p, boundaries.p, time, config)
        XCALibre.Solvers.correct_mass_flux!(mdotf, p_eqn, config)
        correct_velocity!(U, Hv, ∇p, rD, config)
        update_nueff!(nueff, nu, model.turbulence, config)

        if rx < solvers.U.convergence && ry < solvers.U.convergence &&
           rz < solvers.U.convergence && rp < solvers.p.convergence
            @printf("    converged at iter %4d  (rx=%.2e  ry=%.2e  rz=%.2e  rp=%.2e)\n",
                    iter, rx, ry, rz, rp)
            break
        end
    end

    return volume_integral(U, config) ./ (2L)^3
end

# ── Setup ─────────────────────────────────────────────────────────────────────
@info "Building Gmsh 3D unit cell mesh  (L=$L, R=$R, h=$mesh_size)..."
mesh = build_3d_unit_cell(L, R, mesh_size)
@info "Mesh ready: $(length(mesh.cells)) cells, $(length(mesh.boundaries)) patches"

backend  = CPU(); workgroup = 1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

periodic_x = construct_periodic(mesh, backend, :left,   :right)
periodic_y = construct_periodic(mesh, backend, :bottom, :top)
periodic_z = construct_periodic(mesh, backend, :front,  :back)

model = Physics(
    time       = Steady(),
    fluid      = Fluid{Incompressible}(nu = ν),
    turbulence = RANS{Laminar}(),
    energy     = Energy{Isothermal}(),
    domain     = mesh_dev,
)

BCs = assign(
    region = mesh_dev,
    (
        U = [periodic_x..., periodic_y..., periodic_z..., Wall(:sphere, [0.0, 0.0, 0.0])],
        p = [periodic_x..., periodic_y..., periodic_z..., Wall(:sphere)],
    )
)

schemes = (
    U = Schemes(divergence=Upwind),
    p = Schemes(divergence=Linear),
)

solvers = (
    U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-6, relax=0.7),
    p = SolverSetup(solver=Gmres(),    preconditioner=Jacobi(), convergence=1e-6, relax=0.3),
)

config = Configuration(
    solvers    = solvers,
    schemes    = schemes,
    runtime    = Runtime(iterations=2000, write_interval=-1, time_step=1.0),
    hardware   = hardware,
    boundaries = BCs,
)

# ── Permeability tensor ───────────────────────────────────────────────────────
K = zeros(3, 3)
for j in 1:3
    e_j = zeros(3); e_j[j] = 1.0
    @info "Cell problem  e_$j = $e_j"
    w_j = solve_cell_problem_3d(model, config, e_j)
    K[:, j] .= w_j
end

println("\n─────────────────────────────────────────────")
println("Permeability Tensor K:")
display(K)
@printf("\nK_xx = %.6e\nK_yy = %.6e\nK_zz = %.6e\n", K[1,1], K[2,2], K[3,3])
@printf("Symmetry residual  ‖K - Kᵀ‖/‖K‖ = %.2e\n", norm(K - K') / max(norm(K), eps()))
porosity = total_volume(mesh_dev, config) / (2L)^3
@printf("Fluid porosity φ = %.4f\n", porosity)
println("─────────────────────────────────────────────")
@info "Done."
