# ==============================================================================
# 2D Periodic Homogenisation — Square Unit Cell with Circular Inclusion
# ==============================================================================
# This example solves the Stokes cell problems required to compute the
# effective permeability tensor K of a periodic microstructure.
#
# MATHEMATICAL BACKGROUND:
# For a periodic porous medium, the macro-scale Darcy law is derived by
# solving "cell problems" on a representative unit cell Y.
#
# Solves Stokes cell problems (μ = 1) for each coordinate direction e_j:
#   -Δw^(j) + ∇π^(j) = e_j,   ∇·w^(j) = 0   on Y_f (fluid domain)
#
# BOUNDARY CONDITIONS:
#   - Periodic: Outer cell walls (left↔right, bottom↔top) share velocity and pressure.
#   - No-slip (Wall): Circular inclusion boundary (w = 0).
#
# PERMEABILITY CALCULATION:
# The effective permeability K_ij is the volume average of the cell velocity:
#   K_ij = (1/|Y|) ∫_{Y_f} w_i^(j) dΩ
#
# NUMERICAL IMPLEMENTATION:
# 1. Gmsh is used to generate a conformant periodic mesh (opposite faces match).
# 2. XCALibre's `Gmsh2D_mesh` directly converts the Gmsh API data into Mesh2 format.
# 3. `construct_periodic` pairs matched faces to create periodic connectivity.
# 4. The new `PDEOperator` paradigm is used to define the Stokes system abstractly.
# ==============================================================================

using XCALibre
using XCALibre.UNV2
using LinearAlgebra
using KernelAbstractions
using Printf
using Gmsh
using Accessors

# ── Parameters ────────────────────────────────────────────────────────────────
const L         = 1.0     # half-cell side (cell is [-L, L]², side = 2L)
const R         = 0.5     # inclusion radius
const mesh_size = 0.08    # target element size
const ν         = 1.0     # kinematic viscosity

# ── Gmsh mesh generation ─────────────────────────────────────────────────────
"""
    build_2d_unit_cell(L, R, h) → mesh

Generate a periodic triangle mesh on [-L,L]² minus a circle of radius R.
Uses Gmsh API and converts directly to XCALibre Mesh2 format.
"""
function build_2d_unit_cell(L, R, h)
    gmsh.initialize()
    gmsh.model.add("unit_cell_2d")
    gmsh.option.setNumber("General.Terminal", 0)

    # ── Outer square ─────────────────────────────────────────────────────────
    p1 = gmsh.model.geo.addPoint(-L, -L, 0, h)
    p2 = gmsh.model.geo.addPoint( L, -L, 0, h)
    p3 = gmsh.model.geo.addPoint( L,  L, 0, h)
    p4 = gmsh.model.geo.addPoint(-L,  L, 0, h)

    l_bottom = gmsh.model.geo.addLine(p1, p2)
    l_right  = gmsh.model.geo.addLine(p2, p3)
    l_top    = gmsh.model.geo.addLine(p3, p4)
    l_left   = gmsh.model.geo.addLine(p4, p1)

    # ── Circular inclusion ────────────────────────────────────────────────────
    hc = h * 0.5
    c_cen = gmsh.model.geo.addPoint( 0,  0, 0, hc)
    c_p1  = gmsh.model.geo.addPoint( R,  0, 0, hc)
    c_p2  = gmsh.model.geo.addPoint( 0,  R, 0, hc)
    c_p3  = gmsh.model.geo.addPoint(-R,  0, 0, hc)
    c_p4  = gmsh.model.geo.addPoint( 0, -R, 0, hc)

    ca1 = gmsh.model.geo.addCircleArc(c_p1, c_cen, c_p2)
    ca2 = gmsh.model.geo.addCircleArc(c_p2, c_cen, c_p3)
    ca3 = gmsh.model.geo.addCircleArc(c_p3, c_cen, c_p4)
    ca4 = gmsh.model.geo.addCircleArc(c_p4, c_cen, c_p1)

    # ── Surface with hole ────────────────────────────────────────────────────
    outer_loop = gmsh.model.geo.addCurveLoop([l_bottom, l_right, l_top, l_left])
    inner_loop = gmsh.model.geo.addCurveLoop([ca1, ca2, ca3, ca4])
    gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])

    gmsh.model.geo.synchronize()

    # ── Periodic mesh: right←left (x shift 2L), top←bottom (y shift 2L) ────
    Lc = 2L
    tx = [1.,0,0, Lc, 0,1,0,0, 0,0,1,0, 0,0,0,1]
    ty = [1.,0,0,0, 0,1,0, Lc, 0,0,1,0, 0,0,0,1]
    gmsh.model.mesh.setPeriodic(1, [l_right], [l_left],   tx)
    gmsh.model.mesh.setPeriodic(1, [l_top],   [l_bottom], ty)

    # ── Named boundary patches ────────────────────────────────────────────────
    tL = gmsh.model.addPhysicalGroup(1, [l_left]);             gmsh.model.setPhysicalName(1, tL, "left")
    tR = gmsh.model.addPhysicalGroup(1, [l_right]);            gmsh.model.setPhysicalName(1, tR, "right")
    tB = gmsh.model.addPhysicalGroup(1, [l_bottom]);           gmsh.model.setPhysicalName(1, tB, "bottom")
    tT = gmsh.model.addPhysicalGroup(1, [l_top]);              gmsh.model.setPhysicalName(1, tT, "top")
    tC = gmsh.model.addPhysicalGroup(1, [ca1, ca2, ca3, ca4]); gmsh.model.setPhysicalName(1, tC, "circle")

    gmsh.model.mesh.generate(2)

    # ── Direct conversion ────────────────────────────────────────────────────
    mesh = Gmsh2D_mesh()
    gmsh.finalize()
    return mesh
end

# ── Stokes cell-problem solver ────────────────────────────────────────────────
function solve_cell_problem(model, config, e_j; pref=0.0)
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

        if rx < solvers.U.convergence && ry < solvers.U.convergence && rp < solvers.p.convergence
            @printf("    converged at iter %4d  (rx=%.2e  ry=%.2e  rp=%.2e)\n",
                    iter, rx, ry, rp)
            break
        end
    end

    # K_ij = (1/|Y|) ∫_{Y_f} w_i dΩ,  |Y| = (2L)²
    return volume_integral(U) ./ (2L)^2
end

# ── Setup ─────────────────────────────────────────────────────────────────────
@info "Building Gmsh unit cell mesh  (L=$L, R=$R, h=$mesh_size)..."
mesh = build_2d_unit_cell(L, R, mesh_size)
@info "Mesh ready: $(length(mesh.cells)) cells"

backend  = CPU(); workgroup = 1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# Periodic BCs: construct_periodic pairs boundary faces by spatial proximity.
# Conformant pairing is guaranteed by Gmsh's setPeriodic above.
periodic_x = construct_periodic(mesh, backend, :left, :right)
periodic_y = construct_periodic(mesh, backend, :bottom, :top)

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
        U = [periodic_x..., periodic_y..., Wall(:circle, [0.0, 0.0, 0.0])],
        p = [periodic_x..., periodic_y..., Wall(:circle)],
    )
)

schemes = (
    U = Schemes(divergence=Upwind),
    p = Schemes(divergence=Linear),
)

solvers = (
    U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-7, relax=0.7),
    p = SolverSetup(solver=Gmres(),    preconditioner=Jacobi(), convergence=1e-7, relax=0.3),
)

config = Configuration(
    solvers    = solvers,
    schemes    = schemes,
    runtime    = Runtime(iterations=2000, write_interval=-1, time_step=1.0),
    hardware   = hardware,
    boundaries = BCs,
)

# ── Permeability tensor ───────────────────────────────────────────────────────
K = zeros(2, 2)
for j in 1:2
    e_j = zeros(3); e_j[j] = 1.0
    @info "Cell problem  e_$j = $e_j"
    w_j = solve_cell_problem(model, config, e_j)
    K[:, j] .= w_j[1:2]
end

println("\n─────────────────────────────────────────────")
println("Permeability Tensor K:")
display(K)
@printf("\nK_xx = %.6e\nK_yy = %.6e\n", K[1,1], K[2,2])
@printf("Symmetry residual  ‖K - Kᵀ‖/‖K‖ = %.2e\n", norm(K - K') / max(norm(K), eps()))
porosity = total_volume(mesh_dev) / (2L)^2
@printf("Fluid porosity φ = %.4f\n", porosity)
println("─────────────────────────────────────────────")
@info "Done."
