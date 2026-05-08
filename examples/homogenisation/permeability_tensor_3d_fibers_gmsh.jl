# ==============================================================================
# 3D Periodic Homogenisation — Random-Fiber Microstructure
# ==============================================================================
# This example computes the 3x3 permeability tensor K for a complex 3D 
# geometry consisting of random cylinders (fibers).
#
# GEOMETRY CONSTRUCTION (CSG):
# To ensure the microstructure is truly periodic, the fibers must "wrap around" 
# the unit cell boundaries. The following steps are performed via Gmsh OCC:
#   1. Generate N "master" fibers with random positions and orientations.
#   2. Create a 3x3x3 grid of periodic image copies (shifts of ±2L).
#   3. Intersect all these cylinders with the unit cell box [-L, L]³.
#   4. Subtract the resulting periodic fiber fragments from the box.
# This ensures that a fiber exiting the "right" face perfectly re-enters 
# the "left" face.
#
# NUMERICAL METHOD:
# - Solves 3 independent Stokes cell problems (forcing in X, Y, and Z).
# - Uses `pair_periodic_surfaces` to match the fragmented surface patches 
#   created by fiber–wall intersections.
# - Uses the `PDEOperator` paradigm to handle the complex 3D momentum equations.
#
# PERMEABILITY:
#   K_ij = (1/|Y|) ∫_{Y_f} w_i^(j) dΩ, where |Y| = (2L)³
# ==============================================================================

using XCALibre
using XCALibre.UNV3
using LinearAlgebra
using KernelAbstractions
using Random
using Printf
using Gmsh
using Accessors

# ... (rest of the file)

# ── Parameters ────────────────────────────────────────────────────────────────
const L             = 1.0     # half-cell side (cell is [-L, L]³, side = 2L)
const N_fibers      = 1       # number of master fibers
const R_min         = 0.05    # minimum fiber radius
const R_max         = 0.12    # maximum fiber radius
const fiber_length  = 4.0     # fiber length (should exceed 2L for continuity)
const mesh_size     = 0.3     # characteristic element size
const seed          = 44      # RNG seed
const ν             = 1.0     # kinematic viscosity

# ── Utilities ─────────────────────────────────────────────────────────────────
function random_unit_vector(rng::AbstractRNG)
    ϕ = 2π * rand(rng); z = 2rand(rng) - 1; ρ = sqrt(max(0.0, 1 - z^2))
    return (ρ * cos(ϕ), ρ * sin(ϕ), z)
end

"""
    pair_periodic_surfaces(master_tags, slave_tags, shift; tol) → (paired_slave, paired_master)

Match fragmented opposite boundary surface fragments using translation-invariant
geometric signatures (bounding box, centre-of-mass, area, edge lengths).
Identical to the Ferrite reference implementation.
"""
function pair_periodic_surfaces(master_tags::Vector{Int}, slave_tags::Vector{Int},
                                 shift::NTuple{3,Float64}; tol::Float64)
    isempty(master_tags) && error("No periodic master surfaces found.")
    isempty(slave_tags)  && error("No periodic slave surfaces found.")
    length(master_tags) == length(slave_tags) ||
        error("Different number of periodic surface fragments.")

    function signature(tag::Int)
        bbox = gmsh.model.occ.getBoundingBox(2, tag)
        com  = gmsh.model.occ.getCenterOfMass(2, tag)
        area = gmsh.model.occ.getMass(2, tag)
        bcurves = gmsh.model.getBoundary([(2, tag)], false, false, false)
        curve_lengths = sort(Float64[gmsh.model.occ.getMass(1, abs(c[2])) for c in bcurves])
        return (; bbox, com, area, curve_lengths)
    end

    master_sig = Dict(t => signature(t) for t in master_tags)
    slave_sig  = Dict(t => signature(t) for t in slave_tags)
    paired_master = Int[]; paired_slave = Int[]
    used = Set{Int}(); tol_area = max(1e-10, tol^2); tol_len = max(1e-9, tol)

    for s in sort(slave_tags)
        ss = slave_sig[s]; candidates = Tuple{Float64,Int}[]
        for m in master_tags
            m in used && continue
            ms = master_sig[m]
            length(ms.curve_lengths) == length(ss.curve_lengths) || continue
            all(i -> isapprox(ms.curve_lengths[i], ss.curve_lengths[i];
                              atol=tol_len, rtol=1e-6), eachindex(ms.curve_lengths)) || continue
            isapprox(ms.bbox[1]+shift[1], ss.bbox[1]; atol=tol) &&
            isapprox(ms.bbox[2]+shift[2], ss.bbox[2]; atol=tol) &&
            isapprox(ms.bbox[3]+shift[3], ss.bbox[3]; atol=tol) &&
            isapprox(ms.bbox[4]+shift[1], ss.bbox[4]; atol=tol) &&
            isapprox(ms.bbox[5]+shift[2], ss.bbox[5]; atol=tol) &&
            isapprox(ms.bbox[6]+shift[3], ss.bbox[6]; atol=tol) || continue
            isapprox(ms.com[1]+shift[1], ss.com[1]; atol=tol) &&
            isapprox(ms.com[2]+shift[2], ss.com[2]; atol=tol) &&
            isapprox(ms.com[3]+shift[3], ss.com[3]; atol=tol) || continue
            isapprox(ms.area, ss.area; atol=tol_area, rtol=1e-6) || continue
            score = abs(ms.com[1]+shift[1]-ss.com[1]) +
                    abs(ms.com[2]+shift[2]-ss.com[2]) +
                    abs(ms.com[3]+shift[3]-ss.com[3]) + abs(ms.area-ss.area)
            push!(candidates, (score, m))
        end
        isempty(candidates) && error("Could not pair surface fragment tag=$s")
        sort!(candidates, by=first)
        m_best = candidates[1][2]
        push!(paired_master, m_best); push!(paired_slave, s); push!(used, m_best)
    end
    length(used) == length(master_tags) || error("Periodic surface pairing is not one-to-one.")
    return paired_slave, paired_master
end

# ── Gmsh mesh generation ─────────────────────────────────────────────────────
"""
    build_fiber_unit_cell(L, N, R_min, R_max, len, h; seed) → mesh_file

Generate a 3D tet mesh of the unit cell [-L,L]³ minus N random periodic fibers.
Outer faces are fragmented by fiber–wall intersections; `pair_periodic_surfaces`
matches the fragments and `setPeriodic` ensures a conformant periodic mesh.
"""
function build_fiber_unit_cell(L, N_fibers, R_min, R_max, fiber_length, h; seed=42)
    rng = MersenneTwister(seed)
    gmsh.initialize()
    gmsh.model.add("fibers_3d")
    gmsh.option.setNumber("General.Terminal", 0)

    try
        box_tag = gmsh.model.occ.addBox(-L, -L, -L, 2L, 2L, 2L)

        # ── Master fibers ────────────────────────────────────────────────────
        masters = Tuple{Int,Int}[]
        for _ in 1:N_fibers
            r  = R_min + (R_max - R_min) * rand(rng)
            x0 = (2L*rand(rng)-L, 2L*rand(rng)-L, 2L*rand(rng)-L)
            u  = random_unit_vector(rng)
            a  = (x0[1]-0.5*fiber_length*u[1], x0[2]-0.5*fiber_length*u[2],
                  x0[3]-0.5*fiber_length*u[3])
            ax = (fiber_length*u[1], fiber_length*u[2], fiber_length*u[3])
            tag = gmsh.model.occ.addCylinder(a[1], a[2], a[3], ax[1], ax[2], ax[3], r)
            push!(masters, (3, tag))
        end

        # ── 3×3×3 periodic images ─────────────────────────────────────────────
        all_cylinders = Tuple{Int,Int}[]
        for sx in (-2L, 0.0, 2L), sy in (-2L, 0.0, 2L), sz in (-2L, 0.0, 2L)
            for m in masters
                if sx == 0.0 && sy == 0.0 && sz == 0.0
                    push!(all_cylinders, m)
                else
                    c = gmsh.model.occ.copy([m]); gmsh.model.occ.translate(c, sx, sy, sz)
                    append!(all_cylinders, c)
                end
            end
        end

        # ── Crop & subtract ──────────────────────────────────────────────────
        cropped, _ = gmsh.model.occ.intersect(all_cylinders, [(3, box_tag)], -1, true, false)
        isempty(cropped) && error("No fiber intersects the unit cell.")
        out_tags, _ = gmsh.model.occ.cut([(3, box_tag)], cropped, -1, true, true)
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

        # ── Classify surfaces by bounding box ────────────────────────────────
        left_s=Int[]; right_s=Int[]; bottom_s=Int[]; top_s=Int[]
        front_s=Int[]; back_s=Int[]; fiber_s=Int[]
        tol = max(1e-7, 1e-5 * L)

        for surf in gmsh.model.getEntities(2)
            tag = surf[2]
            xmin,ymin,zmin,xmax,ymax,zmax = gmsh.model.occ.getBoundingBox(2, tag)
            if     isapprox(xmin,-L;atol=tol) && isapprox(xmax,-L;atol=tol); push!(left_s,   tag)
            elseif isapprox(xmin, L;atol=tol) && isapprox(xmax, L;atol=tol); push!(right_s,  tag)
            elseif isapprox(ymin,-L;atol=tol) && isapprox(ymax,-L;atol=tol); push!(bottom_s, tag)
            elseif isapprox(ymin, L;atol=tol) && isapprox(ymax, L;atol=tol); push!(top_s,    tag)
            elseif isapprox(zmin,-L;atol=tol) && isapprox(zmax,-L;atol=tol); push!(front_s,  tag)
            elseif isapprox(zmin, L;atol=tol) && isapprox(zmax, L;atol=tol); push!(back_s,   tag)
            else                                                                push!(fiber_s,  tag)
            end
        end

        isempty(left_s) && error("No surfaces found for left face.")
        isempty(right_s) && error("No surfaces found for right face.")
        isempty(bottom_s) && error("No surfaces found for bottom face.")
        isempty(top_s) && error("No surfaces found for top face.")
        isempty(front_s) && error("No surfaces found for front face.")
        isempty(back_s) && error("No surfaces found for back face.")

        # ── Pair fragmented periodic faces, then set periodic mesh ────────────
        Lc = 2L
        tx = [1.,0,0, Lc, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        ty = [1.,0,0,0, 0,1,0, Lc, 0,0,1,0, 0,0,0,1]
        tz = [1.,0,0,0, 0,1,0,0, 0,0,1, Lc, 0,0,0,1]

        right_p, left_p   = pair_periodic_surfaces(left_s,   right_s,  ( Lc, 0.0, 0.0); tol=tol)
        top_p,   bottom_p = pair_periodic_surfaces(bottom_s, top_s,    (0.0,  Lc, 0.0); tol=tol)
        back_p,  front_p  = pair_periodic_surfaces(front_s,  back_s,   (0.0, 0.0,  Lc); tol=tol)

        gmsh.model.mesh.setPeriodic(2, right_p,  left_p,   tx)
        gmsh.model.mesh.setPeriodic(2, top_p,    bottom_p, ty)
        gmsh.model.mesh.setPeriodic(2, back_p,   front_p,  tz)

        # ── Physical Groups ──────────────────────────────────────────────────
        tL  = gmsh.model.addPhysicalGroup(2, left_s);   gmsh.model.setPhysicalName(2, tL,  "left")
        tR  = gmsh.model.addPhysicalGroup(2, right_s);  gmsh.model.setPhysicalName(2, tR,  "right")
        tBo = gmsh.model.addPhysicalGroup(2, bottom_s); gmsh.model.setPhysicalName(2, tBo, "bottom")
        tTo = gmsh.model.addPhysicalGroup(2, top_s);    gmsh.model.setPhysicalName(2, tTo, "top")
        tFr = gmsh.model.addPhysicalGroup(2, front_s);  gmsh.model.setPhysicalName(2, tFr, "front")
        tBa = gmsh.model.addPhysicalGroup(2, back_s);   gmsh.model.setPhysicalName(2, tBa, "back")
        if !isempty(fiber_s)
            tFb = gmsh.model.addPhysicalGroup(2, fiber_s); gmsh.model.setPhysicalName(2, tFb, "fibers")
        end
        tV = gmsh.model.addPhysicalGroup(3, [out_tags[1][2]]); gmsh.model.setPhysicalName(3, tV, "fluid")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        gmsh.model.mesh.generate(3)

        mesh = Gmsh3D_mesh()
        gmsh.finalize()
        return mesh

    catch err
        gmsh.finalize(); rethrow(err)
    end
end

# ── Stokes cell-problem solver ────────────────────────────────────────────────
function solve_cell_problem_fibers(model, config, e_j; pref=0.0)
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

    return volume_integral(U) ./ (2L)^3
end

# ── Mesh generation ───────────────────────────────────────────────────────────
@info "Building fiber unit cell mesh  (L=$L, N=$N_fibers, R=$(R_min)-$(R_max), h=$mesh_size)..."
mesh = build_fiber_unit_cell(L, N_fibers, R_min, R_max, fiber_length, mesh_size; seed=seed)
@info "Mesh loaded: $(length(mesh.cells)) cells, $(length(mesh.boundaries)) patches"

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

has_fibers = :fibers in Symbol.([b.name for b in mesh.boundaries])
BCs = assign(
    region = mesh_dev,
    (
        U = vcat([periodic_x..., periodic_y..., periodic_z...],
                  has_fibers ? [Wall(:fibers, [0.0, 0.0, 0.0])] : []),
        p = vcat([periodic_x..., periodic_y..., periodic_z...],
                  has_fibers ? [Wall(:fibers)] : []),
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
    w_j = solve_cell_problem_fibers(model, config, e_j)
    K[:, j] .= w_j
end

println("\n─────────────────────────────────────────────")
println("Permeability Tensor K:")
display(K)
@printf("\nDiagonal:  K_xx=%.4e  K_yy=%.4e  K_zz=%.4e\n", K[1,1], K[2,2], K[3,3])
@printf("Symmetry residual  ‖K - Kᵀ‖/‖K‖ = %.2e\n", norm(K - K') / max(norm(K), eps()))
porosity = total_volume(mesh_dev) / (2L)^3
@printf("Fluid porosity φ = %.4f\n", porosity)
println("─────────────────────────────────────────────")
@info "Done.  Compare with temp/PoreScaleHomogenisation.jl/homogenization_3d_fibers.jl"
