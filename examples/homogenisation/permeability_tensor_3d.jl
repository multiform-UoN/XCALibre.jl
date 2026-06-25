using XCALibre
using LinearAlgebra
using Accessors
using Printf
using KernelAbstractions

# ==============================================================================
# Multi-Directional Stokes Cell Problem (Permeability Tensor 3D)
# ==============================================================================
function solve_stokes_direction_3d(model, config, J_macro; pref=0.0)
    (; solvers, schemes, runtime, hardware, boundaries) = config
    (; U, p, Uf, pf) = model.momentum
    mesh = model.domain
    (; backend) = hardware

    # Re-initialise fields
    initialise!(U, [0.0, 0.0, 0.0])
    initialise!(p, 0.0)

    ∇p = Grad{schemes.p.gradient}(p)
    mdotf = FaceScalarField(mesh)
    rDf = FaceScalarField(mesh)
    initialise!(rDf, 1.0)
    nueff = FaceScalarField(mesh)
    divHv = ScalarField(mesh)

    macro_grad = VectorField(mesh)
    initialise!(macro_grad, J_macro)

    U_eqn = (
        Time{schemes.U.time}(U)
        + Divergence{schemes.U.divergence}(mdotf, U)
        - Laplacian{schemes.U.laplacian}(nueff, U)
        ==
        - Source(∇p.result) + Source(macro_grad)
    ) → VectorEquation(U, boundaries.U)

    p_eqn = (
        - Laplacian{schemes.p.laplacian}(rDf, p) == - Source(divHv)
    ) → ScalarEquation(p, boundaries.p)

    @reset U_eqn.preconditioner = set_preconditioner(solvers.U.preconditioner, U_eqn)
    @reset p_eqn.preconditioner = set_preconditioner(solvers.p.preconditioner, p_eqn)
    @reset U_eqn.solver = XCALibre._workspace(solvers.U.solver, XCALibre._b(U_eqn, XDir()))
    @reset p_eqn.solver = XCALibre._workspace(solvers.p.solver, XCALibre._b(p_eqn))

    turbulenceModel, config = initialise(model.turbulence, model, mdotf, p_eqn, config)

    (; nu) = model.fluid
    (; iterations) = runtime
    n_cells = length(mesh.cells)

    gradU = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    S = StrainRate(gradU, gradUT, U, Uf)

    Hv = VectorField(mesh)
    rD = ScalarField(mesh)
    prev = KernelAbstractions.zeros(backend, _get_float(mesh), n_cells)

    time = 0.0
    interpolate!(Uf, U, config)
    XCALibre.correct_boundaries!(Uf, U, boundaries.U, time, config)
    flux!(mdotf, Uf, config)
    grad!(∇p, pf, p, boundaries.p, time, config)
    update_nueff!(nueff, nu, model.turbulence, config)

    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    for iter ∈ 1:iterations
        rx, ry, rz = solve_equation!(U_eqn, U, boundaries.U, solvers.U, xdir, ydir, zdir, config)
        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rDf, rD, config)
        remove_pressure_source!(U_eqn, ∇p, config)
        H!(Hv, U, U_eqn, config)
        interpolate!(Uf, Hv, config)
        XCALibre.correct_boundaries!(Uf, Hv, boundaries.U, time, config)
        flux!(mdotf, Uf, config)
        XCALibre.div!(divHv, mdotf, config)
        prev .= p.values
        rp = solve_equation!(p_eqn, p, boundaries.p, solvers.p, config; ref=pref)
        explicit_relaxation!(p, prev, solvers.p.relax, config)
        grad!(∇p, pf, p, boundaries.p, time, config)
        XCALibre.Solvers.correct_mass_flux!(mdotf, p_eqn, config)
        correct_velocity!(U, Hv, ∇p, rD, config)
        update_nueff!(nueff, nu, model.turbulence, config)

        if rx < solvers.U.convergence && ry < solvers.U.convergence && (typeof(mesh) <: Mesh3 ? rz < solvers.U.convergence : true) && rp < solvers.p.convergence
            @info "Converged direction at iter $iter"
            break
        end
    end

    # Calculate volume average velocity
    vols = [cell.volume for cell in mesh.cells]
    vol_tot = sum(vols)
    Ux_avg = sum(Array(U.x.values) .* vols) / vol_tot
    Uy_avg = sum(Array(U.y.values) .* vols) / vol_tot
    Uz_avg = sum(Array(U.z.values) .* vols) / vol_tot

    return [Ux_avg, Uy_avg, Uz_avg]
end

# 1. Setup
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "3d_streamtube_1.0x0.1x0.1_0.08mm.unv"
mesh = UNV3D_mesh(joinpath(grids_dir, grid))

backend = CPU(); workgroup = 1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = 1.0),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
)

# Detect boundary names for 3D mesh
boundary_names = [b.name for b in mesh.boundaries]
@info "Found boundaries: $boundary_names"

# Simple BC setup for periodic-like cell problem (all extrapolated/slip or similar)
BCs = assign(
    region=mesh_dev,
    (
        U = [Extrapolated(name) for name in boundary_names],
        p = [Dirichlet(name, 0.0) for name in boundary_names]
    )
)

schemes = (
    U = Schemes(divergence=Upwind),
    p = Schemes(divergence=Linear)
)

solvers = (
    U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-5, relax=0.7),
    p = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-5, relax=0.3)
)

config = Configuration(solvers=solvers, schemes=schemes, runtime=Runtime(iterations=200, write_interval=-1, time_step=1.0), hardware=hardware, boundaries=BCs)

# 2. Solve for 3D Permeability Tensor
dim = 3
K = zeros(dim, dim)

for j in 1:dim
    e_j = zeros(3)
    e_j[j] = 1.0
    @info "Solving for forcing direction e_$j = $e_j"
    w_j = solve_stokes_direction_3d(model, config, e_j)
    for i in 1:dim
        K[i, j] = w_j[i]
    end
end

println("\n3D Permeability Tensor K:")
display(K)
println("\nSymmetry Residual: ", norm(K - K') / norm(K))
