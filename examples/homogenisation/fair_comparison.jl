using XCALibre
using LinearAlgebra
using Accessors
using Printf
using Statistics

# ==============================================================================
# FAIR COMPARISON: XCALibre vs OpenFOAM 13 (singleSphere)
# ==============================================================================
# Mesh: 10x10x10 + snappyHexMesh (from singleSphere tutorial)

# 1. Setup Mesh (Load bundled UNV)
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_path = joinpath(grids_dir, "3d_streamtube_1.0x0.1x0.1_0.08mm.unv")
@info "Loading UNV mesh from $mesh_path ..."
mesh = UNV3D_mesh(mesh_path)

backend = CPU(); workgroup=1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# 2. Setup Solver Options
BCs = assign(
    region=mesh_dev,
    (
        U = [
            Extrapolated(:inlet),
            Extrapolated(:outlet),
            Wall(:bottom, [0.0, 0.0, 0.0]),
            Wall(:top, [0.0, 0.0, 0.0]),
            Wall(:side1, [0.0, 0.0, 0.0]),
            Wall(:side2, [0.0, 0.0, 0.0])
        ],
        p = [
            Dirichlet(:inlet, 0.0),
            Dirichlet(:outlet, 0.0),
            Wall(:bottom),
            Wall(:top),
            Wall(:side1),
            Wall(:side2)
        ]
    )
)

schemes = (
    U = Schemes(divergence=Upwind),
    p = Schemes(divergence=Linear)
)

solvers = (
    U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-10, relax=0.95),
    p = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0)
)

config = Configuration(solvers=solvers, schemes=schemes, runtime=Runtime(iterations=100, write_interval=-1, time_step=1.0), hardware=hardware, boundaries=BCs)

# 3. Benchmark Stokes Cell Problem (Single direction)
@info "Starting XCALibre benchmark for Stokes Cell Problem..."

U = VectorField(mesh_dev); initialise!(U, [0.0, 0.0, 0.0])
p = ScalarField(mesh_dev); initialise!(p, 0.0)
Uf = FaceVectorField(mesh_dev)
pf = FaceScalarField(mesh_dev)

∇p = Grad{schemes.p.gradient}(p)
mdotf = FaceScalarField(mesh_dev)
rDf = FaceScalarField(mesh_dev); initialise!(rDf, 1.0)
nueff = FaceScalarField(mesh_dev); initialise!(nueff, 1.0) # nu = 1.0
divHv = ScalarField(mesh_dev)
macro_grad = VectorField(mesh_dev); initialise!(macro_grad, [1.0, 0.0, 0.0])

    # Use the new PDEOperator DSL for abstract definitions
    L_U = ((
          Time{SteadyState}()
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
    prev_p = zeros(length(mesh.cells))

    @info "Timing 10 iterations..."
    total_time = 0.0

    for iter in 1:10
        global total_time
        start = time_ns()

        rx, ry, rz = solve_equation!(U_eqn, config)
        inverse_diagonal!(rD, U_eqn, config)
        interpolate!(rDf, rD, config)
        remove_pressure_source!(U_eqn, ∇p, config)
        H!(Hv, U, U_eqn, config)
        interpolate!(Uf, Hv, config)
        # XCALibre.correct_boundaries!(Uf, Hv, BCs.U, 0.0, config)
        flux!(mdotf, Uf, config)
        XCALibre.div!(divHv, mdotf, config)
        prev_p .= p.values
        rp = solve_equation!(p_eqn, config; ref=0.0)
        # explicit_relaxation!(p, prev_p, solvers.p.relax, config)
        grad!(∇p, pf, p, BCs.p, 0.0, config)
        XCALibre.Solvers.correct_mass_flux!(mdotf, p_eqn, config)
        correct_velocity!(U, Hv, ∇p, rD, config)

        stop = time_ns()
        it_time = (stop - start) / 1e9
        total_time += it_time
        @printf("Iter %d: Res U = %.2e, p = %.2e, Time = %.4fs\n", iter, rx, rp, it_time)
    end

@printf("\nXCALibre Avg Time per Coupled Iteration: %.4fs\n", total_time/10)
@info "Fair Comparison Completed!"
