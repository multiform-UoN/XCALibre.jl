using XCALibre
using LinearAlgebra
using Accessors
using Printf
using KernelAbstractions

# ==============================================================================
# Custom Stokes Upscaling Solver (Equivalent to kappaEqn & aEqn in homogenisationFoam)
# ==============================================================================
function stokes_upscaling!(model, config, J_macro; output=VTK(), pref=0.0)
    (; solvers, schemes, runtime, hardware, boundaries) = config
    (; U, p, Uf, pf) = model.momentum
    mesh = model.domain
    (; backend) = hardware

    @info "Setup Stokes Upscaling (kappa & a equations)..."
    ∇p = Grad{schemes.p.gradient}(p)
    mdotf = FaceScalarField(mesh)
    rDf = FaceScalarField(mesh)
    initialise!(rDf, 1.0)
    nueff = FaceScalarField(mesh)
    divHv = ScalarField(mesh)

    # Macroscopic pressure gradient source
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

    @info "Starting SIMPLE loops for Stokes Cell Problem..."
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

        if rx < solvers.U.convergence && ry < solvers.U.convergence && rp < solvers.p.convergence
            @info "Stokes Cell Problem converged at iter $iter. Res: Ux=$rx, p=$rp"
            break
        end
        if iter % 50 == 0
            @printf("Iter %d: Ux Res = %.2e, p Res = %.2e\n", iter, rx, rp)
        end
    end
    
    return mdotf
end

# ==============================================================================
# MAIN SCRIPT: homogenisationFoam in XCALibre.jl
# ==============================================================================

# 1. Setup Mesh & Config
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "flatplate_2D_laminar.unv"), scale=0.001)

backend = CPU(); workgroup = 1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = 1.0), # Viscosity = 1 for standard homogenisation
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
)

BCs = assign(
    region=mesh_dev,
    (
        U = [
            Extrapolated(:inlet),
            Extrapolated(:outlet),
            Wall(:wall, [0.0, 0.0, 0.0]),
            Symmetry(:top)
        ],
        p = [
            Dirichlet(:inlet, 0.0),
            Dirichlet(:outlet, 0.0),
            Wall(:wall),
            Symmetry(:top)
        ],
        psi = [
            Extrapolated(:inlet),
            Extrapolated(:outlet),
            Zerogradient(:wall),
            Symmetry(:top)
        ],
        X_corr = [
            Extrapolated(:inlet),
            Extrapolated(:outlet),
            Zerogradient(:wall),
            Symmetry(:top)
        ]
    )
)

schemes = (
    U = Schemes(divergence=Upwind),
    p = Schemes(divergence=Linear),
    psi = Schemes(divergence=Upwind, laplacian=Linear),
    X_corr = Schemes(divergence=Upwind, laplacian=Linear)
)

solvers = (
    U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-5, relax=0.7),
    p = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-5, relax=0.3),
    psi = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-7, relax=1.0),
    X_corr = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-7, relax=1.0)
)

config = Configuration(
    solvers=solvers, schemes=schemes, 
    runtime=Runtime(iterations=2000, write_interval=-1, time_step=1.0), 
    hardware=hardware, boundaries=BCs
)

initialise!(model.momentum.U, [0.0, 0.0, 0.0])
initialise!(model.momentum.p, 0.0)

# 2. Phase 1: Stokes Cell Problem (Calculate kappa / U)
# We apply a macroscopic pressure gradient in X-direction
J_macro = [1.0, 0.0, 0.0]
mdotf = stokes_upscaling!(model, config, J_macro; pref=0.0)

# Calculate Keff (Upscaled Permeability)
vols = [cell.volume for cell in mesh.cells]
vol_tot = sum(vols)
U_cpu = Array(model.momentum.U.x.values)
Keff_x = sum(U_cpu .* vols) / vol_tot
@printf("\nUpscaled Permeability Keff_xx = %.4e\n\n", Keff_x)

# 3. Phase 2: Spectral Cell Problem (Power Iterations for psi)
psi = ScalarField(mesh_dev)
psi_old = ScalarField(mesh_dev)
psi_rhs = ScalarField(mesh_dev)
initialise!(psi, 1.0)
initialise!(psi_old, 1.0)
initialise!(psi_rhs, 0.0)

gamma = ConstantScalar(0.01) # Molecular Diffusion D
R_val = ConstantScalar(0.1)  # Reaction rate R

psi_eqn = (
      Divergence{schemes.psi.divergence}(mdotf, psi)
    - Laplacian{schemes.psi.laplacian}(gamma, psi)
    + Si(R_val, psi)
    ==
    Source(psi_rhs) # RHS updated in power iterations: lambda * psi_old
) → ScalarEquation(psi, BCs.psi)

@reset psi_eqn.preconditioner = set_preconditioner(solvers.psi.preconditioner, psi_eqn)
@reset psi_eqn.solver = XCALibre._workspace(solvers.psi.solver, XCALibre._b(psi_eqn))

lambda = 1.0
lambda_old = 1.0
power_iters = 50

@info "Starting Spectral Cell Problem (Power Iterations)..."
for p_iter in 1:power_iters
    global lambda
    psi_old.values .= psi.values
    psi_rhs.values .= lambda .* psi_old.values

    # Solve linear system for new psi
    res_psi = solve_equation!(psi_eqn, psi, BCs.psi, solvers.psi, config)
    
    # Rayleigh-like quotient for lambda
    psi_c = Array(psi.values)
    psi_old_c = Array(psi_old.values)
    
    num = sum(psi_old_c .* psi_c .* vols)
    den = sum(psi_c .* psi_c .* vols)
    lambda_new = lambda * (num / den)
    
    err = abs(lambda_new - lambda) / abs(lambda)
    lambda = lambda_new
    
    if p_iter % 5 == 0
        @printf("Power Iter %d: lambda = %.5e (err = %.2e, solver_res = %.2e)\n", p_iter, lambda, err, res_psi)
    end
    
    if err < 1e-5
        @printf("Power Iterations converged! Final lambda = %.5e\n", lambda)
        break
    end
end

# Normalize Psi
psi_c = Array(psi.values)
norm_factor = sum(psi_c .* vols) / vol_tot
psi.values .= psi.values ./ norm_factor

# 4. Phase 3: Cell Corrector Problem (X)
# XEqn: div(U, X) - laplacian(D, X) = sourceX
# where sourceX = U_x - D * (dPsi/dx) ... simplified here for demonstration
@info "\nStarting Cell Corrector Problem..."
X_corr = ScalarField(mesh_dev)
initialise!(X_corr, 0.0)

sourceX = ScalarField(mesh_dev)
sourceX.values .= model.momentum.U.x.values # Approximation of source

X_eqn = (
      Divergence{schemes.X_corr.divergence}(mdotf, X_corr)
    - Laplacian{schemes.X_corr.laplacian}(gamma, X_corr)
    ==
    Source(sourceX)
) → ScalarEquation(X_corr, BCs.X_corr)

@reset X_eqn.preconditioner = set_preconditioner(solvers.X_corr.preconditioner, X_eqn)
@reset X_eqn.solver = XCALibre._workspace(solvers.X_corr.solver, XCALibre._b(X_eqn))

for iter in 1:100
    res_X = solve_equation!(X_eqn, X_corr, BCs.X_corr, solvers.X_corr, config)
    if iter % 20 == 0
        @printf("Corrector Iter %d: Res = %.2e\n", iter, res_X)
    end
    if res_X < solvers.X_corr.convergence
        @info "Cell Corrector converged at iter $iter"
        break
    end
end

@info "homogenisationFoam implementation in Julia completed successfully!"
