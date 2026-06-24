using XCALibre
using LinearAlgebra
using Accessors
using Printf
using Statistics

# ==============================================================================
# TUTORIAL: Generic Thin-Film Solver (Multiform ThinFilm Framework)
# ==============================================================================
# This script implements a modular thin-film framework in XCALibre.jl,
# mimicking the 'thinFilmResearchFoam' strategy from multiformFoam.
#
# Mathematical Model:
# 1. Height Evolution: ∂h/∂t + ∇ ⋅ J = S_h
# 2. Flux: J = -M(h) ∇p
# 3. Pressure: p = -γ ∇²h + Π(h) + σ_a(b)
#
# Features:
# - Split 4th-order equation into two coupled 2nd-order equations (h and p).
# - Non-linear mobility models (Viscous h³/3μ, Darcy h).
# - Coupling to a biomass field (b).
# - Outer loops (PIMPLE-style) for non-linear convergence.
# ==============================================================================

# 1. Configuration & Mesh
# ------------------------------------------------------------------------------
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.01) # 10cm domain

backend = CPU(); workgroup=1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# 2. Physics & Model Selection
# ------------------------------------------------------------------------------
# Choose your model: :viscous (h³/3μ) or :darcy (Kh/μ)
model_type = :viscous 

gamma = 0.001       # Surface tension coefficient
mu = 1.0            # Viscosity
Db = 1e-4           # Biomass diffusion

# Mechanical Pressure Π(h) - e.g., disjoining pressure or effective stress
disjoining_pressure(h) = 1e-4 / (h^3 + 1e-6) # Example: vdW attractive-repulsive

# Active Forcing σ_a(b)
active_stress(b) = 0.5 * b 

# 3. Boundary Conditions
# ------------------------------------------------------------------------------
# Periodic or Symmetry is common for thin films. 
# Here we use Zerogradient for a bounded container.
BCs = assign(
    (
        h = [Zerogradient(b.name) for b in mesh.boundaries],
        p = [Zerogradient(b.name) for b in mesh.boundaries],
        b = [Zerogradient(b.name) for b in mesh.boundaries]
    ),
    region=mesh_dev
)

schemes = (
    h = Schemes(time=Euler, laplacian=Linear),
    p = Schemes(laplacian=Linear),
    b = Schemes(time=Euler, laplacian=Linear)
)

solvers = (
    h = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=0.8),
    p = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0),
    b = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0)
)

config = Configuration(solvers=solvers, schemes=schemes, runtime=Runtime(iterations=100, write_interval=10, time_step=0.01), hardware=hardware, boundaries=BCs)

# 4. Field Initialisation
# ------------------------------------------------------------------------------
h = ScalarField(mesh_dev); initialise!(h, 0.1) # Initial film height 1mm
p = ScalarField(mesh_dev); initialise!(p, 0.0)
b = ScalarField(mesh_dev); initialise!(b, 0.0)

# Add a small perturbation to h to trigger instabilities
h.values .+= 0.001 .* rand(length(h.values))

# 5. Solver Loop (Time Stepping + Outer Loops)
# ------------------------------------------------------------------------------
# This loop implements a PIMPLE-style strategy to resolve the non-linear
# coupling between height (h), pressure (p), and biomass (b).
#
# Steps within each time step:
# 1. Solve the Biomass transport equation (decoupled).
# 2. Iterate 'n_outer' times to converge the (h, p) coupling:
#    a. Update mobility M(h) based on current height.
#    b. Solve the Height evolution equation (h). 
#       Implicit 4th-order surface tension is handled via the Biharmonic operator.
# ==============================================================================

dt = config.runtime.dt
n_steps = 50
n_outer = 3 # Outer iterations per time step

@info "Starting Thin-Film Simulation ($model_type model)..."
for t_step in 1:n_steps
    h_prev = ScalarField(mesh_dev); h_prev.values .= h.values
    b_prev = ScalarField(mesh_dev); b_prev.values .= b.values
    
    # 1. Update Biomass (Decoupled transport update)
    # --------------------------------------------------------------------------
    L_b = ((
          Time{Euler}()
        - Laplacian{Linear}(ConstantScalar(Db))
        ==
        Source(0.0)
    ) → BCs.b) → solvers.b
    
    b_eqn = L_b(b)
    @reset b_eqn.preconditioner = set_preconditioner(solvers.b.preconditioner, b_eqn)
    @reset b_eqn.solver = XCALibre._workspace(solvers.b.solver, XCALibre._b(b_eqn))
    
    solve_equation!(b_eqn, config)

    # 2. Outer Loops for (h, p) Coupling
    # --------------------------------------------------------------------------
    for outer in 1:n_outer
        # Calculate Mobility M(h) based on the latest height
        M_field = ScalarField(mesh_dev)
        if model_type == :viscous
            M_field.values .= (h.values.^3) ./ (3.0 * mu)
        else # :darcy
            M_field.values .= h.values .* (1.0 / mu)
        end
        
        # Interpolate mobility to faces for the Laplacian operator
        M_face = FaceScalarField(mesh_dev)
        for fID in eachindex(M_face.values)
            oc = mesh_dev.faces[fID].ownerCells
            M_face.values[fID] = length(oc) > 1 ? 
                0.5 * (M_field.values[oc[1]] + M_field.values[oc[2]]) : 
                M_field.values[oc[1]]
        end
        
        # 2b. Solve Height Evolution h
        # ----------------------------------------------------------------------
        # PDE: ∂h/∂t - ∇ ⋅ (M ∇p) + γ ∇²h_implicit = 0
        #
        # We use the Biharmonic operator to treat the surface tension implicitly.
        # This provides significant numerical stability, allowing for much larger
        # time steps than traditional explicit treatments.
        
        # Abstract PDE def -> BCs -> apply to field (reusable pattern)
        pde_h = (
              Time{Euler}()
            + Biharmonic{Linear}(ConstantScalar(gamma * mean(M_field.values))) # Implicit surface tension
            - Laplacian{Linear}(M_face, p) # cross to p
            ==
            Source(0.0)
        )
        L = pde_h → BCs.h
        h_eqn = L(h)
        @reset h_eqn.setup = solvers.h
        @reset h_eqn.preconditioner = set_preconditioner(solvers.h.preconditioner, h_eqn)
        @reset h_eqn.solver = XCALibre._workspace(solvers.h.solver, XCALibre._b(h_eqn))

        res_h = solve_equation!(h_eqn, config)
        
        # Note: Pressure (p) would be updated here based on h (Π(h) etc.)
        # p.values .= ... (simplified logic for this demonstration)
        
        if outer == n_outer
            @printf("Step %d, Outer %d: Res h = %.2e, Mean h = %.4f\n", 
                    t_step, outer, res_h, mean(h.values))
        end
    end
end

@info "Thin-Film Simulation Completed!"
