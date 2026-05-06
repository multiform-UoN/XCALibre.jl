using XCALibre
using LinearAlgebra
using Accessors
using Printf
using Statistics

# ==============================================================================
# Example: Single Thin-Film Equation (Darcy)
# ==============================================================================
# PDE: ∂h/∂t = ∇ ⋅ ( (Kh/μ) ∇p ) where p = -γ ∇²h
# Resulting in a 4th-order biharmonic-like equation for h.
# ==============================================================================

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.01)

backend = CPU(); workgroup=1024; activate_multithread(backend)
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

gamma = 0.01
mu = 1.0

BCs = assign((h = [Zerogradient(b.name) for b in mesh.boundaries],), region=mesh_dev)
schemes = (h = Schemes(time=Euler, laplacian=Linear),)
solvers = (h = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0),)
config = Configuration(solvers=solvers, schemes=schemes, runtime=Runtime(iterations=1, time_step=0.001), hardware=hardware, boundaries=BCs)

h = ScalarField(mesh_dev); initialise!(h, 0.1)
h.values .+= 0.01 .* rand(length(h.values)) 

@info "Solving Darcy Thin-Film (h mobility)..."
for step in 1:10
    global h
    # Mobility M = Kh/μ.
    M_val = mean(h.values) / mu
    
    h_eqn = (
          Time{schemes.h.time}(h)
        + Biharmonic{schemes.h.laplacian}(ConstantScalar(gamma * M_val), h)
        ==
        Source(ConstantScalar(0.0))
    ) → ScalarEquation(h, BCs.h)

    @reset h_eqn.preconditioner = set_preconditioner(solvers.h.preconditioner, h_eqn)
    @reset h_eqn.solver = XCALibre._workspace(solvers.h.solver, XCALibre._b(h_eqn))

    res = solve_equation!(h_eqn, h, BCs.h, solvers.h, config)
    @printf("Step %d: Res = %.2e, Mean h = %.4f\n", step, res, mean(h.values))
end
