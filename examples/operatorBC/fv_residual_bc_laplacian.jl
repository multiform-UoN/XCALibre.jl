# =============================================================================
# Residual/Jacobian BC Actions on a Real FV Operator
# =============================================================================
#
# The PDE operator is assembled by XCALibre as usual. The only experiment here is
# the nonlinear boundary equation, represented as explicit Newton-row actions.
# This keeps the finite-volume stencil inspectable and avoids putting sparse
# storage details inside the BC semantics.
# =============================================================================

using XCALibre
using LinearAlgebra
using Printf

include("tutorial_utils.jl")

function assemble_laplacian(phi, BCs, config, solver)
    gamma = ConstantScalar(1.0)
    L_phi = ((-Laplacian{Linear}(gamma) == Source(0.0)) → BCs.phi) → solver
    sys = MonolithicSystem([L_phi(phi)], [phi])
    A_csr, b = assemble_monolithic_system(sys, (BCs.phi,), config)
    return tutorial_sparse_matrix(A_csr), b
end

function main(; max_steps=5)
    mesh_cpu, _ = tutorial_straight_mesh()
    mesh_dev = adapt(CPU(), mesh_cpu)

    phi = ScalarField(mesh_dev)
    initialise!(phi, 0.0)

    ordinary_bcs = [
        Dirichlet(:inlet, 0.0),
        Dirichlet(:outlet, 0.0),
        Dirichlet(:top, 0.0),
        Dirichlet(:bottom, 0.0),
    ]
    BCs = assign(region=mesh_dev, (phi=ordinary_bcs,))

    solver = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0)
    config = Configuration(
        solvers=(phi=solver,),
        schemes=(phi=Schemes(laplacian=Linear),),
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0),
        hardware=Hardware(backend=CPU(), workgroup=1024),
        boundaries=BCs,
    )

    boundary_row = first(mesh_dev.boundary_cellsID)
    nonlinear_bc = LocalScalarResidualBC(
        boundary_row;
        residual = u -> u + u^2 - 10.0,
        jacobian = u -> 1.0 + 2.0*u,
    )

    state = Vector(phi.values)
    state[boundary_row] = 2.0

    root = (-1.0 + sqrt(41.0)) / 2.0
    @printf("Residual BC row: %d, target positive root: %.8f\n", boundary_row, root)

    for step in 1:max_steps
        phi.values .= state
        A, b = assemble_laplacian(phi, BCs, config, solver)

        residual = A * state - b
        J = copy(A)
        rhs = -residual
        apply_boundary_actions!(J, rhs, boundary_actions(nonlinear_bc, state))

        delta = J \ rhs
        state .+= delta

        bc_res = residual_value(nonlinear_bc, state)
        pde_residual = A * state - b
        pde_residual[boundary_row] = 0.0
        linearized_res = norm(J * delta - rhs)

        @printf("step %d: phi[row]=%.8f | BC residual=%.3e | PDE residual excl. BC=%.3e | linear solve residual=%.3e\n",
                step, state[boundary_row], bc_res, norm(pde_residual), linearized_res)
    end
end

main()
