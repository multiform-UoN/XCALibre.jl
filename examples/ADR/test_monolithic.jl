using XCALibre
using LinearAlgebra
using Printf
using Statistics

# Test script for Monolithic solver
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad.unv"), scale=0.01)
backend = CPU(); workgroup = 1024
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

BCs = assign(
    (
        C1 = [Zerogradient(:inlet), Zerogradient(:outlet), Zerogradient(:bottom), Zerogradient(:top)],
        C2 = [Zerogradient(:inlet), Zerogradient(:outlet), Zerogradient(:bottom), Zerogradient(:top)]
    ),
    region=mesh_dev
)

schemes = (C = Schemes(laplacian=Linear),)
solvers = (C = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0),)
config = Configuration(solvers=solvers, schemes=schemes, runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0), hardware=hardware, boundaries=BCs)

C1 = ScalarField(mesh_dev); initialise!(C1, 0.0)
C2 = ScalarField(mesh_dev); initialise!(C2, 0.0)

# Eq 1: C1 = 1
eqn1 = (Si(ConstantScalar(1.0), C1) == Source(ConstantScalar(1.0))) → ScalarEquation(C1, BCs.C1)
# Eq 2: C2 = 2
eqn2 = (Si(ConstantScalar(1.0), C2) == Source(ConstantScalar(2.0))) → ScalarEquation(C2, BCs.C2)

sys = MonolithicSystem([eqn1, eqn2], [C1, C2])
res = solve_monolithic!(sys, (BCs.C1, BCs.C2), config)

@printf("Mean C1: %.4f (Expected: 1.0000)\n", mean(C1.values))
@printf("Mean C2: %.4f (Expected: 2.0000)\n", mean(C2.values))
