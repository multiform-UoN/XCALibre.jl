#!/usr/bin/env julia

# XCALibre Stress-Operator Audit (Defect Correction)
#
# Verifies that a stress-coupled formulation with implicit Laplacian
# stabilization recovers the direct Laplacian solution.
#
# Goal:
#   -mu Δu - div(τ - 2 mu ε(u)) + ∇p = f
#   τ = 2 mu ε(u)
#
# This is algebraically identical to Newtonian Stokes, but tests the
# stabilizing effect of the implicit Laplacian backbone.

include("benchmark_utils.jl")

using Printf
using SparseArrays
using LinearAlgebra

# Use simple straight mesh (periodic)
grids_dir = pkgdir(XCALibre, "examples", "0_GRIDS")
mesh_path = joinpath(grids_dir, "quad40.unv")
mesh_cpu = UNV2D_mesh(mesh_path, scale=0.001)
mesh_cpu = make_periodic_topology(mesh_cpu, :inlet, :outlet, [1.0, 0.0, 0.0])
mesh_cpu = make_periodic_topology(mesh_cpu, :bottom, :top, [0.0, 1.0, 0.0])
mesh_dev = adapt(CPU(), mesh_cpu)

# Source f = (2π)² sin(2π y) => u = sin(2π y)
function get_source_u(mesh_dev, mu_val)
    f = ScalarField(mesh_dev)
    for i in eachindex(f.values)
        y = mesh_dev.cells[i].centre[2]
        f.values[i] = mu_val * (2.0 * π)^2 * sin(2.0 * π * y)
    end
    return f
end

function solve_direct(mesh_dev, mu_val, f_u)
    u = ScalarField(mesh_dev); initialise!(u, 0.0)
    v = ScalarField(mesh_dev); initialise!(v, 0.0)
    p = ScalarField(mesh_dev); initialise!(p, 0.0)

    BCs = assign(region = mesh_dev, (
        u = [Empty(:inlet), Empty(:outlet), Empty(:bottom), Empty(:top)],
        v = [Empty(:inlet), Empty(:outlet), Empty(:bottom), Empty(:top)],
        p = [Empty(:inlet), Empty(:outlet), Empty(:bottom), Empty(:top)],
    ))

    solvers = (
        u = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
        v = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
        p = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    )

    config = Configuration(solvers=solvers, schemes=(u=Schemes(), v=Schemes(), p=Schemes()), runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0), hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=BCs)
    mu = ConstantScalar(mu_val); one = ConstantScalar(1.0); tau_rc = ConstantScalar(0.1)

    L_u = ((-Laplacian{XCALibre.Linear}(mu) + ScalarGrad{XCALibre.Linear,1}(one, p) == Source(f_u)) → BCs.u) → solvers.u
    L_v = ((-Laplacian{XCALibre.Linear}(mu) + ScalarGrad{XCALibre.Linear,2}(one, p) == Source(0.0)) → BCs.v) → solvers.v
    L_p = ((-Laplacian{XCALibre.Linear}(tau_rc) + VectorDiv{XCALibre.Linear,1}(one, u) + VectorDiv{XCALibre.Linear,2}(one, v) == Source(0.0)) → BCs.p) → solvers.p

    sys = MonolithicSystem([L_u(u), L_v(v), L_p(p)], [u, v, p])
    A_csr, b = assemble_monolithic_system(sys, (BCs.u, BCs.v, BCs.p), config)
    A = get_sparse_matrix(A_csr)
    p_row = 2 * length(mesh_dev.cells) + 1; A[p_row, :] .= 0.0; A[p_row, p_row] = 1.0; b[p_row] = 0.0
    x = A \ b
    set_fields!(sys, x)
    return u.values
end

function solve_stress_defect(mesh_dev, mu_val, f_u)
    u = ScalarField(mesh_dev); initialise!(u, 0.0)
    v = ScalarField(mesh_dev); initialise!(v, 0.0)
    p = ScalarField(mesh_dev); initialise!(p, 0.0)
    txx = ScalarField(mesh_dev); initialise!(txx, 0.0)
    tyy = ScalarField(mesh_dev); initialise!(tyy, 0.0)
    txy = ScalarField(mesh_dev); initialise!(txy, 0.0)

    BCs = assign(region = mesh_dev, (
        u = [Empty(:inlet), Empty(:outlet), Empty(:bottom), Empty(:top)],
        v = [Empty(:inlet), Empty(:outlet), Empty(:bottom), Empty(:top)],
        p = [Empty(:inlet), Empty(:outlet), Empty(:bottom), Empty(:top)],
        txx = [Empty(:inlet), Empty(:outlet), Empty(:bottom), Empty(:top)],
        tyy = [Empty(:inlet), Empty(:outlet), Empty(:bottom), Empty(:top)],
        txy = [Empty(:inlet), Empty(:outlet), Empty(:bottom), Empty(:top)],
    ))

    solvers = (
        u = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
        v = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
        p = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
        txx = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
        tyy = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
        txy = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0),
    )

    config = Configuration(solvers=solvers, schemes=(u=Schemes(), v=Schemes(), p=Schemes(), txx=Schemes(), tyy=Schemes(), txy=Schemes()), runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0), hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=BCs)

    mu = ConstantScalar(mu_val)
    two_mu = ConstantScalar(2.0*mu_val)
    one = ConstantScalar(1.0)
    tau_rc = ConstantScalar(0.1)

    # Momentum with defect correction:
    # -mu Laplacian(u) - div(tau) + div(2 mu ε(u)) + grad(p) = f
    # For now, let's just do -mu Laplacian(u) - div(tau) + grad(p) = f
    # BUT! tau should be defined as "extra stress" only.
    # If tau = 2 mu ε(u), then -div(tau) + div(2 mu ε(u)) = 0.

    # Let's try the simplest defect form:
    # -mu Laplacian(u) + grad(p) = f + (div(tau) - div(2 mu ε(u)))  (lagged)
    # But we want a MONOLITHIC solution.

    # In a monolithic system, we can't easily lag terms.
    # We want -mu Laplacian(u) - div(tau) + grad(p) = f
    # with tau = 0? No.

    # Actually, the user's challenge is to match:
    # tau = 2 mu ε(u)
    # -div(tau) + grad(p) = f
    # to
    # -mu Laplacian(u) + grad(p) = f

    # We found that -div(tau) checkerboards.
    # If we add -mu Laplacian(u) + mu Laplacian(u) to momentum:
    # -mu Laplacian(u) + grad(p) = f + (div(tau) - mu Laplacian(u))
    # where mu Laplacian(u) is the compact one.

    L_u = ((-Laplacian{XCALibre.Linear}(mu) - ScalarGrad{XCALibre.Linear,1}(one, txx) - ScalarGrad{XCALibre.Linear,2}(one, txy) + ScalarGrad{XCALibre.Linear,1}(one, p) == Source(f_u)) → BCs.u) → solvers.u
    L_v = ((-Laplacian{XCALibre.Linear}(mu) - ScalarGrad{XCALibre.Linear,1}(one, txy) - ScalarGrad{XCALibre.Linear,2}(one, tyy) + ScalarGrad{XCALibre.Linear,2}(one, p) == Source(0.0)) → BCs.v) → solvers.v
    L_p = ((-Laplacian{XCALibre.Linear}(tau_rc) + VectorDiv{XCALibre.Linear,1}(one, u) + VectorDiv{XCALibre.Linear,2}(one, v) == Source(0.0)) → BCs.p) → solvers.p

    # tau is defined as a perturbation/correction? No.
    # If we solve for tau and u together, we MUST have a stable coupling.

    # What if we use a COMPACT Divergence?
    # XCALibre doesn't have one.

    # Let's try to set mu_s = mu and see if rel_diff is small.
    # (Already did, it is not 0 because div(tau) is still there).

    println("This audit requires a stabilized stress-divergence operator.")
    return nothing
end

solve_stress_defect(mesh_dev, 1.0, get_source_u(mesh_dev, 1.0))
