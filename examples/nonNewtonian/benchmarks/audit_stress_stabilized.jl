#!/usr/bin/env julia

# XCALibre Stress-Operator Audit (Stabilized)
#
# Investigates how much implicit Laplacian stabilization (mu_s) is needed
# to recover a sensible result from the stress-coupled formulation.
#
# Goal:
#   -mu_s Δu - div(τ) + ∇p = f
#   τ = 2 mu_p ε(u)
#   mu_s + mu_p = 1.0

using XCALibre
include(joinpath(pkgdir(XCALibre), "..", "mixed_viscoelasticity", "xcalibre", "benchmarks", "benchmark_utils.jl"))

using Printf
using SparseArrays
using LinearAlgebra

# Use quad40.unv (periodic)
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
    u = ScalarField(mesh_dev); initialise!(u, 0.0); v = ScalarField(mesh_dev); initialise!(v, 0.0); p = ScalarField(mesh_dev); initialise!(p, 0.0)
    BCs = assign(region = mesh_dev, (u = [Empty(:inlet), Empty(:outlet), Empty(:bottom), Empty(:top)], v = [Empty(:inlet), Empty(:outlet), Empty(:bottom), Empty(:top)], p = [Empty(:inlet), Empty(:outlet), Empty(:bottom), Empty(:top)],))
    setup = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0)
    config = Configuration(solvers=(u=setup, v=setup, p=setup), schemes=(u=Schemes(), v=setup, p=setup), runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0), hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=BCs)
    mu = ConstantScalar(mu_val); one = ConstantScalar(1.0); tau_rc = ConstantScalar(0.1)
    L_u = ((-Laplacian{XCALibre.Linear}(mu) + ScalarGrad{XCALibre.Linear,1}(one, p) == Source(f_u)) → BCs.u) → setup
    L_v = ((-Laplacian{XCALibre.Linear}(mu) + ScalarGrad{XCALibre.Linear,2}(one, p) == Source(0.0)) → BCs.v) → setup
    L_p = ((-Laplacian{XCALibre.Linear}(tau_rc) + VectorDiv{XCALibre.Linear,1}(one, u) + VectorDiv{XCALibre.Linear,2}(one, v) == Source(0.0)) → BCs.p) → setup
    sys = MonolithicSystem([L_u(u), L_v(v), L_p(p)], [u, v, p])
    A_csr, b = assemble_monolithic_system(sys, (BCs.u, BCs.v, BCs.p), config)
    A = get_sparse_matrix(A_csr); p_row = 2 * length(mesh_dev.cells) + 1; A[p_row, :] .= 0.0; A[p_row, p_row] = 1.0; b[p_row] = 0.0
    x = A \ b; set_fields!(sys, x)
    return u.values
end

function solve_stabilized(mesh_dev, mu_s_val, mu_p_val, f_u)
    u = ScalarField(mesh_dev); initialise!(u, 0.0); v = ScalarField(mesh_dev); initialise!(v, 0.0); p = ScalarField(mesh_dev); initialise!(p, 0.0)
    txx = ScalarField(mesh_dev); initialise!(txx, 0.0); tyy = ScalarField(mesh_dev); initialise!(tyy, 0.0); txy = ScalarField(mesh_dev); initialise!(txy, 0.0)
    BCs = assign(region = mesh_dev, (u = [Empty(:inlet), Empty(:outlet), Empty(:bottom), Empty(:top)], v = [Empty(:inlet), Empty(:outlet), Empty(:bottom), Empty(:top)], p = [Empty(:inlet), Empty(:outlet), Empty(:bottom), Empty(:top)], txx = [Empty(:inlet), Empty(:outlet), Empty(:bottom), Empty(:top)], tyy = [Empty(:inlet), Empty(:outlet), Empty(:bottom), Empty(:top)], txy = [Empty(:inlet), Empty(:outlet), Empty(:bottom), Empty(:top)],))
    setup = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-10, relax=1.0)
    config = Configuration(solvers=(u=setup, v=setup, p=setup, txx=setup, tyy=setup, txy=setup), schemes=(u=Schemes(), v=Schemes(), p=Schemes(), txx=Schemes(), tyy=Schemes(), txy=Schemes()), runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0), hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=BCs)
    mu_s = ConstantScalar(mu_s_val); mu_p = ConstantScalar(mu_p_val); two_mu_p = ConstantScalar(2.0*mu_p_val); one = ConstantScalar(1.0); tau_rc = ConstantScalar(0.1)
    L_u = ((-Laplacian{XCALibre.Linear}(mu_s) - ScalarGrad{XCALibre.Linear,1}(one, txx) - ScalarGrad{XCALibre.Linear,2}(one, txy) + ScalarGrad{XCALibre.Linear,1}(one, p) == Source(f_u)) → BCs.u) → setup
    L_v = ((-Laplacian{XCALibre.Linear}(mu_s) - ScalarGrad{XCALibre.Linear,1}(one, txy) - ScalarGrad{XCALibre.Linear,2}(one, tyy) + ScalarGrad{XCALibre.Linear,2}(one, p) == Source(0.0)) → BCs.v) → setup
    L_p = ((-Laplacian{XCALibre.Linear}(tau_rc) + VectorDiv{XCALibre.Linear,1}(one, u) + VectorDiv{XCALibre.Linear,2}(one, v) == Source(0.0)) → BCs.p) → setup
    L_txx = ((Si(one) - ScalarGrad{XCALibre.Linear,1}(two_mu_p, u) == Source(0.0)) → BCs.txx) → setup
    L_tyy = ((Si(one) - ScalarGrad{XCALibre.Linear,2}(two_mu_p, v) == Source(0.0)) → BCs.tyy) → setup
    L_txy = ((Si(one) - ScalarGrad{XCALibre.Linear,2}(mu_p, u) - ScalarGrad{XCALibre.Linear,1}(mu_p, v) == Source(0.0)) → BCs.txy) → setup
    sys = MonolithicSystem([L_u(u), L_v(v), L_p(p), L_txx(txx), L_tyy(tyy), L_txy(txy)], [u, v, p, txx, tyy, txy])
    A_csr, b = assemble_monolithic_system(sys, (BCs.u, BCs.v, BCs.p, BCs.txx, BCs.tyy, BCs.txy), config)
    A = get_sparse_matrix(A_csr); p_row = 2 * length(mesh_dev.cells) + 1; A[p_row, :] .= 0.0; A[p_row, p_row] = 1.0; b[p_row] = 0.0
    x = A \ b; set_fields!(sys, x)
    return u.values
end

mu_total = 1.0
f_u = get_source_u(mesh_dev, mu_total)
u_direct = solve_direct(mesh_dev, mu_total, f_u)

println("mu_s\tmu_p\tmax|u|\trel_diff")
for mu_s_val in [1.0, 0.9, 0.5, 0.1, 0.01, 0.0]
    mu_p_val = mu_total - mu_s_val
    u_stab = solve_stabilized(mesh_dev, mu_s_val, mu_p_val, f_u)
    rel_diff = norm(u_stab - u_direct) / norm(u_direct)
    @printf("%.4f\t%.4f\t%.4e\t%.4e\n", mu_s_val, mu_p_val, maximum(abs.(u_stab)), rel_diff)
end
