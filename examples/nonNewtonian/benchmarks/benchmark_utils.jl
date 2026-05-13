# =============================================================================
# Benchmark Utilities for Non-Newtonian Comparison Suite — XCALibre.jl
# =============================================================================

using XCALibre
using Accessors
using LinearAlgebra
using Printf
using Statistics
using SparseArrays

import XCALibre.ModelFramework: Operator, ScalarGrad, VectorDiv, Laplacian, Si, PDEOperator, Model, ModelEquation, ScalarModel, ScalarEquation
import XCALibre.Solve: update_fields!, assemble_monolithic_system

function get_sparse_matrix(A_csr)
    I_row = Vector{Int64}(undef, length(A_csr.nzval))
    for r in 1:(length(A_csr.rowptr)-1)
        for i in A_csr.rowptr[r]:(A_csr.rowptr[r+1]-1)
            I_row[i] = r
        end
    end
    return sparse(I_row, A_csr.colval, A_csr.nzval, size(A_csr)...)
end

function make_periodic_topology(mesh::Mesh2, patch1::Symbol, patch2::Symbol, translation::AbstractVector; tol=1e-5)
    return XCALibre.Mesh.construct_periodic_topology(mesh, patch1, patch2, translation; tol=tol)
end

function make_periodic_topology(mesh::Mesh3, patch1::Symbol, patch2::Symbol, translation::AbstractVector; tol=1e-5)
    return XCALibre.Mesh.construct_periodic_topology(mesh, patch1, patch2, translation; tol=tol)
end

function create_meqn(model, phi, bcs, setup)
    ModelEquation(ScalarModel(), model, ScalarEquation(phi, bcs), setup.solver, setup.preconditioner, setup)
end

function get_straight_mesh()
    grids_dir = pkgdir(XCALibre, "examples", "0_GRIDS")
    mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.025)
    return mesh, adapt(CPU(), mesh)
end

function get_bend_mesh()
    mesh_dir = "/Volumes/OpenFOAM/mixed_viscoelasticity/openfoam_cases/stokes3plus3_bend/viscoelasticChannelBend_stokes_compressibleSolid_KelvinVoigt/constant/polyMesh"
    mesh = FOAM3D_mesh(mesh_dir, scale=1.0)
    return mesh, adapt(CPU(), mesh)
end

function report_results(name, res, u, p, tau=nothing)
    u_max = maximum(abs.(u.values))
    p_max = maximum(abs.(p.values))
    if tau !== nothing
        tau_max = maximum(abs.(tau.values))
        @printf("[%s] Residual: %.2e | max|u|: %.4e | max|p|: %.4e | max|tau|: %.4e\n", name, res, u_max, p_max, tau_max)
    else
        @printf("[%s] Residual: %.2e | max|u|: %.4e | max|p|: %.4e\n", name, res, u_max, p_max)
    end
end
