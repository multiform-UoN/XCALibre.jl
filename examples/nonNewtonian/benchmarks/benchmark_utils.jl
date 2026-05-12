# =============================================================================
# Benchmark Utilities for Non-Newtonian Comparison Suite — XCALibre.jl
# =============================================================================

using XCALibre
using Accessors
using LinearAlgebra
using Printf
using Statistics
using SparseArrays

# ── Extension: Define CORRECT BC methods for ScalarGrad/VectorDiv ─────────────
import XCALibre.Discretise: @define_boundary, AbstractDirichlet, AbstractNeumann, AbstractBoundary
import XCALibre.ModelFramework: Operator, ScalarGrad, VectorDiv, Laplacian, Si, PDEOperator, Model, ModelEquation, ScalarModel, ScalarEquation
import XCALibre.Solve: update_fields!, assemble_monolithic_system

# Dirichlet for ScalarGrad: source = flux * n_I * area * bc.value
@inline function (bc::Dirichlet)(term::Operator{F,P,I_OP,ScalarGrad{T,I}}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I_OP,T,I}
    e_I = face.e[I]; ap = term.sign * term.flux[fID] * e_I * face.area / face.delta
    return -ap, -ap*bc.value
end

# Zerogradient for ScalarGrad: ac = - (flux * n_I * area)
@inline function (bc::Zerogradient)(term::Operator{F,P,I_OP,ScalarGrad{T,I}}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I_OP,T,I}
    return 0.0, 0.0
end

# Dirichlet for VectorDiv: source = flux * n_J * area * bc.value
@inline function (bc::Dirichlet)(term::Operator{F,P,I_OP,VectorDiv{T,J}}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I_OP,T,J}
    e_J = face.e[J]; ap = term.sign * term.flux[fID] * e_J * face.area / face.delta
    return -ap, -ap*bc.value
end

# Zerogradient for VectorDiv: ac = - (flux * n_J * area)
@inline function (bc::Zerogradient)(term::Operator{F,P,I_OP,VectorDiv{T,J}}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I_OP,T,J}
    return 0.0, 0.0
end

function get_sparse_matrix(A_csr)
    I_row = Vector{Int64}(undef, length(A_csr.nzval))
    for r in 1:(length(A_csr.rowptr)-1)
        for i in A_csr.rowptr[r]:(A_csr.rowptr[r+1]-1)
            I_row[i] = r
        end
    end
    return sparse(I_row, A_csr.colval, A_csr.nzval, size(A_csr)...)
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
