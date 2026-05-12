# =============================================================================
# Corrected Non-Newtonian Comparison Suite (V16) — XCALibre.jl
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
    Sf_I = face.normal[I] * face.area * face.nsign[fID] # Outward normal component
    coeff = term.sign * term.flux[fID] * Sf_I
    return 0.0, coeff * bc.value
end

# Zerogradient for ScalarGrad: ac = - (flux * n_I * area)
@inline function (bc::Zerogradient)(term::Operator{F,P,I_OP,ScalarGrad{T,I}}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I_OP,T,I}
    Sf_I = face.normal[I] * face.area * face.nsign[fID]
    coeff = term.sign * term.flux[fID] * Sf_I
    return -coeff, 0.0
end

# Dirichlet for VectorDiv: source = flux * n_J * area * bc.value
@inline function (bc::Dirichlet)(term::Operator{F,P,I_OP,VectorDiv{T,J}}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I_OP,T,J}
    Sf_J = face.normal[J] * face.area * face.nsign[fID]
    coeff = term.sign * term.flux[fID] * Sf_J
    return 0.0, coeff * bc.value
end

# Zerogradient for VectorDiv: ac = - (flux * n_J * area)
@inline function (bc::Zerogradient)(term::Operator{F,P,I_OP,VectorDiv{T,J}}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I_OP,T,J}
    Sf_J = face.normal[J] * face.area * face.nsign[fID]
    coeff = term.sign * term.flux[fID] * Sf_J
    return -coeff, 0.0
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

# ── 1. Benchmark Definitions ──────────────────────────────────────────────────

function run_practical_stokes(mesh_dev)
    @info "Running Practical Branch: Stokes"
    u = ScalarField(mesh_dev); v = ScalarField(mesh_dev); p = ScalarField(mesh_dev)
    initialise!.([u, v, p], 0.0)
    
    u_bcs = [Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0), Zerogradient(:inlet), Zerogradient(:outlet)]
    v_bcs = [Dirichlet(:top, 0.0), Dirichlet(:bottom, 0.0), Zerogradient(:inlet), Zerogradient(:outlet)]
    p_bcs = [Zerogradient(:top), Zerogradient(:bottom), Zerogradient(:inlet), Zerogradient(:outlet)]
    
    solvers = (u=SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0), 
               v=SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0), 
               p=SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-12, relax=1.0))
    config = Configuration(solvers=solvers, schemes=(u=Schemes(), v=Schemes(), p=Schemes()),
                           runtime=Runtime(iterations=1, write_interval=-1, time_step=0.01), 
                           hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=(u=u_bcs, v=v_bcs, p=p_bcs))
    
    mu = ConstantScalar(1.0); one = ConstantScalar(1.0); trc = ConstantScalar(0.01)
    
    # -μ∇²u + ∂p/∂x = f_x
    L_u = ((- Laplacian{XCALibre.Linear}(mu) + ScalarGrad{XCALibre.Linear,1}(one, p) == Source(1.0)) → u_bcs) → solvers.u
    L_v = ((- Laplacian{XCALibre.Linear}(mu) + ScalarGrad{XCALibre.Linear,2}(one, p) == Source(0.0)) → v_bcs) → solvers.v
    L_p = ((- Laplacian{XCALibre.Linear}(trc) + VectorDiv{XCALibre.Linear,1}(one, u) + VectorDiv{XCALibre.Linear,2}(one, v) == Source(0.0)) → p_bcs) → solvers.p
    
    sys = MonolithicSystem([L_u(u), L_v(v), L_p(p)], [u, v, p])
    
    A_csr, b_mono = assemble_monolithic_system(sys, (u_bcs, v_bcs, p_bcs), config)
    
    # Exact Pinning of pressure
    n_cells = length(mesh_dev.cells)
    mono_row = 2 * n_cells + 1 
    A_julia = get_sparse_matrix(A_csr)
    A_julia[mono_row, :] .= 0.0
    A_julia[mono_row, mono_row] = 1.0
    b_mono[mono_row] = 0.0
    
    x = A_julia \ b_mono
    XCALibre.Solve.set_fields!(sys, x)
    
    @info "Stokes Practical Result" max_u=maximum(abs.(u.values))
end

# ── 2. Run Benchmarks ─────────────────────────────────────────────────────────

@info "Initializing Mesh..."
grids_dir = pkgdir(XCALibre, "examples", "0_GRIDS")
mesh      = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.025)
mesh_dev  = adapt(CPU(), mesh)

run_practical_stokes(mesh_dev)

@info "Comparison Suite Finished."
