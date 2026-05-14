# =============================================================================
# BC RESIDUAL EXTENSION INTERFACE
# =============================================================================
#
# PURPOSE
# -------
# Provide kernel infrastructure for extending the matrix-free residual path
# to GENUINELY NONLINEAR or CUSTOM BC types — cases where @define_boundary
# is insufficient or unavailable.
#
# DESIGN RATIONALE
# -----------------
# Standard linear BCs (Dirichlet, Zerogradient, Neumann, Robin) are handled
# canonically by @define_boundary(BC, Operator) → (ap, bp) pairs.
# Their residual contribution is generically derivable as:
#
#   r[cID] += ap * phi_P - bp
#   Jv[cID] += ap * v[cID]
#
# No separate bc_residual/bc_jvp_coeff methods are needed for standard BCs.
# Defining them here would duplicate (ap, bp) algebra, introduce a second
# source of truth, and risk formula divergence on non-orthogonal meshes.
#
# THIS INTERFACE IS FOR:
#   NonLinearRobin  — phi-dependent flux; @define_boundary errors intentionally
#   Custom coupled  — cross-field constraints (traction-pressure, viscoelastic)
#   Integral BCs    — global conservation, periodic-like constraints
#   Research BCs    — any new type not yet in the @define_boundary registry
#
# HOW TO USE
# ----------
# 1. Define bc_residual for your custom BC type:
#
#      @inline function bc_residual(bc::MyBC, phi_P::F, face, gamma_f::F) where F
#          # scalar residual contribution; convention: r[cID] += bc_residual(...)
#      end
#      @inline function bc_jvp_coeff(bc::MyBC, phi_P::F, face, gamma_f::F) where F
#          # ∂(bc_residual)/∂phi_P; convention: Jv[cID] += bc_jvp_coeff(...)*v[cID]
#      end
#
# 2. Dispatch over a BC tuple (mixed standard + custom is fine — only types that
#    have bc_residual defined will be dispatched into these kernels):
#
#      add_bc_residuals!(r, phi_vals, gamma_vals, bcs, mesh, backend, workgroup)
#      add_bc_jvps!(Jv, v, phi_vals, gamma_vals, bcs, mesh, backend, workgroup)
#
# NOTE: Standard BCs (Dirichlet, Zerogradient, etc.) deliberately do NOT have
# bc_residual methods here. Use @define_boundary for those.
#
# GPU COMPATIBILITY
# -----------------
# bc_residual_kernel! and bc_jvp_kernel! use @kernel + Atomix.@atomic.
# Dispatch on bc::T is resolved at compile time per BC type — zero runtime
# polymorphism in the hot loop.
# Custom BC types need Adapt.@adapt_structure for GPU device execution.

using XCALibre
using KernelAbstractions
using Atomix
using StaticArrays
using LinearAlgebra
using ForwardDiff

# =============================================================================
# NonLinearRobin — primary use case for the extension interface
# =============================================================================
#
# NonLinearRobin represents:  gamma * ∂phi/∂n = f(phi)   (nonlinear Neumann)
#
# @define_boundary NonLinearRobin Laplacian{Linear} raises an intentional error:
# the BC must be linearized (via linearize_bcs) before the assembled path.
# For the MATRIX-FREE path (JFNK inner loop, GPU residual evaluation), the
# nonlinear residual is evaluated directly here without linearization.
#
# Residual: the BC prescribes the outward flux as f(phi_P).
# In the FV residual convention (r[cID] = A*phi - b):
#   ap = 0  (no implicit dependence after removing the assembly step)
#   bp = gamma_f * area * f(phi_P)   (nonlinear prescribed flux)
#   r_bc[cID] = ap*phi_P - bp = -gamma_f * area * f(phi_P)
#
# Consistency: for constant f(phi)=c, bc_residual(NonLinearRobin) reduces to
# @define_boundary Neumann with value=c (gives bp = J*area*c, r = -J*area*c).
#
# JVP coefficient: ∂(bc_residual)/∂phi_P = -gamma_f * area * f'(phi_P)
# ForwardDiff.derivative is allocation-free on scalars and device-safe on CPU.
# For GPU, provide bc.value with a NonlinearMap(f, df) and use df directly
# instead of calling ForwardDiff.

@inline function bc_residual(bc::NonLinearRobin, phi_P::F, face, gamma_f::F) where F
    f0 = F(bc.value.flux_func(phi_P))
    return -(gamma_f * F(face.area)) * f0
end

@inline function bc_jvp_coeff(bc::NonLinearRobin, phi_P::F, face, gamma_f::F) where F
    df0 = F(ForwardDiff.derivative(bc.value.flux_func, phi_P))
    return -(gamma_f * F(face.area)) * df0
end

# =============================================================================
# GENERIC GPU KERNELS
# =============================================================================
# Type dispatch on bc::T is resolved at compile time — each unique T in the
# bc_tuple gets its own compiled specialisation.
# Requires: bc_residual(bc::T, ...) and bc_jvp_coeff(bc::T, ...) to be defined.

@kernel function bc_residual_kernel!(
    r::AbstractArray{F},
    phi::AbstractArray{F},
    gamma::AbstractArray{F},
    faces,
    cells,
    boundary_cellsID,
    bc::T,
    patch_start::Int,
    patch_stop::Int
) where {F, T}
    k = @index(Global)
    fID = k + patch_start - 1
    @inbounds begin
        if fID <= patch_stop
            cellID  = boundary_cellsID[fID]
            face    = faces[fID]
            gamma_f = gamma[fID]
            phi_P   = phi[cellID]
            Atomix.@atomic r[cellID] += bc_residual(bc, phi_P, face, gamma_f)
        end
    end
end

@kernel function bc_jvp_kernel!(
    Jv::AbstractArray{F},
    v::AbstractArray{F},
    phi::AbstractArray{F},
    gamma::AbstractArray{F},
    faces,
    cells,
    boundary_cellsID,
    bc::T,
    patch_start::Int,
    patch_stop::Int
) where {F, T}
    k = @index(Global)
    fID = k + patch_start - 1
    @inbounds begin
        if fID <= patch_stop
            cellID  = boundary_cellsID[fID]
            face    = faces[fID]
            gamma_f = gamma[fID]
            phi_P   = phi[cellID]
            coeff   = bc_jvp_coeff(bc, phi_P, face, gamma_f)
            Atomix.@atomic Jv[cellID] += coeff * v[cellID]
        end
    end
end

# =============================================================================
# DISPATCHERS — iterate over typed BC tuple (compile-time unrolled)
# =============================================================================

function add_bc_residuals!(r, phi_vals, gamma_vals, bcs::Tuple, mesh, backend, workgroup)
    (; faces, cells, boundary_cellsID) = mesh
    for bc in bcs
        ps = bc.IDs_range.start;  pe = bc.IDs_range.stop
        ndrange = pe - ps + 1;    ndrange == 0 && continue
        k! = bc_residual_kernel!(_setup(backend, workgroup, ndrange)...)
        k!(r, phi_vals, gamma_vals, faces, cells, boundary_cellsID, bc, ps, pe)
        KernelAbstractions.synchronize(backend)
    end
end

function add_bc_jvps!(Jv, v, phi_vals, gamma_vals, bcs::Tuple, mesh, backend, workgroup)
    (; faces, cells, boundary_cellsID) = mesh
    for bc in bcs
        ps = bc.IDs_range.start;  pe = bc.IDs_range.stop
        ndrange = pe - ps + 1;    ndrange == 0 && continue
        k! = bc_jvp_kernel!(_setup(backend, workgroup, ndrange)...)
        k!(Jv, v, phi_vals, gamma_vals, faces, cells, boundary_cellsID, bc, ps, pe)
        KernelAbstractions.synchronize(backend)
    end
end
