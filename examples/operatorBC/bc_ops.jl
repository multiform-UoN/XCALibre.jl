# =============================================================================
# BC RESIDUAL INTERFACE: bc_residual / bc_jvp_coeff on existing XCALibre types
# =============================================================================
#
# PURPOSE
# -------
# Extend XCALibre's semantic BC types (Dirichlet, Zerogradient, Neumann, Robin)
# with two scalar @inline functions:
#
#   bc_residual(bc, phi_P, face, gamma_f) → F
#     The boundary face's contribution to r[cID] = R(phi)[cID].
#     Convention: r[cID] += bc_residual(...), consistent with scatter-add kernels.
#
#   bc_jvp_coeff(bc, phi_P, face, gamma_f) → F
#     ∂(bc_residual)/∂phi_P — diagonal Jacobian coefficient.
#     (J·v)[cID] += bc_jvp_coeff(...) * v[cID].
#
# These are the SAME semantic BC objects used in the assembled FV path.
# The choice of computational realization is made by the CALLER:
#
#   Assembled (primary path, @define_boundary):
#     Dirichlet → (ap, bp) → CSR row modification
#
#   Matrix-free residual (complementary, via these methods):
#     bc_residual(bc::Dirichlet, ...) → scatter-add to r[cID]
#
#   Assembled-via-residual interface (A3, hypothetical):
#     bc_jvp_coeff → write to A[cID,cID];  bc_residual(bc,0,...) → negate for b[cID]
#
# No separate "Op" wrapper types are needed. Use existing BCs directly:
#   bc_tuple = BCs.C   # or any Tuple of XCALibre BC types
#   residual_ops!(r, phi, gamma, source, bc_tuple, mesh, backend, workgroup)
#
# GPU NOTE
# --------
# On CPU (current), no extra Adapt work is needed.
# For GPU device execution, XCALibre's BC types would need @adapt_structure
# in src/Discretise/ (a small, self-contained addition — not done here).
#
# GEOMETRY CONVENTION
# --------------------
# D_f computed via the over-relaxed formula (Ef projection):
#   Sf = area * normal;  Ef = ((Sf⋅Sf)/(Sf⋅e)) * e;  D_f = gamma * norm(Ef) / delta
# Reduces to gamma * area / delta on orthogonal meshes.
# Consistent with @define_boundary Laplacian{Linear}.

using XCALibre
using KernelAbstractions
using Atomix
using StaticArrays
using LinearAlgebra

# =============================================================================
# GEOMETRY HELPER — shared by all BC types
# =============================================================================

@inline function _face_Df(face, gamma_f::F) where F
    (; area, delta, normal, e) = face
    Sf = area * normal
    Ef = ((Sf ⋅ Sf) / (Sf ⋅ e)) * e
    return gamma_f * norm(Ef) / delta
end

# =============================================================================
# bc_residual / bc_jvp_coeff — methods on existing XCALibre BC types
# =============================================================================

# Dirichlet: phi = value at face
#   residual = D_f * (phi_P - value)   (≡ ap*phi_P - bp from assembled path)
#   jvp_coeff = D_f                    (constant, phi-independent)
@inline function bc_residual(bc::Dirichlet, phi_P::F, face, gamma_f::F) where F
    return _face_Df(face, gamma_f) * (phi_P - F(bc.value))
end

@inline function bc_jvp_coeff(bc::Dirichlet, phi_P::F, face, gamma_f::F) where F
    return _face_Df(face, gamma_f)
end

# Zerogradient: ∂phi/∂n = 0 at face — no flux, no residual contribution
@inline bc_residual(::Zerogradient, phi_P::F, face, gamma_f::F) where F = zero(F)
@inline bc_jvp_coeff(::Zerogradient, phi_P::F, face, gamma_f::F) where F = zero(F)

# Neumann: prescribed normal flux gamma * dphi/dn = flux
#   residual = -flux * area  (constant; phi-independent)
#   jvp_coeff = 0
@inline function bc_residual(bc::Neumann, phi_P::F, face, gamma_f::F) where F
    return -F(bc.value) * F(face.area)
end

@inline bc_jvp_coeff(::Neumann, phi_P::F, face, gamma_f::F) where F = zero(F)

# Robin: a*phi + b*(gamma * dphi/dn) = c
#   coeff = gamma * area / (a*delta + b)
#   residual  = coeff * (a * phi_P - c)
#   jvp_coeff = coeff * a
@inline function bc_residual(bc::Robin, phi_P::F, face, gamma_f::F) where F
    (; area, delta) = face
    a, b, val = F(bc.value.a), F(bc.value.b), F(bc.value.value)
    coeff = gamma_f * area / (a * delta + b)
    return coeff * (a * phi_P - val)
end

@inline function bc_jvp_coeff(bc::Robin, phi_P::F, face, gamma_f::F) where F
    (; area, delta) = face
    a, b = F(bc.value.a), F(bc.value.b)
    return (gamma_f * area / (a * delta + b)) * a
end

# =============================================================================
# GENERIC GPU KERNELS — one per operation, generic over BC type T
# =============================================================================
# Type dispatch on bc::T is resolved at compile time per BC type.
# Each unique T in the bc_tuple gets its own compiled kernel specialisation.

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

# =============================================================================
# HIGH-LEVEL RESIDUAL AND JVP FUNCTIONS
# =============================================================================

function residual_ops!(r, phi_vals, gamma_vals, source, bcs::Tuple, mesh, backend, workgroup)
    fill!(r, 0)
    laplacian_residual!(r, phi_vals, gamma_vals, source, mesh, backend, workgroup)
    add_bc_residuals!(r, phi_vals, gamma_vals, bcs, mesh, backend, workgroup)
end

function jvp_ops!(Jv, v, phi_vals, gamma_vals, source, bcs::Tuple, mesh, backend, workgroup;
                  ε = sqrt(eps(eltype(phi_vals))))
    n  = length(phi_vals)
    r0 = zeros(eltype(phi_vals), n)
    r1 = zeros(eltype(phi_vals), n)
    residual_ops!(r0, phi_vals,           gamma_vals, source, bcs, mesh, backend, workgroup)
    residual_ops!(r1, phi_vals .+ ε .* v, gamma_vals, source, bcs, mesh, backend, workgroup)
    @. Jv = (r1 - r0) / ε
end
