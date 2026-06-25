export inner_product!
export double_inner_product!
export magnitude!, magnitude2!, square!

inner_product!(S::F, ∇1::Grad, ∇2::Grad, config) where F<:ScalarField = begin
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(S)
    kernel! = _inner_product!(_setup(backend, workgroup, ndrange)...)
    kernel!(S, ∇1, ∇2)
    # KernelAbstractions.synchronize(backend)
end

@kernel function _inner_product!(S::F, ∇1::Grad, ∇2::Grad) where F<:ScalarField
    i = @index(Global)
    @inbounds S[i] = ∇1[i]⋅∇2[i]
end

double_inner_product!(s, t0::AbstractTensorField, t2, config) = begin
    xcal_foreach(s, config) do i
        t1 = 2*t0[i] - (2/3)*t0[i]*I
        s[i] = tr(t1 * t2[i])
    end
end

# function magnitude!(magS::ScalarField, S::AbstractVectorField, config)
function magnitude!(magS::ScalarField, S, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(magS)
    kernel! = _magnitude!(_setup(backend, workgroup, ndrange)...)
    kernel!(magS, S)
    # KernelAbstractions.synchronize(backend)
end

# @kernel function _magnitude!(magS::ScalarField, S::AbstractVectorField)
@kernel function _magnitude!(magS::AbstractScalarField, S)
    i = @index(Global)
    @inbounds magS[i] = norm(S[i])
end

function magnitude2!(
    magS, S, config; scale_factor=1.0
    )
    (; hardware) = config
    (; backend, workgroup) = hardware

    scale = eltype(magS)(scale_factor)
    ndrange = length(magS)
    kernel! = _magnitude2!(_setup(backend, workgroup, ndrange)...)
    kernel!(magS, S, scale)
    # KernelAbstractions.synchronize(backend)
end

@kernel function _magnitude2!(
    magS::ScalarField, S::AbstractTensorField, scale_factor
    )
    i = @index(Global)
    @inbounds begin
        Sjk = S[i]
        magS[i] = (Sjk⋅Sjk) * scale_factor
    end
end

@kernel function _magnitude2!(
    magS::AbstractScalarField, S::AbstractVectorField, scale_factor
    )
    i = @index(Global)
    @inbounds begin
        Si = S[i]
        magS[i] = (Si⋅Si) * scale_factor
    end
end

function square!(psi2, psi, config; scale_factor=1.0)
    (; hardware) = config
    (; backend, workgroup) = hardware

    scale = eltype(psi2)(scale_factor)
    ndrange = length(psi2)
    kernel! = _square!(_setup(backend, workgroup, ndrange)...)
    kernel!(psi2, psi, scale)
    nothing
end

@kernel function _square!(
    psi2::AbstractTensorField, psi::AbstractVectorField, scale_factor
    )
    i = @index(Global)

    @inbounds begin
        vi = psi[i]
        psi2[i] = vi*vi'
    end
end
