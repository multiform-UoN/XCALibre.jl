export →, PDEOperator

const TemplateTerm = Union{OperatorTemplate,NonlinearOperatorTemplate}

@inline _negate_template(t::OperatorTemplate) = @set t.sign = -t.sign
@inline _negate_template(t::NonlinearOperatorTemplate) = @set t.op.sign = -t.op.sign
# Pre-bound cross-field Operator: negate sign in-place
@inline _negate_template(t::Operator) = @set t.sign = -t.sign

Base.:+(a::TemplateTerm, b::TemplateTerm) = PDEOperator((a, b), (), (), nothing)
Base.:+(a::PDEOperator, b::TemplateTerm) = PDEOperator((a.templates..., b), a.sources, a.BCs, a.setup)
Base.:+(a::PDEOperator, b::Src) = PDEOperator(a.templates, (a.sources..., b), a.BCs, a.setup)

# Accept pre-bound Operator as a "frozen" term in a PDEOperator expression.
# These are cross-field coupling terms (ScalarGrad, VectorDiv, GradDiv off-diagonal)
# whose phi is already set and must not be rebound when the operator is applied.
Base.:+(a::PDEOperator, b::Operator) = PDEOperator((a.templates..., b), a.sources, a.BCs, a.setup)
Base.:-(a::PDEOperator, b::Operator) =
    PDEOperator((a.templates..., _negate_template(b)), a.sources, a.BCs, a.setup)

Base.:-(a::TemplateTerm) = PDEOperator((_negate_template(a),), (), (), nothing)

Base.:-(a::TemplateTerm, b::TemplateTerm) = PDEOperator((a, _negate_template(b)), (), (), nothing)

Base.:-(a::PDEOperator, b::TemplateTerm) =
    PDEOperator((a.templates..., _negate_template(b)), a.sources, a.BCs, a.setup)

Base.:(==)(L::PDEOperator, s::Src) = PDEOperator(L.templates, (s,), L.BCs, L.setup)
Base.:(==)(L::PDEOperator, s::Vector{<:Src}) = PDEOperator(L.templates, Tuple(s), L.BCs, L.setup)

# PDEOperator scalar multiplication — scales all fluxes and sources by s
@inline _scale_template(t::OperatorTemplate, s) = @set t.flux = ScaledFlux(t.flux, s)
@inline _scale_template(t::NonlinearOperatorTemplate, s) = @set t.op.flux = ScaledFlux(t.op.flux, s)
@inline _scale_template(t::Operator, s) = @set t.flux = ScaledFlux(t.flux, s)
@inline _scale_source(src::Src, s) = @set src.field = ScaledFlux(src.field, s)

Base.:*(L::PDEOperator, s::Number) = PDEOperator(
    Tuple(map(t -> _scale_template(t, s), L.templates)),
    Tuple(map(src -> _scale_source(src, s), L.sources)),
    L.BCs, L.setup
)
Base.:*(s::Number, L::PDEOperator) = L * s

Base.:+(a::Operator, b::Operator) = [a, b]
Base.:+(a::Vector{<:AbstractOperator}, b::Operator) = [a..., b]
Base.:+(a::Vector{<:AbstractOperator}, b::NonlinearOperator) = [a..., b]
Base.:+(a::Operator, b::NonlinearOperator) = [a, b]
Base.:+(a::NonlinearOperator, b::Operator) = [a, b]
Base.:+(a::NonlinearOperator, b::NonlinearOperator) = [a, b]

Base.:-(a::Operator) = begin
    @reset a.sign = -1
    [a]
end
Base.:-(a::NonlinearOperator) = begin
    @reset a.op.sign = -1
    [a]
end
Base.:-(a::Operator, b::Operator) = begin
    @reset b.sign = -1
    [a, b]
end
Base.:-(a::Operator, b::NonlinearOperator) = begin
    @reset b.op.sign = -1
    [a, b]
end
Base.:-(a::NonlinearOperator, b::Operator) = begin
    @reset b.sign = -1
    [a, b]
end
Base.:-(a::NonlinearOperator, b::NonlinearOperator) = begin
    @reset b.op.sign = -1
    [a, b]
end
Base.:-(a::Vector{<:AbstractOperator}, b::Operator) = begin
    @reset b.sign = -1
    [a..., b]
end
Base.:-(a::Vector{<:AbstractOperator}, b::NonlinearOperator) = begin
    @reset b.op.sign = -1
    [a..., b]
end

# Source operations

Base.:+(a::Src, b::Src) = [a, b]
Base.:+(a::Vector{<:Src}, b::Src) = [a..., b]

Base.:-(a::Src) = begin
    @reset a.sign = -1
    [a]
end

Base.:-(a::Src, b::Src) = begin
    @reset b.sign = -1
    [a, b]
end
Base.:-(a::Vector{<:Src}, b::Src) = begin
    @reset b.sign = -1
    [a..., b]
end

# Equality operation for model wrapper

Base.:(==)(a::Operator, b::Src) = begin
    Model{1,1}((a,),(b,))
end
Base.:(==)(a::NonlinearOperator, b::Src) = begin
    Model{1,1}((a,),(b,))
end

Base.:(==)(a::Vector{<:AbstractOperator}, b::Src) = begin
    Model{length(a),1}((a...,),(b,))
end

Base.:(==)(a::Operator, b::Vector{<:Src}) = begin
    Model{1,length(b)}((a...,),(b...,))
end
Base.:(==)(a::NonlinearOperator, b::Vector{<:Src}) = begin
    Model{1,length(b)}((a...,),(b...,))
end

Base.:(==)(a::Vector{<:AbstractOperator}, b::Vector{<:Src}) = begin
    Model{length(a), length(b)}((a...,),(b...,))
end

(→)(model::Model{TN,SN,T,S}, eqn::AbstractEquation) where {TN,SN,T,S}= begin
    # To-do: Add runtime check to ensure both sides are consistent (for now document)
    if S.parameters[1].parameters[1] <: AbstractScalarField
        ModelEquation(ScalarModel(), model, eqn, nothing, nothing, nothing)
    elseif S.parameters[1].parameters[1] <: AbstractVectorField
        ModelEquation(VectorModel(), model, eqn, nothing, nothing, nothing)
    end
end
