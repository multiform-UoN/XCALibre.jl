export →

# sign of a term, also when it wraps an Operator (NonlinearOperator)
_with_sign(t::Operator, s) = @set t.sign = s
_with_sign(t, s) = @set t.op.sign = s

Base.:+(a::AbstractOperator, b::AbstractOperator) = [a, b]
Base.:+(a::Vector{<:AbstractOperator}, b::AbstractOperator) = [a..., b]

Base.:-(a::AbstractOperator) = begin
    a = _with_sign(a, -1)
    [a]
end
Base.:-(a::AbstractOperator, b::AbstractOperator) = begin
    b = _with_sign(b, -1)
    [a, b]
end
Base.:-(a::Vector{<:AbstractOperator}, b::AbstractOperator) = begin
    b = _with_sign(b, -1)
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

Base.:(==)(a::AbstractOperator, b::Src) = begin
    Model{1,1}((a,),(b,))
end

Base.:(==)(a::Vector{<:AbstractOperator}, b::Src) = begin
    Model{length(a),1}((a...,),(b,))
end

Base.:(==)(a::AbstractOperator, b::Vector{<:Src}) = begin
    Model{1,length(b)}((a,),(b...,))
end

Base.:(==)(a::Vector{<:AbstractOperator}, b::Vector{<:Src}) = begin
    Model{length(a), length(b)}((a...,),(b...,))
end

(→)(model::Model{TN,SN,T,S}, eqn::AbstractEquation) where {TN,SN,T,S}= begin
    # To-do: Add runtime check to ensure both sides are consistent (for now document)
    if S.parameters[1].parameters[1] <: AbstractScalarField
        ModelEquation(ScalarModel(), model, eqn, (), ())
    elseif S.parameters[1].parameters[1] <: AbstractVectorField
        ModelEquation(VectorModel(), model, eqn, (), ())
    end
end
