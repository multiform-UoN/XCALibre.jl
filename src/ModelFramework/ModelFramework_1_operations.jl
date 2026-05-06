export →

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
        ModelEquation(ScalarModel(), model, eqn, nothing, nothing)
    elseif S.parameters[1].parameters[1] <: AbstractVectorField
        ModelEquation(VectorModel(), model, eqn, nothing, nothing)
    end
end
