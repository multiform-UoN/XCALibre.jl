export NeumannFunction

# abstract type XCALibreUserFunctor end

"""
    NeumannFunction(ID, value) <: AbstractNeumann

Neumann boundary condition defined with user-provided function.

# Input
- `ID` Name of the boundary given as a symbol (e.g. :inlet). Internally it gets replaced with the boundary index ID
- `value` Custom function defining the desired Neumann boundary condition.

# Function requirements

`value(coords, time, index)` returns the face-normal gradient (outward normal) at the face
centre `coords`, with `index` the local face index (1 to N) on the patch, as for
`DirichletFunction`. The face value is the boundary-cell value plus `delta` times this
gradient. For `XCALibreUserFunctor` values the condition still acts as a zero gradient.
"""
struct NeumannFunction{I,V,R<:UnitRange} <: AbstractNeumann
    ID::I 
    value::V
    IDs_range::R
end
Adapt.@adapt_structure NeumannFunction

# Face-normal gradient of a Neumann-type condition at local face index i
@inline _neumann_gradient(bc::NeumannFunction{I,V}, coords, time, i) where {I,V<:Function} =
    bc.value(coords, time, i)
@inline _neumann_gradient(bc::NeumannFunction, coords, time, i) = 0.0

@define_boundary NeumannFunction Laplacian{Linear} begin
    J = term.flux[fID]
    0.0, -term.sign*J*faces.area[fID]*_neumann_gradient(bc, faces.centre[fID], time, i)
end

@define_boundary NeumannFunction Divergence{Linear} begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    ap, -_neumann_gradient(bc, faces.centre[fID], time, i)*ap*faces.delta[fID]
end

@define_boundary NeumannFunction Divergence{Upwind} begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    ap, -_neumann_gradient(bc, faces.centre[fID], time, i)*ap*faces.delta[fID]
end

@define_boundary NeumannFunction Divergence{LUST} begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    ap, -_neumann_gradient(bc, faces.centre[fID], time, i)*ap*faces.delta[fID]
end

# Bounded: upwind (ap,0) minus ap on diagonal -> (0,0)
@define_boundary NeumannFunction Divergence{BoundedUpwind} begin
    0.0, 0.0
end

@define_boundary NeumannFunction Si begin
    0.0, 0.0
end