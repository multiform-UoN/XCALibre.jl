export NonLinearRobin, linearize_boundary, update_nonlinear_robin, update_nonlinear_robin!

struct NonLinearRobinValue{M}
    map::M
end
Adapt.@adapt_structure NonLinearRobinValue

@inline Base.getproperty(value::NonLinearRobinValue, name::Symbol) =
    name === :flux_func ? getfield(value, :map).func :
    name === :derivative ? getfield(value, :map).derivative :
    getfield(value, name)

struct NonLinearRobin{I,V,R<:UnitRange} <: AbstractBoundary
    ID::I
    value::V
    IDs_range::R
end
Adapt.@adapt_structure NonLinearRobin

NonLinearRobin(name::Symbol, map::NonlinearMap) =
    NonLinearRobin(name, NonLinearRobinValue(map), 0:0)
NonLinearRobin(name::Symbol, flux_func::Function) =
    NonLinearRobin(name, NonlinearMap(flux_func))
NonLinearRobin(name::Symbol, flux_func::Function, derivative::Function) =
    NonLinearRobin(name, NonlinearMap(flux_func, derivative))

adapt_value(value::NonLinearRobinValue, mesh) = value

@define_boundary NonLinearRobin Laplacian{Linear} begin
    error("NonLinearRobin must be linearized to Robin before assembly. Call update_nonlinear_robin or linearize_physics first.")
end

@inline _nonlinear_robin_value(map::NonlinearMap, value) = map.func(value)

@inline function _nonlinear_robin_derivative(map::NonlinearMap, value, derivative)
    if map.derivative !== nothing
        return map.derivative(value)
    elseif derivative !== nothing
        return derivative(map, value)
    else
        error("NonLinearRobin requires an analytic derivative for GPU-safe lowering. Construct it as NonLinearRobin(name, f, df), NonLinearRobin(name, NonlinearMap(f, df)), or call from linearize_physics on a supported CPU AD path.")
    end
end

@inline function _assert_cpu_nonlinear_robin(field)
    backend = _get_backend(field.mesh)
    backend isa CPU && return nothing
    error("NonLinearRobin patch linearization currently requires a CPU-resident field because it computes a patch representative value before assembly. For GPU solves, linearize on a CPU field or use a matrix-free residual path with a device-callable flux and derivative.")
end

function _patch_average(bc::NonLinearRobin, field)
    boundary_cellsID = field.mesh.boundary_cellsID
    patch_ids = bc.IDs_range
    nfaces = length(patch_ids)
    nfaces == 0 && return zero(_get_float(field.mesh))

    acc = zero(field[boundary_cellsID[first(patch_ids)]])
    for fID in patch_ids
        cID = boundary_cellsID[fID]
        acc += field[cID]
    end
    return acc / nfaces
end

"""
    linearize_boundary(bc::NonLinearRobin, phi0; derivative=nothing)

Lower a nonlinear flux condition `grad(phi).n = f(phi)` to an affine Robin
condition around `phi0`.
"""
function linearize_boundary(bc::NonLinearRobin, phi0; derivative=nothing)
    map = bc.value.map
    f0 = _nonlinear_robin_value(map, phi0)
    df0 = _nonlinear_robin_derivative(map, phi0, derivative)
    F = typeof(phi0)

    # grad(phi).n = f(phi0) + f'(phi0)(phi - phi0)
    # => grad(phi).n - f'(phi0)phi = f(phi0) - f'(phi0)phi0
    return Robin(
        bc.ID,
        RobinValue(a=F(-df0), b=one(F), value=F(f0 - df0 * phi0)),
        bc.IDs_range,
    )
end

"""
    update_nonlinear_robin(field_bcs, field; derivative=nothing)

Return boundary conditions where every `NonLinearRobin` has been lowered to a
standard `Robin` condition using a patch-average value of `field`.
"""
function update_nonlinear_robin(field_bcs::Tuple, field; derivative=nothing)
    any(bc -> bc isa NonLinearRobin, field_bcs) || return field_bcs
    _assert_cpu_nonlinear_robin(field)

    return map(field_bcs) do bc
        if bc isa NonLinearRobin
            phi0 = _patch_average(bc, field)
            linearize_boundary(bc, phi0; derivative=derivative)
        else
            bc
        end
    end
end

function update_nonlinear_robin(field_bcs::AbstractVector, field; derivative=nothing)
    return update_nonlinear_robin(Tuple(field_bcs), field; derivative=derivative)
end

function update_nonlinear_robin(BCs::NamedTuple, field; derivative=nothing)
    names = propertynames(BCs)
    updated = map(values(BCs)) do field_bcs
        field_bcs isa Tuple || field_bcs isa AbstractVector ?
            update_nonlinear_robin(field_bcs, field; derivative=derivative) :
            field_bcs
    end
    return NamedTuple{names}(updated)
end

"""
    update_nonlinear_robin!(field_bcs, field; derivative=nothing)

In-place convenience wrapper for mutable boundary vectors. Immutable boundary
containers such as tuples and named tuples are returned as updated copies.
"""
function update_nonlinear_robin!(field_bcs::AbstractVector, field; derivative=nothing)
    updated = update_nonlinear_robin(field_bcs, field; derivative=derivative)
    for i in eachindex(field_bcs)
        field_bcs[i] = updated[i]
    end
    return field_bcs
end

update_nonlinear_robin!(field_bcs::Tuple, field; derivative=nothing) =
    update_nonlinear_robin(field_bcs, field; derivative=derivative)
update_nonlinear_robin!(BCs::NamedTuple, field; derivative=nothing) =
    update_nonlinear_robin(BCs, field; derivative=derivative)
