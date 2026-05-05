using ForwardDiff
export NonLinearRobin, update_nonlinear_robin!

struct NonLinearRobinValue{Fun}
    flux_func::Fun
end
Adapt.@adapt_structure NonLinearRobinValue

struct NonLinearRobin{I,V,R<:UnitRange} <: AbstractBoundary
    ID::I 
    value::V
    IDs_range::R 
end
Adapt.@adapt_structure NonLinearRobin

NonLinearRobin(name::Symbol, flux_func::Function) = begin
    NonLinearRobin(name, NonLinearRobinValue(flux_func), 0:0)
end

adapt_value(value::NonLinearRobinValue, mesh) = value

"""
    update_nonlinear_robin!(BCs, field)

Updates Robin boundary conditions that were derived from NonLinearRobin.
This should be called inside the simulation loop before `solve_equation!`.
"""
function update_nonlinear_robin!(BCs, field)
    mesh = field.mesh
    boundaries = mesh.boundaries
    boundary_cellsID = mesh.boundary_cellsID
    
    # We assume the user has a NamedTuple of BCs for different fields
    # We look for Robin BCs that have a 'source_nonlinear' property or similar
    # For now, let's just make a manual mapping or a specific struct.
    
    # Better: The user provides the NonLinearRobin definitions, 
    # and we update the corresponding Robin BCs in the config.
end

# We can actually make NonLinearRobin a "Generator" for Robin BCs.
# But XCALibre's config.boundaries is usually a NamedTuple of Tuples of BCs.

@define_boundary NonLinearRobin Laplacian{Linear} begin
    # This shouldn't be called directly in the kernel if we linearize outside.
    # But if we want it to be "automatic", we could try to linearize here.
    # However, ForwardDiff is NOT GPU compatible and slow in kernels.
    error("NonLinearRobin must be linearized before discretisation. Use linearize_boundary!")
end

# Implementation of a helper to transform NonLinearRobin to Robin
function linearize_boundary(bc::NonLinearRobin, c_val)
    f = bc.value.flux_func
    # Use ForwardDiff to get f(c) and f'(c)
    f0 = f(c_val)
    df0 = ForwardDiff.derivative(f, c_val)
    
    # grad(c).n = f(c)  =>  grad(c).n \approx f(c0) + f'(c0)(c - c0)
    # grad(c).n - f'(c0)c = f(c0) - f'(c0)c0
    # Robin: b*grad(c).n + a*c = value
    # Matches: b = 1, a = -df0, value = f0 - df0*c_val
    return Robin(bc.ID, RobinValue(a=-df0, b=1.0, value=f0 - df0*c_val), bc.IDs_range)
end

function update_nonlinear_robin(field_bcs, field)
    # Only proceed if we have NonLinearRobin BCs
    has_nonlinear = any(bc -> typeof(bc) <: NonLinearRobin, field_bcs)
    !has_nonlinear && return field_bcs

    mesh = field.mesh
    boundary_cellsID = mesh.boundary_cellsID
    vals = field.values
    
    new_bcs = map(field_bcs) do bc
        if typeof(bc) <: NonLinearRobin
            # Linearize using patch-averaged value
            patch_ids = bc.IDs_range
            sum_val = 0.0
            count = 0
            for fID in patch_ids
                cID = boundary_cellsID[fID]
                sum_val += vals[cID]
                count += 1
            end
            avg_val = count > 0 ? sum_val / count : 0.0
            return linearize_boundary(bc, avg_val)
        else
            return bc
        end
    end
    return Tuple(new_bcs)
end
