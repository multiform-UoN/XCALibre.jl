export apply_boundary_conditions!, apply_bc_residuals!



apply_boundary_conditions!(eqn, config; time=nothing, component=nothing) = begin
    _apply_boundary_conditions!(eqn.model, get_bcs(eqn), eqn, component, time, config)
end

apply_boundary_conditions!(eqn, BCs, component, time, config) = begin
    _apply_boundary_conditions!(eqn.model, BCs, eqn, component, time, config)
end

# Apply Boundaries Function
function _apply_boundary_conditions!(
    model::Model{TN,SN,T,S}, BCs::B, eqn, component, time, config) where {TN,SN,T,S,B}
    nTerms = length(model.terms)

    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Retriecve variables for function
    mesh = get_phi(eqn).mesh
    A = _A(eqn)
    b = _b(eqn, component)

    # Deconstruct mesh to required fields
    (; faces, cells, boundary_cellsID) = mesh

    # Call sparse array field accessors
    colval = _colval(A)
    rowptr = _rowptr(A)
    nzval = _nzval(A)

    # Test implementation looking over all boundary faces
    nbfaces = length(mesh.boundary_cellsID)

    # Ensure BCs is a Tuple for the @generated kernel unrolling
    BCs_tuple = Tuple(BCs)

    for BC ∈ BCs
        facesID_range = BC.IDs_range
        # update user defined boundary storage (if needed)
        update_user_boundary!(BC, faces, cells, facesID_range, time, config)
    end

    ndrange = nbfaces
    if ndrange > 0
        kernel! = apply_boundary_conditions_kernel!(_setup(backend, workgroup, ndrange)...)
        kernel!(
            model, BCs_tuple, model.terms, faces, cells, boundary_cellsID, colval, rowptr, nzval, b, component, time, ndrange=ndrange
            )
        KernelAbstractions.synchronize(backend)
    end
end

update_user_boundary!(
    BC::AbstractBoundary, faces, cells, facesID_range, time, config) = nothing

# Apply boundary conditions kernel definition
# Experimental implementation

@kernel function apply_boundary_conditions_kernel!(
    model::Model{TN,SN,T,S}, BCs, terms,
    faces, cells, boundary_cellsID, colval, rowptr, nzval, b, component, time
    ) where {TN,SN,T,S}
    fID = @index(Global)

    calculate_coefficients(
        BCs, model, terms, faces, cells, boundary_cellsID, colval, rowptr, nzval, b, component, time, fID)
end

function calculate_coefficients(
    BCs, model, terms, faces, cells, boundary_cellsID, colval, rowptr, nzval, b, component, time, fID)

    # Non-generated loop over boundary patches.
    # This is safe for both Tuple and Vector BCs.
    for BC ∈ BCs
        (; start, stop) = BC.IDs_range
        if start <= fID <= stop
            i = fID - start + 1
            cellID = boundary_cellsID[fID]
            face = faces[fID]
            cell = cells[cellID]

            zcellID = spindex(rowptr, colval, cellID, cellID)
            AP, BP = apply!(
                model, BC, terms,
                colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
                )
            Atomix.@atomic nzval[zcellID] += AP
            Atomix.@atomic b[cellID] += BP
            return nothing
        end
    end
    return nothing
end

@generated function get_BC(BCs, index)
    N = length(BCs.parameters)
    exprs = Expr(:block)
    for i ∈ 1:N
        ex = quote
            if index == $i
                @inbounds BC = BCs[$i]
                (; start, stop) = BC.IDs_range
                return BC, start, stop
            end
        end
        push!(exprs.args, ex)
    end
    return exprs
end



# Current implementation

# @kernel function apply_boundary_conditions_kernel!(
#     model::Model{TN,SN,T,S}, BC, terms,
#     faces, cells, start_ID, boundary_cellsID, colval, rowptr, nzval, b, component, time
#     ) where {TN,SN,T,S}
#     i = @index(Global)

#     # Redefine thread index to correct starting ID
#     j = i + start_ID - 1
#     fID = j

#     # Retrieve workitem cellID, cell and face
#     cellID = boundary_cellsID[j]
#     face = faces[fID]
#     cell = cells[cellID]

#     zcellID = spindex(rowptr, colval, cellID, cellID)

#     # Call apply generated function
#     AP, BP = apply!(
#         model, BC, terms,
#         colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
#         )
#     Atomix.@atomic nzval[zcellID] += AP
#     Atomix.@atomic b[cellID] += BP
# end

# Apply generated function definition
@generated function apply!(
    model::Model{TN,SN,T,S}, BC, terms, colval, rowptr, nzval::AbstractArray{F},
    cellID, zcellID, cell, face, fID, i, component, time
    ) where {TN,SN,T,S,F}

    # Definition of main assignment loop (one per patch)
    func_calls = Expr[]
    for t ∈ 1:TN
        call = quote
            ap, bp = BC(
                terms[$t],
                colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
                )
            AP += F(ap)
            BP += F(bp)
        end
        push!(func_calls, call)
    end
    quote
        z = zero(F)
        AP = z
        BP = z
        $(func_calls...)
        return AP, BP
    end
end

# =============================================================================
# BC RESIDUAL PATH
# =============================================================================
# apply_bc_residuals!(r, eqn, config) adds boundary-face contributions to a
# pre-allocated residual vector.  The formula is identical to the assembly path:
#   r[cellID] += AP * phi_P - BP
# where (AP, BP) come from the same apply!(model, BC, terms, ...) generated
# function used during matrix assembly — zero formula duplication.
#
# This is the missing piece to form the FULL residual without re-assembling A:
#   residual!(r, eqn, config) = explicit_residual! (interior) + apply_bc_residuals! (BCs)
#
# NOTE: Standard BCs (Dirichlet, Neumann, Robin, etc.) are handled here.
# For genuinely nonlinear BCs (NonLinearRobin), additionally call
# add_bc_residuals! from the bc_ops extension interface.

apply_bc_residuals!(r, eqn, config; component=nothing, time=nothing) = begin
    _apply_bc_residuals!(r, eqn.model, get_bcs(eqn), eqn, component, time, config)
end

function _apply_bc_residuals!(
    r::AbstractVector, model::Model{TN,SN,T,S}, BCs::B, eqn, component, time, config
) where {TN,SN,T,S,B}
    (; hardware) = config
    (; backend, workgroup) = hardware
    mesh = get_phi(eqn).mesh
    A = _A(eqn)
    phi_vals = get_values(get_phi(eqn), component)
    (; faces, cells, boundary_cellsID) = mesh
    colval = _colval(A)
    rowptr = _rowptr(A)
    nzval  = _nzval(A)
    BCs_tuple = Tuple(BCs)
    for BC ∈ BCs
        update_user_boundary!(BC, faces, cells, BC.IDs_range, time, config)
    end
    nbfaces = length(mesh.boundary_cellsID)
    if nbfaces > 0
        kernel! = _bc_residuals_kernel!(_setup(backend, workgroup, nbfaces)...)
        kernel!(
            r, phi_vals, model, BCs_tuple, model.terms,
            faces, cells, boundary_cellsID, colval, rowptr, nzval, component, time,
            ndrange=nbfaces
        )
        KernelAbstractions.synchronize(backend)
    end
end

@kernel function _bc_residuals_kernel!(
    r::AbstractArray{F}, phi_vals::AbstractArray{F},
    model::Model{TN,SN,T,S}, BCs, terms,
    faces, cells, boundary_cellsID, colval, rowptr, nzval, component, time
) where {F,TN,SN,T,S}
    fID = @index(Global)
    _accumulate_bc_residual!(
        r, phi_vals, BCs, model, terms,
        faces, cells, boundary_cellsID, colval, rowptr, nzval, component, time, fID
    )
end

function _accumulate_bc_residual!(
    r, phi_vals, BCs, model, terms,
    faces, cells, boundary_cellsID, colval, rowptr, nzval, component, time, fID
)
    for BC ∈ BCs
        (; start, stop) = BC.IDs_range
        if start <= fID <= stop
            i       = fID - start + 1
            cellID  = boundary_cellsID[fID]
            face    = faces[fID]
            cell    = cells[cellID]
            zcellID = spindex(rowptr, colval, cellID, cellID)
            AP, BP  = apply!(
                model, BC, terms,
                colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
            )
            Atomix.@atomic r[cellID] += AP * phi_vals[cellID] - BP
            return nothing
        end
    end
    return nothing
end

# Boundary indices generated function definition
@generated function boundary_indices(mesh::M, BCs::B) where {M<:AbstractMesh,B}

    # Definition of main boundary indices loop (one per patch)
    unpacked_BCs = []
    for i ∈ 1:length(BCs.parameters)
        unpack = quote
            name = BCs[$i].name
            index = boundary_index(boundaries, name)
            BC_indices = (BC_indices..., index)
        end
        push!(unpacked_BCs, unpack)
    end
    quote
        boundaries = mesh.boundaries
        BC_indices = ()
        $(unpacked_BCs...)
        return BC_indices
    end
end
