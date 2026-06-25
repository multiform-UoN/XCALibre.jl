export discretise!, update_equation!, assemble_matrix!, assemble_rhs!, explicit_residual!

function discretise!(
    eqn::ModelEquation{T,M,E,S,P}, prev, config; rho_prev=_get_flux(eqn.model.terms[1])) where {T<:VectorModel,M,E,S,P}
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware

    # Retrieve variabels for defition
    mesh = get_phi(eqn).mesh
    model = eqn.model

    # Sparse array and b accessor call
    A = _A(eqn)
    A0 = _A0(eqn)
    (; bx, by, bz) = eqn.equation

    # Sparse array fields accessors
    nzval = _nzval(A)
    nzval0 = _nzval(A0)
    colval = _colval(A)
    rowptr = _rowptr(A)

    # reset storage of sparse matrix
    z = zero(eltype(nzval))
    xcal_foreach(nzval, config) do i
        nzval0[i] = z
    end


    # Call discretise kernel
    ndrange = length(mesh.cells)
    kernel! = _discretise_vector_model!(_setup(backend, workgroup, ndrange)...)
    kernel!(model, model.terms, model.sources, mesh, nzval0, nzval, colval, rowptr, bx, by, bz, prev, runtime, rho_prev)
    # KernelAbstractions.synchronize(backend)
end

# @kernel function _discretise_vector_model!(
#     model::Model{TN,SN,T,S}, terms, sources, mesh, nzval0::AbstractArray{F}, nzval, colval, rowptr, bx, by, bz, prev, runtime) where {TN,SN,T,S,F}
@kernel function _discretise_vector_model!(
    model::Model{TN,SN,T,S}, terms::TERMS, sources::SRCS, mesh, nzval0::AbstractArray{F}, nzval, colval, rowptr, bx, by, bz, prev, runtime, rho_prev) where {TN,SN,T,S,F,TERMS,SRCS}
    i = @index(Global)
    # Extract mesh fields for kernel
    (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh

    @inbounds begin
        # Define workitem cell and extract required fields
        cell = cells[i]
        (; faces_range, volume) = cell


        # Set index for sparse array values on diagonal
        cIndex = spindex(rowptr, colval, i, i)

        # For loop over workitem cell faces
        ac_sum = zero(F)
        for fi in faces_range
            # Retrieve indices for discretisation
            fID = cell_faces[fi]
            ns = cell_nsign[fi] # normal sign
            face = faces[fID]
            nID = cell_neighbours[fi]
            cellN = cells[nID]

            # Set index for sparse array values at workitem cell neighbour index
            nIndex = spindex(rowptr, colval, i, nID)


            # Call scheme generated fucntion
            ac, an, _ = _scheme!(model, terms, nzval0, cell, face,  cellN, ns, i, nID, cIndex, nIndex, fID, prev, runtime)
            ac_sum += ac
            nzval0[nIndex] = an

        end


        # Call scheme source generated function NEEDS UPDATING!
        ac, bx1, by1, bz1 = _scheme_source!(model, terms, cell, i, cIndex, prev, runtime, rho_prev)

        nzval0[cIndex] = ac_sum + ac

        # Call sources generated function
        bx2, by2, bz2 = _sources!(model, sources, volume, i)
        bx[i] = bx1 + bx2
        by[i] = by1 + by2
        bz[i] = bz1 + bz2
    end
end

function discretise!(
    eqn::ModelEquation{T,M,E,S,P}, prev, config; rho_prev=_get_flux(eqn.model.terms[1])) where {T<:ScalarModel,M,E,S,P}

    (; hardware, runtime) = config
    (; backend, workgroup) = hardware

    # Retrieve variabels for defition
    mesh = get_phi(eqn).mesh
    model = eqn.model

    # Sparse array and b accessor call
    A = _A(eqn)
    b = _b(eqn)

    # Sparse array fields accessors
    nzval = _nzval(A)
    colval = _colval(A)
    rowptr = _rowptr(A)

    # reset storage of sparse matrix
    z = zero(eltype(nzval))
    xcal_foreach(nzval, config) do i
        nzval[i] = z
    end


    # Call discretise kernel
    ndrange = length(mesh.cells)
    kernel! = _discretise_scalar_model!(_setup(backend, workgroup, ndrange)...)
    kernel!(model, model.terms, model.sources, mesh, nzval, colval, rowptr, b, prev, runtime, rho_prev)
    # KernelAbstractions.synchronize(backend)
end

# Discretise kernel function
# @kernel function _discretise_scalar_model!(
#     model::Model{TN,SN,T,S}, terms, sources, mesh, nzval::AbstractArray{F}, colval, rowptr, b, prev, runtime) where {TN,SN,T,S,F}
@kernel function _discretise_scalar_model!(
    model::Model{TN,SN,T,S}, terms::TERMS, sources::SRCS, mesh, nzval::AbstractArray{F}, colval, rowptr, b, prev, runtime, rho_prev) where {TN,SN,T,S,F,TERMS,SRCS}

    i = @index(Global)
    # Extract mesh fields for kernel
    (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh

    @inbounds begin
        # Define workitem cell and extract required fields
        cell = cells[i]
        (; faces_range, volume) = cell

        # Set index for sparse array values on diagonal!
        cIndex = spindex(rowptr, colval, i, i)
        b[i] = zero(F)

        # For loop over workitem cell faces
        ac_sum = zero(F)
        for fi in faces_range
            # Retrieve indices for discretisation
            fID = cell_faces[fi]
            ns = cell_nsign[fi] # normal sign
            face = faces[fID]
            nID = cell_neighbours[fi]
            cellN = cells[nID]

            # Set index for sparse array values at workitem cell neighbour index
            nIndex = spindex(rowptr, colval, i, nID)

            # Call scheme generated fucntion
            ac, an, bface = _scheme!(model, terms, nzval, cell, face,  cellN, ns, i, nID, cIndex, nIndex, fID, prev, runtime)
            ac_sum += ac
            nzval[nIndex] = an
            b[i] += bface
        end

        # Call scheme source generated function
        ac, b1 = _scheme_source!(model, terms, cell, i, cIndex, prev, runtime, rho_prev)
        nzval[cIndex] = ac_sum + ac

        # Call sources generated function
        b2 = _sources!(model, sources, volume, i)
        b[i] += b2 + b1
    end
end

return_quote(x, t) = :(nothing)

# Scheme generated function definition
# @generated function _scheme!(model::Model{TN,SN,T,S}, terms, nzval, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime) where {TN,SN,T,S}
@generated function _scheme!(model::Model{TN,SN,T,S}, terms::TERMS, nzval::AbstractArray{F}, cell, face,  cellN, ns, cID, nID, cIndex, nIndex, fID, prev, runtime) where {TN,SN,T,S,TERMS,F}
    # Allocate expression array to store scheme function
    out = Expr(:block)

    # Loop over number of terms and store scheme function in array
    for t in 1:TN
        function_call_scheme = quote
            ac, an, b = scheme_contribution!(terms[$t], nzval, cell, face,  cellN, ns, cID, nID, cIndex, nIndex, fID, prev, runtime)
            AC += F(ac)
            AN += F(an)
            B += F(b)
        end
        push!(out.args, function_call_scheme)
    end
    # out
    quote
        z = zero(F)
        AC = z
        AN = z
        B = z
        $(out.args...)
        return AC, AN, B
    end
end

# Scheme source generated function definition
@generated function _scheme_source!(model::Model{TN,SN,T,S}, terms::TERMS, cell::Cell{F}, cID, cIndex, prev, runtime, rho_prev) where {TN,SN,T,S,TERMS,F}
    # Allocate expression array to store scheme_source function
    out = Expr(:block)

    # Determine scalar vs vector model.
    # When SN>0 use source field type; when SN==0 fall back to first term's phi type.
    # (S = Tuple{} when no sources, so S.parameters[1] would throw for SN==0.)
    phi_field_type = if SN > 0
        S.parameters[1].parameters[1]
    elseif TN > 0
        T.parameters[1].parameters[2]  # Operator{F, P, I, Type} → P is the phi field
    else
        AbstractScalarField  # empty model: default to scalar path
    end

    # Loop over number of terms and store scheme_source function in array
    if phi_field_type <: AbstractScalarField
        for t in 1:TN
            function_call_scheme_source = quote
                ac, b = scheme_source!(terms[$t], cell, cID, cIndex, prev, runtime, rho_prev)
                AC += F(ac)
                B += F(b)
            end
            push!(out.args, function_call_scheme_source)
        end
        return quote
            z = zero(F)
            AC = z
            B = z
            $(out.args...)
            return AC, B
        end
    elseif phi_field_type <: AbstractVectorField
        for t in 1:TN
            function_call_scheme_source = quote
                ac, bx = scheme_source!(terms[$t], cell, cID, cIndex, prev.x, runtime, rho_prev)
                ac, by = scheme_source!(terms[$t], cell, cID, cIndex, prev.y, runtime, rho_prev)
                ac, bz = scheme_source!(terms[$t], cell, cID, cIndex, prev.z, runtime, rho_prev)
                AC += F(ac) # assuming ac's for all directions are equal
                BX += F(bx)
                BY += F(by)
                BZ += F(bz)
            end
            push!(out.args, function_call_scheme_source)
        end
        return quote
            z = zero(F)
            AC = z
            BX = z
            BY = z
            BZ = z
            $(out.args...)
            return AC, BX, BY, BZ
        end
    end
end

@inline function _scheme_source!(model::Model, terms, cell, cID, cIndex, prev, runtime)
    rho_prev = (length(terms) > 0 && hasproperty(terms[1], :flux)) ? terms[1].flux : ConstantScalar(1.0)
    return _scheme_source!(model, terms, cell, cID, cIndex, prev, runtime, rho_prev)
end

# Sources generated function definition
@generated function _sources!(model::Model{TN,SN,T,S}, sources::SRC, volume::F, cID) where {TN,SN,T,S,SRC,F}
    # Allocate expression array to store source function
    out = Expr(:block)

    # Same scalar/vector detection as _scheme_source!
    phi_field_type = if SN > 0
        S.parameters[1].parameters[1]
    elseif TN > 0
        T.parameters[1].parameters[2]
    else
        AbstractScalarField
    end

    # Loop over number of terms and store source function in array
    if phi_field_type <: AbstractScalarField
        for s in 1:SN
            expression_call_sources = quote
                (; field, sign) = sources[$s]
                B += F(sign*field[cID]*volume)
            end
            push!(out.args, expression_call_sources)
        end
        return quote
            B = zero(F)
            $(out.args...)
            return B
        end
    elseif phi_field_type <: AbstractVectorField
        for s in 1:SN
            expression_call_sources = quote
                (; field, sign) = sources[$s]
                Bx += F(sign*field.x[cID]*volume)
                By += F(sign*field.y[cID]*volume)
                Bz += F(sign*field.z[cID]*volume)
            end
            push!(out.args, expression_call_sources)
        end
        return quote
            z = zero(F)
            Bx = z
            By = z
            Bz = z
            $(out.args...)
            return Bx, By, Bz
        end
    end
end

@kernel function set_nzval!(nzval::AbstractArray{T}) where T
    i = @index(Global)

    @inbounds begin
        nzval[i] = zero(T)
    end
end

# Reset main equation to reuse in segregated solver
function update_equation!(eqn::ModelEquation{T,M,E,S,P}, config) where {T<:VectorModel,M,E,S,P}
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware

    # Sparse array and b accessor call
    A = _A(eqn)
    A0 = _A0(eqn)

    # Sparse array fields accessors
    nzval0 = _nzval(A0)
    nzval = _nzval(A)

    # Call set nzval to zero kernel
    ndrange = length(nzval0)
    kernel! = _update_equation!(_setup(backend, workgroup, ndrange)...)
    kernel!(nzval, nzval0)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function _update_equation!(nzval, nzval0)
    i = @index(Global)

    @inbounds begin
        nzval[i] = nzval0[i]
    end
end

# ---------------------------------------------------------
# PHASE 4: Split Assembly
# ---------------------------------------------------------

function assemble_matrix!(eqn::ModelEquation{T,M,E,S,P}, config) where {T<:ScalarModel,M,E,S,P}
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware
    mesh = get_phi(eqn).mesh
    A = _A(eqn)
    nzval = _nzval(A)
    colval = _colval(A)
    rowptr = _rowptr(A)
    z = zero(eltype(nzval))
    xcal_foreach(nzval, config) do i
        nzval[i] = z
    end
    ndrange = length(mesh.cells)
    kernel! = _assemble_matrix_scalar_model!(_setup(backend, workgroup, ndrange)...)
    kernel!(eqn.model, eqn.model.terms, mesh, nzval, colval, rowptr, get_values(get_phi(eqn), nothing), runtime)
end

@kernel function _assemble_matrix_scalar_model!(model::Model{TN,SN,T,S}, terms::TERMS, mesh, nzval::AbstractArray{F}, colval, rowptr, prev, runtime) where {TN,SN,T,S,F,TERMS}
    i = @index(Global)
    (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh
    @inbounds begin
        cell = cells[i]
        (; faces_range) = cell
        cIndex = spindex(rowptr, colval, i, i)
        ac_sum = zero(F)
        for fi in faces_range
            fID = cell_faces[fi]
            ns = cell_nsign[fi]
            face = faces[fID]
            nID = cell_neighbours[fi]
            cellN = cells[nID]
            nIndex = spindex(rowptr, colval, i, nID)
            ac, an, _ = _scheme!(model, terms, nzval, cell, face, cellN, ns, i, nID, cIndex, nIndex, fID, prev, runtime)
            ac_sum += ac
            nzval[nIndex] = an
        end
        ac, _ = _scheme_source!(model, terms, cell, i, cIndex, prev, runtime)
        nzval[cIndex] = ac_sum + ac
    end
end

function assemble_matrix!(eqn::ModelEquation{T,M,E,S,P}, config) where {T<:VectorModel,M,E,S,P}
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware
    mesh = get_phi(eqn).mesh
    A = _A(eqn)
    A0 = _A0(eqn)
    nzval = _nzval(A)
    nzval0 = _nzval(A0)
    colval = _colval(A)
    rowptr = _rowptr(A)
    z = zero(eltype(nzval))
    xcal_foreach(nzval, config) do i
        nzval0[i] = z
    end
    ndrange = length(mesh.cells)
    kernel! = _assemble_matrix_vector_model!(_setup(backend, workgroup, ndrange)...)
    kernel!(eqn.model, eqn.model.terms, mesh, nzval0, colval, rowptr, get_phi(eqn), runtime)
end

@kernel function _assemble_matrix_vector_model!(model::Model{TN,SN,T,S}, terms::TERMS, mesh, nzval0::AbstractArray{F}, colval, rowptr, prev, runtime) where {TN,SN,T,S,F,TERMS}
    i = @index(Global)
    (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh
    @inbounds begin
        cell = cells[i]
        (; faces_range) = cell
        cIndex = spindex(rowptr, colval, i, i)
        ac_sum = zero(F)
        for fi in faces_range
            fID = cell_faces[fi]
            ns = cell_nsign[fi]
            face = faces[fID]
            nID = cell_neighbours[fi]
            cellN = cells[nID]
            nIndex = spindex(rowptr, colval, i, nID)
            ac, an, _ = _scheme!(model, terms, nzval0, cell, face, cellN, ns, i, nID, cIndex, nIndex, fID, prev, runtime)
            ac_sum += ac
            nzval0[nIndex] = an
        end
        ac, _, _, _ = _scheme_source!(model, terms, cell, i, cIndex, prev, runtime)
        nzval0[cIndex] = ac_sum + ac
    end
end

function assemble_rhs!(eqn::ModelEquation{T,M,E,S,P}, source::AbstractSource, config) where {T<:ScalarModel,M,E,S,P}
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware
    mesh = get_phi(eqn).mesh
    b = _b(eqn)
    xcal_foreach(b, config) do i
        b[i] = zero(eltype(b))
    end
    ndrange = length(mesh.cells)
    kernel! = _assemble_rhs_scalar_model!(_setup(backend, workgroup, ndrange)...)
    temp_sources = (source,)
    kernel!(eqn.model, eqn.model.terms, temp_sources, mesh, b, get_values(get_phi(eqn), nothing), runtime)
end

@kernel function _assemble_rhs_scalar_model!(model::Model{TN,SN,T,S}, terms::TERMS, sources::SRCS, mesh, b::AbstractArray{F}, prev, runtime) where {TN,SN,T,S,F,TERMS,SRCS}
    i = @index(Global)
    (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh
    @inbounds begin
        cell = cells[i]
        (; faces_range, volume) = cell
        b[i] = zero(F)
        for fi in faces_range
            fID = cell_faces[fi]
            ns = cell_nsign[fi]
            face = faces[fID]
            nID = cell_neighbours[fi]
            cellN = cells[nID]
            _, _, bface = _scheme!(model, terms, b, cell, face, cellN, ns, i, nID, 1, 1, fID, prev, runtime)
            b[i] += bface
        end
        _, b1 = _scheme_source!(model, terms, cell, i, 1, prev, runtime)
        b2 = _sources!(model, sources, volume, i)
        b[i] += b2 + b1
    end
end

# ---------------------------------------------------------
# PHASE 5: Matrix-Free Evaluation
# ---------------------------------------------------------
# TWO RESIDUAL PATHS — keep these distinct:
#
#   explicit_residual!(r, eqn, phi, config)
#     INTERIOR-ONLY explicit evaluation. Loops over cell.faces_range which
#     contains only interior face indices by mesh topology design.
#     BC face contributions are absent. Used internally as the interior half
#     of the explicit path in residual!(explicit=true).
#
#   residual!(r, eqn, config)            ← the one users should call
#     FULL residual = interior + BC contributions. Two paths:
#       explicit=false (default): A·φ − b from the assembled sparse matrix
#                                 (fvm:: style — BCs already in A and b)
#       explicit=true:            explicit_residual! + apply_bc_residuals!
#                                 (fvc:: style — re-evaluates fluxes directly)
#     Both give the same result for linear problems.

function explicit_residual!(r::AbstractVector, eqn::ModelEquation{T,M,E,S,P}, phi, config) where {T<:ScalarModel,M,E,S,P}
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware
    mesh = get_phi(eqn).mesh
    ndrange = length(mesh.cells)
    kernel! = _explicit_residual_scalar!(_setup(backend, workgroup, ndrange)...)
    kernel!(eqn.model, eqn.model.terms, eqn.model.sources, mesh, r, get_values(phi, nothing), runtime)
    KernelAbstractions.synchronize(backend)
end

@kernel function _explicit_residual_scalar!(model::Model{TN,SN,T,S}, terms::TERMS, sources::SRCS, mesh, r::AbstractArray{F}, prev, runtime) where {TN,SN,T,S,F,TERMS,SRCS}
    i = @index(Global)
    (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh
    @inbounds begin
        cell = cells[i]
        (; faces_range, volume) = cell
        r[i] = zero(F)
        ac_sum = zero(F)
        an_phi_sum = zero(F)
        b_sum = zero(F)
        for fi in faces_range
            fID = cell_faces[fi]
            ns = cell_nsign[fi]
            face = faces[fID]
            nID = cell_neighbours[fi]
            cellN = cells[nID]
            ac, an, bface = _scheme!(model, terms, r, cell, face, cellN, ns, i, nID, 1, 1, fID, prev, runtime)
            ac_sum += ac
            an_phi_sum += an * prev[nID]
            b_sum += bface
        end
        ac, b1 = _scheme_source!(model, terms, cell, i, 1, prev, runtime)
        b2 = _sources!(model, sources, volume, i)

        # r = A*phi - b
        r[i] = (ac_sum + ac) * prev[i] + an_phi_sum - (b_sum + b1 + b2)
    end
end
