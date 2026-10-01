export discretise!, update_equation!, assemble_matrix!, assemble_rhs!, explicit_residual!

# NEW SECTION: kernel arguments
# Kernel arguments are copied by value into per-thread local memory, so terms, sources and fields
# reach the discretise kernels without their mesh; phi keeps only the face_gDiff column schemes read.
_kernel_field(f, m=()) = f
_kernel_field(f::ScalarField, m=()) = ScalarField(f.values, m)
_kernel_field(f::FaceScalarField, m=()) = FaceScalarField(f.values, m)
_kernel_field(f::VectorField, m=()) = VectorField(_kernel_field(f.x), _kernel_field(f.y), _kernel_field(f.z), m)
_kernel_field(f::FaceVectorField, m=()) =
    FaceVectorField(_kernel_field(f.x), _kernel_field(f.y), _kernel_field(f.z), m)

_kernel_term(t::Operator, m) =
    Operator(_kernel_field(t.flux), _kernel_field(t.phi, m), t.sign, t.type)
_kernel_term(t::NonlinearOperator, m) = NonlinearOperator(_kernel_term(t.op, m), t.map)
_kernel_term(t::AffineOperator, m) = AffineOperator(
    _kernel_term(t.op, m), _kernel_field(t.jacobian), _kernel_field(t.offset),
    _kernel_field(t.reference), t.map)

_kernel_model(model, mesh) = begin
    m = (; face_gDiff=mesh.face_gDiff)
    terms = map(t -> _kernel_term(t, m), model.terms)
    sources = map(s -> Src(_kernel_field(s.field), s.sign), model.sources)
    terms, sources
end

function discretise!(
    eqn::ModelEquation{T,M,E,S,P}, prev, config; rho_prev=_get_flux(eqn.model.terms[1])) where {T<:VectorModel,M,E,S,P}
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware

    # Retrieve variabels for defition
    mesh = _term_phi(eqn.model.terms[1]).mesh
    model = eqn.model

    # Sparse array and b accessor call
    A = _A(eqn)
    A0 = _A0(eqn)
    (; bx, by, bz) = eqn.equation

    # Sparse array fields accessors
    nzval = _nzval(A)
    nzval0 = _nzval(A0)
    (; diag_nz, face_nz) = eqn.equation

    _pattern_extended(nzval0, mesh) && fill_nzval!(nzval0, config)

    terms, sources = _kernel_model(model, mesh)
    (; cells, faces, cell_faces, cell_neighbours, cell_nsign) = mesh
    ndrange = length(cells)
    kernel! = _sized(_discretise_vector_model!, backend, workgroup, ndrange)
    kernel!(terms, sources, cells, faces, cell_faces, cell_neighbours, cell_nsign, nzval0,
        diag_nz, face_nz, bx, by, bz, _kernel_field(prev), runtime, _kernel_field(rho_prev))
    # # KernelAbstractions.synchronize(backend)
end

@kernel function _discretise_vector_model!(
    terms::TERMS, sources::SRCS, cells, faces, cell_faces, cell_neighbours, cell_nsign,
    nzval0::AbstractArray{F}, diag_nz, face_nz, bx, by, bz, prev, runtime, rho_prev) where {F,TERMS,SRCS}
    i = @index(Global)

    @inbounds begin
        # Define workitem cell and extract required fields
        faces_range = cells.faces_range[i]
        volume = cells.volume[i]


        cIndex = diag_nz[i]

        # For loop over workitem cell faces
        ac_sum = zero(F)
        for fi in faces_range
            # Retrieve indices for discretisation
            fID = cell_faces[fi]
            ns = cell_nsign[fi] # normal sign
            nID = cell_neighbours[fi]
            nIndex = face_nz[fi]


            # Call scheme generated fucntion
            ac, an, _ = _scheme!(terms, nzval0, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)
            ac_sum += ac
            nzval0[nIndex] = an

        end

        
        # Call scheme source generated function NEEDS UPDATING!
        ac, bx1, by1, bz1 = _scheme_source!(terms, cells, i, cIndex, prev, runtime, rho_prev)
        
        nzval0[cIndex] = ac_sum + ac

        # Call sources generated function
        bx2, by2, bz2 = _sources!(sources, volume, i)
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
    mesh = _term_phi(eqn.model.terms[1]).mesh
    model = eqn.model

    # Sparse array and b accessor call
    A = _A(eqn)
    b = _b(eqn)

    # Sparse array fields accessors
    nzval = _nzval(A)
    (; diag_nz, face_nz) = eqn.equation

    _pattern_extended(nzval, mesh) && fill_nzval!(nzval, config)

    terms, sources = _kernel_model(model, mesh)
    (; cells, faces, cell_faces, cell_neighbours, cell_nsign) = mesh
    ndrange = length(cells)
    kernel! = _sized(_discretise_scalar_model!, backend, workgroup, ndrange)
    kernel!(terms, sources, cells, faces, cell_faces, cell_neighbours, cell_nsign, nzval,
        diag_nz, face_nz, b, _kernel_field(prev), runtime, _kernel_field(rho_prev))
    # # KernelAbstractions.synchronize(backend)
end

# the kernels assign every diagonal and one entry per cell face, which is the whole pattern unless a
# boundary condition added entries (periodic) or two faces share a cell pair; only then is a reset needed
_pattern_extended(nzval, mesh) = length(nzval) != length(mesh.cells) + length(mesh.cell_faces)

fill_nzval!(nzval, config) = begin
    z = zero(eltype(nzval))
    xcal_foreach(nzval, config) do i
        nzval[i] = z
    end
end

@kernel function _discretise_scalar_model!(
    terms::TERMS, sources::SRCS, cells, faces, cell_faces, cell_neighbours, cell_nsign,
    nzval::AbstractArray{F}, diag_nz, face_nz, b, prev, runtime, rho_prev) where {F,TERMS,SRCS}

    i = @index(Global)

    @inbounds begin
        # Define workitem cell and extract required fields
        faces_range = cells.faces_range[i]
        volume = cells.volume[i]

        cIndex = diag_nz[i]

        # For loop over workitem cell faces
        ac_sum = zero(F)
        b_face = zero(F)
        for fi in faces_range
            # Retrieve indices for discretisation
            fID = cell_faces[fi]
            ns = cell_nsign[fi] # normal sign
            nID = cell_neighbours[fi]
            nIndex = face_nz[fi]

            # Call scheme generated fucntion
            ac, an, bf = _scheme!(terms, nzval, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)
            ac_sum += ac
            b_face += bf
            nzval[nIndex] = an
        end
        
        # Call scheme source generated function
        ac, b1 = _scheme_source!(terms, cells, i, cIndex, prev, runtime, rho_prev)
        nzval[cIndex] = ac_sum + ac

        # Call sources generated function
        b2 = _sources!(sources, volume, i)
        b[i] = b2 + b1 + b_face
    end
end

return_quote(x, t) = :(nothing)

# Scheme generated function definition
@generated function _scheme!(
    terms::TERMS, nzval::AbstractArray{F}, cells, faces,
    nID, ns, cIndex, nIndex, fID, prev, runtime
    ) where {TERMS,F}
    TN = fieldcount(TERMS)
    # Allocate expression array to store scheme function
    out = Expr(:block)

    # Loop over number of terms and store scheme function in array
    for t in 1:TN
        function_call_scheme = quote
            ac, an = scheme!(terms[$t], nzval, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)
            bf = _scheme_face_rhs(terms[$t], nzval, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)
            AC += F(ac)
            AN += F(an)
            B += F(bf)
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
@generated function _scheme_source!(terms::TERMS, cells::AbstractVector{<:Cell{F}}, cID, cIndex, prev::P, runtime, rho_prev) where {TERMS,F,P}
    TN = fieldcount(TERMS)
    # Allocate expression array to store scheme_source function
    out = Expr(:block)
    
    # Loop over number of terms and store scheme_source function in array
    if !(P <: AbstractVectorField)
        for t in 1:TN
            function_call_scheme_source = quote
                ac, b = scheme_source!(terms[$t], cells, cID, cIndex, prev, runtime, rho_prev)
                AC += F(ac)
                B += F(b)
            end
            push!(out.args, function_call_scheme_source)
        end
        return quote
            z = zero(F)
            ac = z
            b = z
            AC = z
            B = z
            $(out.args...)
            return AC, B
        end
    else
        for t in 1:TN
            function_call_scheme_source = quote
                ac, bx = scheme_source!(terms[$t], cells, cID, cIndex, prev.x, runtime, rho_prev)
                ac, by = scheme_source!(terms[$t], cells, cID, cIndex, prev.y, runtime, rho_prev)
                ac, bz = scheme_source!(terms[$t], cells, cID, cIndex, prev.z, runtime, rho_prev)
                AC += F(ac) # assuming ac's for all directions are equal
                BX += F(bx)
                BY += F(by)
                BZ += F(bz)
            end
            push!(out.args, function_call_scheme_source)
        end
        return quote
            z = zero(F)
            ac = z
            bx = z
            by = z
            bz = z
            AC = z
            BX = z
            BY = z
            BZ = z
            $(out.args...)
            return AC, BX, BY, BZ
        end
    end
end

# Operator scaling wraps a source field in ScaledFlux. Classification has to see the
# field underneath, otherwise the generated source kernel has no scalar or vector body.
_unscaled_flux_type(::Type{ScaledFlux{F,V}}) where {F,V} = _unscaled_flux_type(F)
_unscaled_flux_type(::Type{F}) where {F} = F

# Sources generated function definition
@generated function _sources!(
    sources::SRC, volume::F, cID
    ) where {SRC,F}
    SN = fieldcount(SRC)
    # Allocate expression array to store source function
    out = Expr(:block)
    field_type = _unscaled_flux_type(SRC.parameters[1].parameters[1])

    # Loop over number of terms and store source function in array
    if field_type <: AbstractScalarField
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
    elseif field_type <: AbstractVectorField
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
    kernel! = _sized(_update_equation!, backend, workgroup, ndrange)
    kernel!(nzval, nzval0)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function _update_equation!(nzval, nzval0) 
    i = @index(Global)

    @inbounds begin
        nzval[i] = nzval0[i]
    end
end

# Split assembly and matrix-free evaluation for the PDE layer. Coefficients come from the
# same _scheme! / _scheme_source! used by discretise!, including the affine face offset.
function assemble_matrix!(eqn::ModelEquation{T,M,E,S,P}, config) where {T<:ScalarModel,M,E,S,P}
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware
    mesh = get_phi(eqn).mesh
    A = _A(eqn)
    nzval = _nzval(A)
    (; diag_nz, face_nz) = eqn.equation
    fill_nzval!(nzval, config)
    terms, _ = _kernel_model(eqn.model, mesh)
    (; cells, faces, cell_faces, cell_neighbours, cell_nsign) = mesh
    ndrange = length(cells)
    kernel! = _sized(_assemble_matrix_scalar!, backend, workgroup, ndrange)
    kernel!(terms, cells, faces, cell_faces, cell_neighbours, cell_nsign, nzval,
        diag_nz, face_nz, _kernel_field(get_phi(eqn)), runtime,
        _kernel_field(_get_flux(eqn.model.terms[1])))
end

@kernel function _assemble_matrix_scalar!(
    terms::TERMS, cells, faces, cell_faces, cell_neighbours, cell_nsign,
    nzval::AbstractArray{F}, diag_nz, face_nz, prev, runtime, rho_prev) where {F,TERMS}
    i = @index(Global)
    @inbounds begin
        faces_range = cells.faces_range[i]
        cIndex = diag_nz[i]
        ac_sum = zero(F)
        for fi in faces_range
            fID = cell_faces[fi]
            ns = cell_nsign[fi]
            nID = cell_neighbours[fi]
            nIndex = face_nz[fi]
            ac, an, _ = _scheme!(terms, nzval, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)
            ac_sum += ac
            nzval[nIndex] = an
        end
        ac, _ = _scheme_source!(terms, cells, i, cIndex, prev, runtime, rho_prev)
        nzval[cIndex] = ac_sum + ac
    end
end

function assemble_rhs!(
    eqn::ModelEquation{T,M,E,S,P}, source::AbstractSource, config
    ) where {T<:ScalarModel,M,E,S,P}
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware
    mesh = get_phi(eqn).mesh
    b = _b(eqn)
    z = zero(eltype(b))
    xcal_foreach(b, config) do i
        b[i] = z
    end
    terms, _ = _kernel_model(eqn.model, mesh)
    sources = (Src(_kernel_field(source.field), source.sign),)
    (; cells, faces, cell_faces, cell_neighbours, cell_nsign) = mesh
    (; diag_nz, face_nz) = eqn.equation
    ndrange = length(cells)
    kernel! = _sized(_assemble_rhs_scalar!, backend, workgroup, ndrange)
    kernel!(terms, sources, cells, faces, cell_faces, cell_neighbours, cell_nsign,
        diag_nz, face_nz, b, _kernel_field(get_phi(eqn)), runtime,
        _kernel_field(_get_flux(eqn.model.terms[1])))
end

@kernel function _assemble_rhs_scalar!(
    terms::TERMS, sources::SRCS, cells, faces, cell_faces, cell_neighbours, cell_nsign,
    diag_nz, face_nz, b::AbstractArray{F}, prev, runtime, rho_prev) where {F,TERMS,SRCS}
    i = @index(Global)
    @inbounds begin
        faces_range = cells.faces_range[i]
        volume = cells.volume[i]
        cIndex = diag_nz[i]
        b_face = zero(F)
        for fi in faces_range
            fID = cell_faces[fi]
            ns = cell_nsign[fi]
            nID = cell_neighbours[fi]
            nIndex = face_nz[fi]
            _, _, bf = _scheme!(terms, b, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)
            b_face += bf
        end
        _, b1 = _scheme_source!(terms, cells, i, cIndex, prev, runtime, rho_prev)
        b2 = _sources!(sources, volume, i)
        b[i] = b2 + b1 + b_face
    end
end

function explicit_residual!(
    r::AbstractVector, eqn::ModelEquation{T,M,E,S,P}, phi, config
    ) where {T<:ScalarModel,M,E,S,P}
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware
    mesh = get_phi(eqn).mesh
    terms, sources = _kernel_model(eqn.model, mesh)
    (; cells, faces, cell_faces, cell_neighbours, cell_nsign) = mesh
    (; diag_nz, face_nz) = eqn.equation
    ndrange = length(cells)
    kernel! = _sized(_explicit_residual_scalar!, backend, workgroup, ndrange)
    kernel!(terms, sources, cells, faces, cell_faces, cell_neighbours, cell_nsign,
        diag_nz, face_nz, r, _kernel_field(phi), runtime,
        _kernel_field(_get_flux(eqn.model.terms[1])))
    KernelAbstractions.synchronize(backend)
    return r
end

@kernel function _explicit_residual_scalar!(
    terms::TERMS, sources::SRCS, cells, faces, cell_faces, cell_neighbours, cell_nsign,
    diag_nz, face_nz, r::AbstractArray{F}, prev, runtime, rho_prev) where {F,TERMS,SRCS}
    i = @index(Global)
    @inbounds begin
        faces_range = cells.faces_range[i]
        volume = cells.volume[i]
        cIndex = diag_nz[i]
        ac_sum = zero(F)
        an_phi = zero(F)
        b_face = zero(F)
        for fi in faces_range
            fID = cell_faces[fi]
            ns = cell_nsign[fi]
            nID = cell_neighbours[fi]
            nIndex = face_nz[fi]
            ac, an, bf = _scheme!(terms, r, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)
            ac_sum += ac
            an_phi += an * prev[nID]
            b_face += bf
        end
        ac, b1 = _scheme_source!(terms, cells, i, cIndex, prev, runtime, rho_prev)
        b2 = _sources!(sources, volume, i)
        r[i] = (ac_sum + ac) * prev[i] + an_phi - (b_face + b1 + b2)
    end
end
