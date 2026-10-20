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
_kernel_field(t::TensorField, m=()) = TensorField(
    _kernel_field(t.xx), _kernel_field(t.xy), _kernel_field(t.xz),
    _kernel_field(t.yx), _kernel_field(t.yy), _kernel_field(t.yz),
    _kernel_field(t.zx), _kernel_field(t.zy), _kernel_field(t.zz), m)

_kernel_term(t::Operator, m) = Operator(
    _kernel_field(t.flux), _kernel_field(t.phi, m), t.sign, _kernel_type(t.type), _kernel_field(t.grad))
_kernel_term(t::NonlinearOperator, m) = NonlinearOperator(_kernel_term(t.op, m), t.map)
_kernel_term(t::AffineOperator, m) = AffineOperator(
    _kernel_term(t.op, m), _kernel_field(t.jacobian), _kernel_field(t.offset),
    _kernel_field(t.reference), t.map)

# same term with its operator type in kernel form (fields untouched), keeping nonlinear wrappers
_typed_term(t::Operator) = Operator(t.flux, t.phi, t.sign, _kernel_type(t.type), t.grad)
_typed_term(t::NonlinearOperator) = NonlinearOperator(_typed_term(t.op), t.map)
_typed_term(t::AffineOperator) = AffineOperator(
    _typed_term(t.op), t.jacobian, t.offset, t.reference, t.map)

_kernel_model(model, mesh) = begin
    m = (; face_gDiff=mesh.face_gDiff)
    terms = map(t -> _kernel_term(t, m), model.terms)
    sources = map(s -> Src(_kernel_field(s.field), s.sign), model.sources)
    terms, sources
end

function discretise!(
    eqn::ModelEquation{T,M,E,S,P}, prev, config; rho_prev=_get_flux(eqn.model.terms[1]), time=nothing
    ) where {T<:VectorModel,M,E,S,P}
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware
    prev = time_level!(eqn.model.terms[1], prev, rho_prev, time, config)

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

    _sync_gradients!(model.terms, mesh, config)
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
        bf = _face_source_init(terms, zero(SVector{3,F}))
        for fi in faces_range
            # Retrieve indices for discretisation
            fID = cell_faces[fi]
            ns = cell_nsign[fi] # normal sign
            nID = cell_neighbours[fi]
            nIndex = face_nz[fi]


            # Call scheme generated fucntion
            ac, an = _scheme!(terms, nzval0, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)
            ac_sum += ac
            nzval0[nIndex] = an
            bf = _scheme_face_source!(terms, bf, cells, faces, i, nID, ns, fID)
        end

        
        # Call scheme source generated function NEEDS UPDATING!
        ac, bx1, by1, bz1 = _scheme_source!(terms, cells, i, cIndex, prev, runtime, rho_prev)
        
        nzval0[cIndex] = ac_sum + ac

        # Call sources generated function
        bx2, by2, bz2 = _sources!(sources, volume, i)
        bx[i] = _add_face_source(bx1 + bx2, bf, 1)
        by[i] = _add_face_source(by1 + by2, bf, 2)
        bz[i] = _add_face_source(bz1 + bz2, bf, 3)
    end
end

function discretise!(
    eqn::ModelEquation{T,M,E,S,P}, prev, config; rho_prev=_get_flux(eqn.model.terms[1]), time=nothing
    ) where {T<:ScalarModel,M,E,S,P}

    (; hardware, runtime) = config
    (; backend, workgroup) = hardware
    prev = time_level!(eqn.model.terms[1], prev, rho_prev, time, config)

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

    _sync_gradients!(model.terms, mesh, config)
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
        bf = _face_source_init(terms, zero(F))
        for fi in faces_range
            # Retrieve indices for discretisation
            fID = cell_faces[fi]
            ns = cell_nsign[fi] # normal sign
            nID = cell_neighbours[fi]
            nIndex = face_nz[fi]

            # Call scheme generated fucntion
            ac, an = _scheme!(terms, nzval, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)
            ac_sum += ac
            nzval[nIndex] = an
            bf = _scheme_face_source!(terms, bf, cells, faces, i, nID, ns, fID)
        end
        
        # Call scheme source generated function
        ac, b1 = _scheme_source!(terms, cells, i, cIndex, prev, runtime, rho_prev)
        nzval[cIndex] = ac_sum + ac

        # Call sources generated function
        b2 = _sources!(sources, volume, i)
        b[i] = _add_face_source(b2 + b1, bf, 1)
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
            AC += F(ac)
            AN += F(an)
        end
        push!(out.args, function_call_scheme)
    end
    # out
    quote
        z = zero(F)
        AC = z
        AN = z
        $(out.args...)
        return AC, AN
    end
end

# Face source: `bf` accumulates the explicit face contributions of the operators with `has_face_source`
# (SVector{3} for vector, scalar otherwise); it is `nothing` when no operator has one, so equations
# without second-order schemes carry no extra work.
@generated function _face_source_init(terms::TERMS, z) where {TERMS}
    any(t -> has_face_source(fieldtype(TERMS, t)), 1:fieldcount(TERMS)) ? :(z) : :(nothing)
end

@generated function _scheme_face_source!(
    terms::TERMS, bf, cells, faces, cID, nID, ns, fID) where {TERMS}
    idx = [t for t in 1:fieldcount(TERMS) if has_face_source(fieldtype(TERMS, t))]
    isempty(idx) && return :(bf)
    calls = [:(bf += scheme_face_source!(terms[$t], cells, faces, cID, nID, ns, fID)) for t in idx]
    quote
        $(calls...)
        return bf
    end
end

@inline _add_face_source(b, ::Nothing, k) = b
@inline _add_face_source(b, bf::AbstractVector, k) = b + bf[k]
@inline _add_face_source(b, bf::Number, k) = b + bf

# Ghost gradients are not consistent on other ranks; distributed meshes extend this hook
sync_gradient!(grad, mesh, config) = nothing

@generated function _sync_gradients!(terms::TERMS, mesh, config) where {TERMS}
    idx = [t for t in 1:fieldcount(TERMS)
        if has_face_source(fieldtype(TERMS, t)) && hasfield(fieldtype(TERMS, t), :grad)]
    isempty(idx) && return :(nothing)
    calls = [:(sync_gradient!(terms[$t].grad, mesh, config)) for t in idx]
    :($(calls...); nothing)
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
# same _scheme! / _scheme_source! / _scheme_face_source! used by discretise!.
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
            ac, an = _scheme!(terms, nzval, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)
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
    _sync_gradients!(eqn.model.terms, mesh, config)
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
        bf = _face_source_init(terms, zero(F))
        for fi in faces_range
            bf = _scheme_face_source!(
                terms, bf, cells, faces, i, cell_neighbours[fi], cell_nsign[fi], cell_faces[fi])
        end
        _, b1 = _scheme_source!(terms, cells, i, cIndex, prev, runtime, rho_prev)
        b2 = _sources!(sources, volume, i)
        b[i] = _add_face_source(b2 + b1, bf, 1)
    end
end

function explicit_residual!(
    r::AbstractVector, eqn::ModelEquation{T,M,E,S,P}, phi, config
    ) where {T<:ScalarModel,M,E,S,P}
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware
    mesh = get_phi(eqn).mesh
    _sync_gradients!(eqn.model.terms, mesh, config)
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
        bf = _face_source_init(terms, zero(F))
        for fi in faces_range
            fID = cell_faces[fi]
            ns = cell_nsign[fi]
            nID = cell_neighbours[fi]
            nIndex = face_nz[fi]
            ac, an = _scheme!(terms, r, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)
            ac_sum += ac
            an_phi += an * prev[nID]
            bf = _scheme_face_source!(terms, bf, cells, faces, i, nID, ns, fID)
        end
        ac, b1 = _scheme_source!(terms, cells, i, cIndex, prev, runtime, rho_prev)
        b2 = _sources!(sources, volume, i)
        r[i] = (ac_sum + ac) * prev[i] + an_phi - _add_face_source(b1 + b2, bf, 1)
    end
end
