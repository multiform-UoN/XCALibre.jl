# Boundary conditions for conservative Gauss ScalarGrad/VectorDiv coupling.
#
# These operators assemble volume-integrated face fluxes of the form
#   coeff * phi_f, coeff = sign * flux[f] * Sf[component].
# Boundary normals are stored outward-facing for the boundary cell, so no
# additional owner/neighbour sign is needed here.

@inline _boundary_face_value(bc::Dirichlet, face, time, i, component) = bc.value
@inline _boundary_face_value(bc::DirichletFunction, face, time, i, component) =
    bc.value(face.centre, time, i)

@inline function _scalargrad_coeff(term, face, fID, ::Val{I_ROW}) where {I_ROW}
    return term.sign * term.flux[fID] * face.area * face.normal[I_ROW]
end

@inline function _vectordiv_coeff(term, face, fID, ::Val{J_COL}) where {J_COL}
    return term.sign * term.flux[fID] * face.area * face.normal[J_COL]
end

@inline function (bc::Dirichlet)(
    term::Operator{F,P,S,ScalarGrad{T,I_ROW}},
    colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
) where {F,P,S,T,I_ROW}
    coeff = _scalargrad_coeff(term, face, fID, Val(I_ROW))
    return 0.0, -coeff * _boundary_face_value(bc, face, time, i, component)
end

@inline function (bc::DirichletFunction)(
    term::Operator{F,P,S,ScalarGrad{T,I_ROW}},
    colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
) where {F,P,S,T,I_ROW}
    coeff = _scalargrad_coeff(term, face, fID, Val(I_ROW))
    return 0.0, -coeff * _boundary_face_value(bc, face, time, i, component)
end

@inline function (bc::Dirichlet)(
    term::Operator{F,P,S,VectorDiv{T,J_COL}},
    colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
) where {F,P,S,T,J_COL}
    coeff = _vectordiv_coeff(term, face, fID, Val(J_COL))
    return 0.0, -coeff * _boundary_face_value(bc, face, time, i, component)
end

@inline function (bc::DirichletFunction)(
    term::Operator{F,P,S,VectorDiv{T,J_COL}},
    colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
) where {F,P,S,T,J_COL}
    coeff = _vectordiv_coeff(term, face, fID, Val(J_COL))
    return 0.0, -coeff * _boundary_face_value(bc, face, time, i, component)
end

@inline function (bc::Union{Zerogradient,Extrapolated,Outlet,Symmetry})(
    term::Operator{F,P,S,ScalarGrad{T,I_ROW}},
    colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
) where {F,P,S,T,I_ROW}
    coeff = _scalargrad_coeff(term, face, fID, Val(I_ROW))
    return coeff, 0.0
end

@inline function (bc::Union{Zerogradient,Extrapolated,Outlet,Symmetry})(
    term::Operator{F,P,S,VectorDiv{T,J_COL}},
    colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
) where {F,P,S,T,J_COL}
    coeff = _vectordiv_coeff(term, face, fID, Val(J_COL))
    return coeff, 0.0
end

@inline function (bc::Empty)(
    term::Operator{F,P,S,ScalarGrad{T,I_ROW}},
    colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
) where {F,P,S,T,I_ROW}
    return 0.0, 0.0
end

@inline function (bc::Empty)(
    term::Operator{F,P,S,VectorDiv{T,J_COL}},
    colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
) where {F,P,S,T,J_COL}
    return 0.0, 0.0
end
