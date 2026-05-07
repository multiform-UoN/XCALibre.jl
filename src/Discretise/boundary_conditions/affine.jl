@inline function _affine_inner_operator(term::AffineOperator)
    Operator(term.op.flux, term.reference, term.op.sign, term.op.type)
end

@inline function _affine_mapped_value(term::AffineOperator, value::Number)
    term.map(value)
end

@inline function _affine_mapped_value(term::AffineOperator, value)
    error("Affine nonlinear operators currently support scalar boundary values only. Got $(typeof(value)).")
end

@inline function _affine_apply_identity_boundary(
    bc, term::AffineOperator, colval, rowptr, nzval,
    cellID, zcellID, cell, face, fID, i, component, time
)
    inner = _affine_inner_operator(term)
    ap, bp = bc(inner, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time)
    jac = term.jacobian[cellID]
    off = term.offset[cellID]
    return ap * jac, bp - ap * off
end

@inline function _affine_apply_mapped_dirichlet(
    bc::Dirichlet, term::AffineOperator, colval, rowptr, nzval,
    cellID, zcellID, cell, face, fID, i, component, time
)
    mapped_bc = Dirichlet(bc.ID, _affine_mapped_value(term, bc.value), bc.IDs_range)
    return _affine_apply_identity_boundary(
        mapped_bc, term, colval, rowptr, nzval,
        cellID, zcellID, cell, face, fID, i, component, time
    )
end

@inline function _affine_apply_mapped_wall(
    bc::Wall{I,V,R}, term::AffineOperator, colval, rowptr, nzval,
    cellID, zcellID, cell, face, fID, i, component, time
) where {I,V<:Number,R}
    mapped_bc = Wall(bc.ID, _affine_mapped_value(term, bc.value), bc.IDs_range)
    return _affine_apply_identity_boundary(
        mapped_bc, term, colval, rowptr, nzval,
        cellID, zcellID, cell, face, fID, i, component, time
    )
end

@inline (bc::Dirichlet)(
    term::AffineOperator, colval, rowptr, nzval,
    cellID, zcellID, cell, face, fID, i, component, time
) = _affine_apply_mapped_dirichlet(
    bc, term, colval, rowptr, nzval,
    cellID, zcellID, cell, face, fID, i, component, time
)

@inline (bc::Extrapolated)(
    term::AffineOperator, colval, rowptr, nzval,
    cellID, zcellID, cell, face, fID, i, component, time
) = _affine_apply_identity_boundary(
    bc, term, colval, rowptr, nzval,
    cellID, zcellID, cell, face, fID, i, component, time
)

@inline (bc::Zerogradient)(
    term::AffineOperator, colval, rowptr, nzval,
    cellID, zcellID, cell, face, fID, i, component, time
) = _affine_apply_identity_boundary(
    bc, term, colval, rowptr, nzval,
    cellID, zcellID, cell, face, fID, i, component, time
)

@inline (bc::Symmetry)(
    term::AffineOperator, colval, rowptr, nzval,
    cellID, zcellID, cell, face, fID, i, component, time
) = _affine_apply_identity_boundary(
    bc, term, colval, rowptr, nzval,
    cellID, zcellID, cell, face, fID, i, component, time
)

@inline (bc::Outlet)(
    term::AffineOperator, colval, rowptr, nzval,
    cellID, zcellID, cell, face, fID, i, component, time
) = _affine_apply_identity_boundary(
    bc, term, colval, rowptr, nzval,
    cellID, zcellID, cell, face, fID, i, component, time
)

@inline (bc::Wall{I,V,R})(
    term::AffineOperator, colval, rowptr, nzval,
    cellID, zcellID, cell, face, fID, i, component, time
) where {I,V<:Number,R} = _affine_apply_mapped_wall(
    bc, term, colval, rowptr, nzval,
    cellID, zcellID, cell, face, fID, i, component, time
)

@inline function (bc::AbstractBoundary)(
    term::AffineOperator, colval, rowptr, nzval,
    cellID, zcellID, cell, face, fID, i, component, time
)
    error("Boundary $(typeof(bc)) is not supported for affine nonlinear operators yet.")
end
