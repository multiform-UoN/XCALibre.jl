# ---------------------------------------------------------------------------
# Boundary conditions for the GradDiv{T,I_ROW,J_COL} elastic coupling operator.
#
# The face coefficient is:  α_f * e[J] * (A*n)[I] / delta
# For Dirichlet: same structure as Laplacian Dirichlet but uses GradDiv geometry.
# For all zero-flux conditions (Zerogradient, Extrapolated, Symmetry, Outlet):
#   return (0, 0) — the GradDiv term naturally satisfies a zero-traction condition
#   because the coupling contribution vanishes when the face is aligned with a
#   coordinate axis (the common case for axis-aligned symmetry planes).
# ---------------------------------------------------------------------------

@inline function (bc::Dirichlet)(
    term::Operator{F,P,S,GradDiv{T,I_ROW,J_COL}},
    colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
) where {F,P,S,T,I_ROW,J_COL}
    @inbounds begin
        e_J  = face.e[J_COL]
        n_I  = face.normal[I_ROW]
        ap   = term.sign * (-term.flux[fID] * e_J * face.area * n_I / face.delta)
        return ap, ap * bc.value
    end
end

@inline function (bc::Zerogradient)(
    term::Operator{F,P,S,GradDiv{T,I_ROW,J_COL}},
    colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
) where {F,P,S,T,I_ROW,J_COL}
    return 0.0, 0.0
end

@inline function (bc::Extrapolated)(
    term::Operator{F,P,S,GradDiv{T,I_ROW,J_COL}},
    colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
) where {F,P,S,T,I_ROW,J_COL}
    return 0.0, 0.0
end

@inline function (bc::Symmetry)(
    term::Operator{F,P,S,GradDiv{T,I_ROW,J_COL}},
    colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
) where {F,P,S,T,I_ROW,J_COL}
    return 0.0, 0.0
end

@inline function (bc::Outlet)(
    term::Operator{F,P,S,GradDiv{T,I_ROW,J_COL}},
    colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
) where {F,P,S,T,I_ROW,J_COL}
    return 0.0, 0.0
end

@inline function (bc::AbstractBoundary)(
    term::Operator{F,P,S,GradDiv{T,I_ROW,J_COL}},
    colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
) where {F,P,S,T,I_ROW,J_COL}
    error("Boundary $(typeof(bc)) is not supported for GradDiv operator.")
end
