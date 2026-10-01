export scheme!, scheme_source!

#= NOTE:
In source scheme the following indices are used and should be used with care:
cID - Index of the cell outer loop. Use to index "b" 
cIndex - Index of the cell based on sparse matrix. Use to index "nzval_array"
=#

# TIME 

# SteadyState
@inline function scheme!(
    term::Operator{F,P,I,Time{SteadyState}}, 
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P,I}
    # nothing
    z = zero(eltype(nzval_array))
    z, z
end
@inline scheme_source!(
    term::Operator{F,P,I,Time{SteadyState}}, cells, cID, cIndex, prev, runtime, rho_prev)  where {F,P,I} = begin
    z = zero(eltype(cells.volume))
    z, z
end

## Euler
@inline function scheme!(
    term::Operator{F,P,I,Time{Euler}}, 
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P,I}
    0.0, 0.0 # add types if this approach works
end

@inline scheme_source!(
    term::Operator{F,P,I,Time{Euler}}, cells, cID, cIndex, prev, runtime, rho_prev)  where {F,P<:ScalarField,I} = begin
        volume = cells.volume[cID]
        vol_rdt = volume/runtime.dt[1]
        rho = term.flux[cID]
        
        ac = rho * vol_rdt
        b = rho_prev[cID]*prev[cID]*vol_rdt
        return ac, b
end
@inline scheme_source!(
    term::Operator{F,P,I,Time{Euler}}, cells, cID, cIndex, prev, runtime, rho_prev)  where {F,P<:VectorField,I} = begin # Special case for U_eqn (rho)
        volume = cells.volume[cID]
        vol_rdt = volume/runtime.dt[1]
        rho = term.flux[cID]
        
        # Increment sparse and b arrays 
        ac = rho * vol_rdt
        b = rho_prev[cID]*prev[cID]*vol_rdt
        return ac, b
end

## Crank-Nicholson
@inline function scheme!(
    term::Operator{F,P,I,Time{CrankNicolson}}, 
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P,I}

    0.0, 0.0 # add types if this approach works
end
@inline scheme_source!(
    term::Operator{F,P,I,Time{CrankNicolson}}, cells, cID, cIndex, prev, runtime, rho_prev)  where {F,P<:ScalarField,I} = begin
        volume = cells.volume[cID]
        vol_rdt = volume/runtime.dt[1]
        rho = term.flux[cID]
        
        ac = rho * vol_rdt
        b = rho_prev[cID]*prev[cID]*vol_rdt # Careful with non U_eqn (e.g. T eqn.)
        return ac, b
end
@inline scheme_source!(
    term::Operator{F,P,I,Time{CrankNicolson}}, cells, cID, cIndex, prev, runtime, rho_prev)  where {F,P<:VectorField,I} = begin
        volume = cells.volume[cID]
        vol_rdt = volume/runtime.dt[1]
        rho = term.flux[cID]
        
        ac = rho * vol_rdt
        b = rho_prev[cID]*prev[cID]*vol_rdt
        return ac, b
end

# LAPLACIAN

@inline function scheme!(
    term::Operator{F,P,I,Laplacian{Linear}}, 
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}

    (; face_gDiff) = term.phi.mesh
    ap = term.sign*term.flux[fID]*face_gDiff[fID]

    # Increment sparse array
    ac = -ap
    an = ap
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Laplacian{Linear}}, cells, cID, cIndex, prev, runtime, rho_prev)  where {F,P,I} = begin
    0.0, 0.0
end

# DIVERGENCE

# Linear
@inline function scheme!(
    term::Operator{F,P,I,Divergence{Linear}}, 
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}

    w = faces.weight[fID]
    # signbit(ns) ? w = one(w) - w : w
    half = typeof(w)(0.5)
    w = half + ns*(w - half)
    
    # Calculate link coefficients
    ap = term.sign*(term.flux[fID]*ns)
    ac = ap*w
    an = ap*(one(w) - w)
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{Linear}}, cells, cID, cIndex, prev, runtime, rho_prev) where {F,P,I} = begin
    0.0, 0.0
end

# Upwind
@inline function scheme!(
    term::Operator{F,P,I,Divergence{Upwind}}, 
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    # Calculate link coefficients
    ap = term.sign*(term.flux[fID]*ns)
    z = zero(ap)
    ac = max(ap, z)
    an = -max(-ap, z)
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{Upwind}}, cells, cID, cIndex, prev, runtime, rho_prev) where {F,P,I} = begin
    0.0, 0.0
end

# LUST
@inline function scheme!(
    term::Operator{F,P,I,Divergence{LUST}}, 
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    
    w = faces.weight[fID]
    signbit(ns) ? w = one(w) - w : w

    # Calculate link coefficients
    ap = term.sign*(term.flux[fID]*ns)
    acLinear = ap*w 
    anLinear = ap*(one(w) - w)
    z = zero(ap)
    acUpwind = max(ap, z)
    anUpwind = -max(-ap, z)
    three_quarters = typeof(ap)(0.75)
    quarter = typeof(ap)(0.25)
    ac = three_quarters*acLinear + quarter*acUpwind
    an = three_quarters*anLinear + quarter*anUpwind
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{LUST}}, cells, cID, cIndex, prev, runtime, rho_prev) where {F,P,I} = begin
    0.0, 0.0
end

# BoundedUpwind
@inline function scheme!(
    term::Operator{F,P,I,Divergence{BoundedUpwind}}, 
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    # $$\mathcal{D}_{bounded} = \sum_f \phi_f \psi_f - \psi_P \sum_f \phi_f$$
    # phif =  max(phif, 0) - max(-phi_f, 0)$
    # phif psif =  max(phif, 0) psi_P - max(-phi_f, 0)$ psi_N
    ap = term.sign*(term.flux[fID]*ns)
    z = zero(ap)
    ac = max(-ap, z)
    an = -max(-ap, z)
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{BoundedUpwind}}, cells, cID, cIndex, prev, runtime, rho_prev) where {F,P,I} = begin
    0.0, 0.0
end


# IMPLICIT SOURCE
@inline function scheme!(
    term::Operator{F,P,I,Si}, 
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    z = zero(eltype(nzval_array))
    z, z
end
@inline scheme_source!(
    term::Operator{F,P,I,Si}, cells, cID, cIndex, prev, runtime, rho_prev)  where {F,P,I} = begin
    
    # Retrieve and calculate flux for cell 
    flux = term.sign*term.flux[cID]*cells.volume[cID] # indexed with cID
    ac = flux # indexed with cIndex
    ac, zero(ac)
end

# Mathematical operators used by the PDE layer. They follow the upstream
# scheme!(term, nzval, cells, faces, nID, ...) and scheme_source!(term, cells, cID, ...)
# calling convention. Face geometry is read from the columnar face array.

@inline _owner_cell(faces, fID, nID) = begin
    owners = faces.ownerCells[fID]
    owners[1] == nID ? owners[2] : owners[1]
end

@inline _scheme_face_rhs(term, nzval, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime) =
    zero(eltype(nzval))

@inline function scheme!(term::NonlinearOperator, nzval, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)
    scheme!(term.op, nzval, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)
end
@inline function scheme_source!(term::NonlinearOperator, cells, cID, cIndex, prev, runtime, rho_prev)
    scheme_source!(term.op, cells, cID, cIndex, prev, runtime, rho_prev)
end

@inline function scheme!(
    term::AffineOperator, nzval, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)
    ac, an = scheme!(term.op, nzval, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)
    cID = _owner_cell(faces, fID, nID)
    return ac * term.jacobian[cID], an * term.jacobian[nID]
end

@inline function _scheme_face_rhs(
    term::AffineOperator, nzval, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)
    ac, an = scheme!(term.op, nzval, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)
    cID = _owner_cell(faces, fID, nID)
    return -(ac * term.offset[cID] + an * term.offset[nID])
end

@inline function scheme_source!(
    term::AffineOperator, cells, cID, cIndex, prev, runtime, rho_prev)
    ac, b = scheme_source!(term.op, cells, cID, cIndex, term.reference, runtime, rho_prev)
    return ac * term.jacobian[cID], b - ac * term.offset[cID]
end

# Monolithic assembly still walks one cell and one face at a time. Bridge that
# loop onto the upstream scheme! so both paths share the same coefficients.
@inline function scheme_contribution!(
    term, nzval, cell, face, cellN, ns, cID, nID, cIndex, nIndex, fID, prev, runtime)
    mesh = _term_phi(term).mesh
    ac, an = scheme!(term, nzval, mesh.cells, mesh.faces, nID, ns, cIndex, nIndex, fID, prev, runtime)
    bface = _scheme_face_rhs(term, nzval, mesh.cells, mesh.faces, nID, ns, cIndex, nIndex, fID, prev, runtime)
    return ac, an, bface
end

# BIHARMONIC. Orthogonal-mesh stencil: area/delta^2, no non-orthogonal correction.
@inline function scheme!(
    term::Operator{F,P,I,Biharmonic{T}},
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime
    ) where {F,P,I,T}
    coeff = term.flux[fID] * faces.area[fID] / faces.delta[fID]
    ac = coeff / faces.delta[fID]
    return ac, -ac
end
@inline scheme_source!(
    term::Operator{F,P,I,Biharmonic{T}}, cells, cID, cIndex, prev, runtime, rho_prev
    ) where {F,P,I,T} = (0.0, 0.0)

# GRADDIV. Two-point elastic coupling: sign * flux * e[J] * (A n)[I] / delta.
@inline function scheme!(
    term::Operator{F,P,I,GradDiv{T,I_ROW,J_COL}},
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime
) where {F,P,I,T,I_ROW,J_COL}
    face = faces[fID]
    ap = term.sign * term.flux[fID] * face.e[J_COL] * face.area * face.normal[I_ROW] / face.delta
    return -ap, ap
end
@inline scheme_source!(
    term::Operator{F,P,I,GradDiv{T,I_ROW,J_COL}}, cells, cID, cIndex, prev, runtime, rho_prev
) where {F,P,I,T,I_ROW,J_COL} = (0.0, 0.0)

# SCALARGRAD. Gauss integral of d(phi)/dx_I, linear face interpolation.
@inline function scheme!(
    term::Operator{F,P,I,ScalarGrad{T,I_ROW}},
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime
) where {F,P,I,T,I_ROW}
    face = faces[fID]
    w = typeof(face.weight)(0.5) + ns * (face.weight - typeof(face.weight)(0.5))
    Sf_I = ns * face.area * face.normal[I_ROW]
    ap = term.sign * term.flux[fID] * Sf_I
    return ap * w, ap * (one(w) - w)
end
@inline scheme_source!(
    term::Operator{F,P,I,ScalarGrad{T,I_ROW}}, cells, cID, cIndex, prev, runtime, rho_prev
) where {F,P,I,T,I_ROW} = (0.0, 0.0)

# VECTORDIV. Gauss integral of d(u_J)/dx_J.
@inline function scheme!(
    term::Operator{F,P,I,VectorDiv{T,J_COL}},
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime
) where {F,P,I,T,J_COL}
    face = faces[fID]
    w = typeof(face.weight)(0.5) + ns * (face.weight - typeof(face.weight)(0.5))
    Sf_J = ns * face.area * face.normal[J_COL]
    ap = term.sign * term.flux[fID] * Sf_J
    return ap * w, ap * (one(w) - w)
end
@inline scheme_source!(
    term::Operator{F,P,I,VectorDiv{T,J_COL}}, cells, cID, cIndex, prev, runtime, rho_prev
) where {F,P,I,T,J_COL} = (0.0, 0.0)

@inline function scheme!(
    term::Operator{F,P,I,CoupledSi},
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime
    ) where {F,P,I}
    z = zero(eltype(nzval_array))
    z, z
end
@inline function scheme_source!(
    term::Operator{F,P,I,CoupledSi}, cells, cID, cIndex, prev, runtime, rho_prev
) where {F,P,I}
    ac = term.sign * term.flux[cID] * cells.volume[cID]
    return ac, zero(ac)
end

# NONLINEAR IMPLICIT SOURCE. linearize_physics normally replaces this with Si
# before discretise!. The explicit value is kept for residual and Picard paths.
@inline function scheme!(
    term::Operator{F,P,I,NonLinearSi{Fun}},
    nzval_array, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime
) where {F,P,I,Fun}
    z = zero(eltype(nzval_array))
    z, z
end
@inline function scheme_source!(
    term::Operator{F,P,I,NonLinearSi{Fun}}, cells, cID, cIndex, prev, runtime, rho_prev
) where {F,P,I,Fun}
    val = prev[cID]
    b = -term.sign * term.type.func(val) * cells.volume[cID]
    return zero(val), b
end
