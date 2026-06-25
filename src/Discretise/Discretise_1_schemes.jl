export scheme!, scheme_source!

#= NOTE:
In source scheme the following indices are used and should be used with care:
cID - Index of the cell outer loop. Use to index "b"
cIndex - Index of the cell based on sparse matrix. Use to index "nzval_array"
=#

@inline function scheme_contribution!(
    term::Operator,
    nzval_array, cell, face, cellN, ns, cID, nID, cIndex, nIndex, fID, prev, runtime
)
    ac, an = scheme!(term, nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime)
    return ac, an, zero(ac + an)
end

@inline function scheme_contribution!(
    term::AffineOperator,
    nzval_array, cell, face, cellN, ns, cID, nID, cIndex, nIndex, fID, prev, runtime
)
    ac, an = scheme!(term.op, nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime)
    jac_c = term.jacobian[cID]
    jac_n = term.jacobian[nID]
    off_c = term.offset[cID]
    off_n = term.offset[nID]
    return ac * jac_c, an * jac_n, -(ac * off_c + an * off_n)
end

@inline function scheme_source!(
    term::AffineOperator, cell, cID, cIndex, prev, runtime
)
    ac, b = scheme_source!(term.op, cell, cID, cIndex, term.reference, runtime)
    return ac * term.jacobian[cID], b - ac * term.offset[cID]
end

@inline function scheme_source!(
    term::AffineOperator, cell, cID, cIndex, prev, runtime, rho_prev
)
    ac, b = scheme_source!(term.op, cell, cID, cIndex, term.reference, runtime, rho_prev)
    return ac * term.jacobian[cID], b - ac * term.offset[cID]
end

# Fallback for operators that do not require rho_prev
@inline scheme_source!(
    term::Operator, cell, cID, cIndex, prev, runtime, rho_prev
) = scheme_source!(term, cell, cID, cIndex, prev, runtime)

# TIME

# SteadyState
@inline function scheme!(
    term::Operator{F,P,I,TimeTerm{SteadyState}},
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P,I}
    # nothing
    0.0, 0.0 # add types if this approach works
end
@inline scheme_source!(
    term::Operator{F,P,I,TimeTerm{SteadyState}}, cell, cID, cIndex, prev, runtime, rho_prev)  where {F,P,I} = begin
    z = zero(cell.volume)
    z, z
end

## Euler
@inline function scheme!(
    term::Operator{F,P,I,TimeTerm{Euler}},
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P,I}

    0.0, 0.0 # add types if this approach works
end
@inline scheme_source!(
    term::Operator{F,P,I,TimeTerm{Euler}}, cell, cID, cIndex, prev, runtime, rho_prev)  where {F,P<:ScalarField,I} = begin
        volume = cell.volume
        vol_rdt = volume/runtime.dt[1]
        rho = term.flux[cID]

        ac = rho * vol_rdt
        b = rho_prev[cID]*prev[cID]*vol_rdt
        return ac, b
end
@inline scheme_source!(
    term::Operator{F,P,I,TimeTerm{Euler}}, cell, cID, cIndex, prev, runtime, rho_prev)  where {F,P<:VectorField,I} = begin # Special case for U_eqn (rho)
        volume = cell.volume
        vol_rdt = volume/runtime.dt[1]
        rho = term.flux[cID]

        # Increment sparse and b arrays
        ac = rho * vol_rdt
        b = rho_prev[cID]*prev[cID]*vol_rdt
        return ac, b
end

## Crank-Nicholson
@inline function scheme!(
    term::Operator{F,P,I,TimeTerm{CrankNicolson}},
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P,I}

    0.0, 0.0 # add types if this approach works
end
@inline scheme_source!(
    term::Operator{F,P,I,TimeTerm{CrankNicolson}}, cell, cID, cIndex, prev, runtime, rho_prev)  where {F,P<:ScalarField,I} = begin
        volume = cell.volume
        vol_rdt = volume/runtime.dt[1]
        rho = term.flux[cID]

        ac = rho * vol_rdt
        b = rho_prev[cID]*prev[cID]*vol_rdt # Careful with non U_eqn (e.g. T eqn.)
        return ac, b
end
@inline scheme_source!(
    term::Operator{F,P,I,TimeTerm{CrankNicolson}}, cell, cID, cIndex, prev, runtime, rho_prev)  where {F,P<:VectorField,I} = begin
        volume = cell.volume
        vol_rdt = volume/runtime.dt[1]
        rho = term.flux[cID]

        ac = rho * vol_rdt
        b = rho_prev[cID]*prev[cID]*vol_rdt
        return ac, b
end

# LAPLACIAN

@inline function scheme!(
    term::Operator{F,P,I,Laplacian{Linear}},
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}


    (; area, normal, delta, e) = face
    Sf = ns*area*normal
    Af = norm(Sf)

    ## Potential simplified form for performance, needs checking before use in release
    # dPN = cellN.centre - cell.centre
    # n = ns*normal
    # Ef = dPN*(norm(n)^2/(dPN⋅n))*area # this works
    # Ef = dPN*(one(typeof(ns))/(dPN⋅n))*area # a little faster but a few more iter

    # Use form below to ensure correctness, could be simplified for performance
    e = ns*e # original
    Ef = ((Sf⋅Sf)/(Sf⋅e))*e # original
    Ef_mag = norm(Ef)
    ap = term.sign*(term.flux[fID]*Ef_mag)/delta


    # ap = term.sign*(term.flux[fID]*area)/delta # Initial form used

    # ap = term.sign*(term.flux[fID]*Af)/Δ # minimum correction formulation

    # Test formulation using vector d instead of e to explore any stability benefits
    # Ef = ((Sf⋅Sf)/(Sf⋅d))*d
    # Ef_mag = norm(Ef)
    # ap = term.sign*(term.flux[fID]*Ef_mag)/Δ

    # Increment sparse array
    ac = -ap
    an = ap
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Laplacian{Linear}}, cell, cID, cIndex, prev, runtime)  where {F,P,I} = begin
    0.0, 0.0
end

# DIVERGENCE

# Linear
@inline function scheme!(
    term::Operator{F,P,I,Divergence{Linear}},
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}

    w = face.weight
    # signbit(ns) ? w = one(w) - w : w
    w = 0.5 + ns*(w - 0.5)

    # Calculate link coefficients
    ap = term.sign*(term.flux[fID]*ns)
    ac = ap*w
    an = ap*(one(w) - w)
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{Linear}}, cell, cID, cIndex, prev, runtime) where {F,P,I} = begin
    0.0, 0.0
end

# Upwind
@inline function scheme!(
    term::Operator{F,P,I,Divergence{Upwind}},
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    # Calculate link coefficients
    ap = term.sign*(term.flux[fID]*ns)
    ac = max(ap, 0.0)
    an = -max(-ap, 0.0)
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{Upwind}}, cell, cID, cIndex, prev, runtime) where {F,P,I} = begin
    0.0, 0.0
end

# LUST
@inline function scheme!(
    term::Operator{F,P,I,Divergence{LUST}},
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}

    w = face.weight
    signbit(ns) ? w = one(w) - w : w

    # Calculate link coefficients
    ap = term.sign*(term.flux[fID]*ns)
    acLinear = ap*w
    anLinear = ap*(one(w) - w)
    acUpwind = max(ap, 0.0)
    anUpwind = -max(-ap, 0.0)
    ac = 0.75*acLinear + 0.25*acUpwind
    an = 0.75*anLinear + 0.25*anUpwind
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{LUST}}, cell, cID, cIndex, prev, runtime) where {F,P,I} = begin
    0.0, 0.0
end

# BoundedUpwind
@inline function scheme!(
    term::Operator{F,P,I,Divergence{BoundedUpwind}},
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    # $$\mathcal{D}_{bounded} = \sum_f \phi_f \psi_f - \psi_P \sum_f \phi_f$$
    # phif =  max(phif, 0) - max(-phi_f, 0)$
    # phif psif =  max(phif, 0) psi_P - max(-phi_f, 0)$ psi_N
    ap = term.sign*(term.flux[fID]*ns)
    ac = max(-ap, 0.0)
    an = -max(-ap, 0.0)
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{BoundedUpwind}}, cell, cID, cIndex, prev, runtime) where {F,P,I} = begin
    0.0, 0.0
end



# BIHARMONIC OPERATOR
# NOTE: orthogonal-mesh assumption — uses area/delta² stencil with no non-orthogonal
# correction. Results degrade on skewed or non-orthogonal meshes.
@inline function scheme!(
    term::Operator{F,P,I,Biharmonic{T}},
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I,T}

    (; area, delta) = face
    gamma = term.flux[fID]
    coeff = gamma * area / delta

    # MASTER (Self) contribution to diagonal
    # Note: On a standard 1st-order mesh, we approximate it by Laplacian(Laplacian(phi)).
    ac = coeff / delta

    # NEIGHBOUR contribution
    an = -coeff / delta

    return ac, an
end

@inline scheme_source!(
    term::Operator{F,P,I,Biharmonic{T}}, cell, cID, cIndex, prev, runtime) where {F,P,I,T} = begin
    0.0, 0.0
end

# GRADDIV — two-point elastic coupling operator
# Returns face coefficient α_f * e[J] * (A*n)[I] / delta for the (I,J) block.
# Note: ns² = 1, so the coefficient is independent of which side we assemble from.
@inline function scheme!(
    term::Operator{F,P,I,GradDiv{T,I_ROW,J_COL}},
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
) where {F,P,I,T,I_ROW,J_COL}
    e_J  = face.e[J_COL]
    n_I  = face.normal[I_ROW]
    ap   = term.sign * term.flux[fID] * e_J * face.area * n_I / face.delta
    return -ap, ap
end

@inline scheme_source!(
    term::Operator{F,P,I,GradDiv{T,I_ROW,J_COL}}, cell, cID, cIndex, prev, runtime
) where {F,P,I,T,I_ROW,J_COL} = (0.0, 0.0)

# SCALARGRAD — volume-integrated gradient of a scalar field, component I
#
# Conservative Gauss FVM approximation of ∫ ∂φ/∂x_I dV:
#   Σ_f φ_f S_f[I]
# with linear face interpolation.  This must not use a two-point difference
# divided by delta; that would assemble a Laplacian-like stencil and break the
# pressure/divergence saddle structure on collocated meshes.
#
# Typical uses: pressure-gradient in momentum equations and any off-diagonal
# scalar-gradient coupling.
# In a MonolithicSystem, `term.phi` determines the column block.
@inline function scheme!(
    term::Operator{F,P,I,ScalarGrad{T,I_ROW}},
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
) where {F,P,I,T,I_ROW}
    w = 0.5 + ns * (face.weight - 0.5)
    Sf_I = ns * face.area * face.normal[I_ROW]
    ap = term.sign * term.flux[fID] * Sf_I
    return ap * w, ap * (one(w) - w)
end

@inline scheme_source!(
    term::Operator{F,P,I,ScalarGrad{T,I_ROW}}, cell, cID, cIndex, prev, runtime
) where {F,P,I,T,I_ROW} = (0.0, 0.0)

# VECTORDIV — volume-integrated J-th component of divergence
#
# Conservative Gauss FVM approximation of ∫ ∂u_J/∂x_J dV:
#   Σ_f u_{J,f} S_f[J]
# with linear face interpolation.
#
# Typical uses: ∇·u in continuity/pressure equations and volumetric strain
# coupling.
# Sum VectorDiv{T,J} over J = 1…d to form ∇·u in a scalar equation.
# In a MonolithicSystem, `term.phi` (= u_J) determines the column block.
@inline function scheme!(
    term::Operator{F,P,I,VectorDiv{T,J_COL}},
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
) where {F,P,I,T,J_COL}
    w = 0.5 + ns * (face.weight - 0.5)
    Sf_J = ns * face.area * face.normal[J_COL]
    ap = term.sign * term.flux[fID] * Sf_J
    return ap * w, ap * (one(w) - w)
end

@inline scheme_source!(
    term::Operator{F,P,I,VectorDiv{T,J_COL}}, cell, cID, cIndex, prev, runtime
) where {F,P,I,T,J_COL} = (0.0, 0.0)

# CoupledSi
@inline scheme!(
    term::Operator{F,P,I,CoupledSi}, nzval, cell, face,  cellN, ns, cID, nID, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I} = begin
    0.0, 0.0, 0.0
end

@inline function scheme_source!(
    term::Operator{F,P,I,CoupledSi}, cell, cID, cIndex, prev, runtime
) where {F,P,I}
    return term.sign * term.flux[cID] * cell.volume, 0.0
end

# IMPLICIT SOURCE
@inline function scheme!(
    term::Operator{F,P,I,Si},
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    0.0, 0.0
end
@inline scheme_source!(
    term::Operator{F,P,I,Si}, cell, cID, cIndex, prev, runtime)  where {F,P,I} = begin

    # Retrieve and calculate flux for cell
    flux = term.sign*term.flux[cID]*cell.volume # indexed with cID
    ac = flux # indexed with cIndex
    ac, 0.0
end

# NONLINEAR IMPLICIT SOURCE
#
# NonLinearSi stores an arbitrary function f(φ) as a type tag.
# In the normal newton_solve! path, linearize_physics converts NonLinearSi → Si
# BEFORE discretise! is called, so this scheme is never reached there.
#
# This scheme enables two additional use cases:
#   1. explicit_residual!(r, eqn, phi, config) — matrix-free evaluation of the
#      true nonlinear residual F(phi) = A_lin·phi + f(phi)·vol − b,
#      used directly in JFNK without any workaround.
#   2. Picard iteration — solve_equation! on a NonLinearSi equation treats
#      f(φ^k) as an explicit source and iterates (less robust than Newton).
#
# Contribution to r in explicit_residual!:
#   r[i] += 0·φᵢ − b = −(−sign·f(φᵢ)·vol) = sign·f(φᵢ)·vol   ✓
@inline function scheme!(
    term::Operator{F,P,I,NonLinearSi{Fun}},
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
) where {F,P,I,Fun}
    0.0, 0.0   # source only — no face flux contribution
end

@inline function scheme_source!(
    term::Operator{F,P,I,NonLinearSi{Fun}}, cell, cID, cIndex, prev, runtime
) where {F,P,I,Fun}
    val = prev[cID]
    b   = -term.sign * term.type.func(val) * cell.volume
    zero(val), b   # ac=0 (nothing on diagonal), b carries explicit nonlinear value
end
