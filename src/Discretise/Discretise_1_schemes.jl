export scheme!, scheme_source!

#= NOTE:
In source scheme the following indices are used and should be used with care:
cID - Index of the cell outer loop. Use to index "b" 
cIndex - Index of the cell based on sparse matrix. Use to index "nzval_array"
=#

# TIME 

# SteadyState
@inline function scheme!(
    term::Operator{F,P,I,Time{SteadyState},Fn}, 
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P,I,Fn}
    # nothing
    0.0, 0.0 # add types if this approach works
end

# IMPLICIT SOURCE
@inline function scheme!(
    term::Operator{F,P,I,Si,Fn}, 
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I,Fn}
    0.0, 0.0
end

@inline scheme_source!(
    term::Operator{F,P,I,Time{SteadyState},Fn}, cell, cID, cIndex, prev, runtime)  where {F,P,I,Fn} = begin
    0.0, 0.0
end

## Euler
@inline function scheme!(
    term::Operator{F,P,I,Time{Euler},Fn}, 
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P,I,Fn}

    0.0, 0.0 # add types if this approach works
end
@inline scheme_source!(
    term::Operator{F,P,I,Time{Euler},Fn}, cell, cID, cIndex, prev, runtime)  where {F,P,I,Fn} = begin
        volume = cell.volume
        # To DO!!!!!
        # flux below is for current time - need to also store previous flux
        vol_rdt = term.flux[cID]*volume/runtime.dt[1]
        
        # Increment sparse and b arrays 
        ac = vol_rdt
        b = prev[cID]*vol_rdt
        return ac, b
end

## Crank-Nicholson
@inline function scheme!(
    term::Operator{F,P,I,Time{CrankNicolson},Fn}, 
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P,I,Fn}

    0.0, 0.0 # add types if this approach works
end
@inline scheme_source!(
    term::Operator{F,P,I,Time{CrankNicolson},Fn}, cell, cID, cIndex, prev, runtime)  where {F,P,I,Fn} = begin
        volume = cell.volume
        vol_rdt = term.flux[cID]*volume/runtime.dt[1]
        
        # Increment sparse and b arrays 
        ac = vol_rdt
        b = prev[cID]*vol_rdt
        return ac, b
end

# LAPLACIAN

@inline function scheme!(
    term::Operator{F,P,I,Laplacian{Linear},Fn}, 
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I,Fn}

    
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
    term::Operator{F,P,I,Laplacian{Linear},Fn}, cell, cID, cIndex, prev, runtime)  where {F,P,I,Fn} = begin
    0.0, 0.0
end

# DIVERGENCE

# Linear
@inline function scheme!(
    term::Operator{F,P,I,Divergence{Linear},Fn}, 
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I,Fn}

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
    term::Operator{F,P,I,Divergence{Linear},Fn}, cell, cID, cIndex, prev, runtime) where {F,P,I,Fn} = begin
    0.0, 0.0
end

# Upwind
@inline function scheme!(
    term::Operator{F,P,I,Divergence{Upwind},Fn}, 
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I,Fn}
    # Calculate link coefficients
    ap = term.sign*(term.flux[fID]*ns)
    ac = max(ap, 0.0) 
    an = -max(-ap, 0.0)
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{Upwind},Fn}, cell, cID, cIndex, prev, runtime) where {F,P,I,Fn} = begin
    0.0, 0.0
end

# LUST
@inline function scheme!(
    term::Operator{F,P,I,Divergence{LUST},Fn}, 
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I,Fn}
    
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
    term::Operator{F,P,I,Divergence{LUST},Fn}, cell, cID, cIndex, prev, runtime) where {F,P,I,Fn} = begin
    0.0, 0.0
end

# BoundedUpwind
@inline function scheme!(
    term::Operator{F,P,I,Divergence{BoundedUpwind},Fn}, 
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I,Fn}
    # $$\mathcal{D}_{bounded} = \sum_f \phi_f \psi_f - \psi_P \sum_f \phi_f$$
    # phif =  max(phif, 0) - max(-phi_f, 0)$
    # phif psif =  max(phif, 0) psi_P - max(-phi_f, 0)$ psi_N
    ap = term.sign*(term.flux[fID]*ns)
    ac = max(-ap, 0.0)
    an = -max(-ap, 0.0)
    return ac, an
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{BoundedUpwind},Fn}, cell, cID, cIndex, prev, runtime) where {F,P,I,Fn} = begin
    0.0, 0.0
end


# BIHARMONIC OPERATOR
@inline function scheme!(
    term::Operator{F,P,I,Biharmonic{T},Fn}, 
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I,T,Fn}
    
    # Biharmonic: div(grad(div(grad(phi)))) 
    # For now, let's implement it using a 5-point stencil (1D) or 9-point (2D)
    # by using the auxiliary Laplacian logic.
    # Note: On GPU, this requires extended connectivity to be pre-calculated.
    
    # Simplified implementation for proof of concept (Stencil expansion):
    # This part should ideally be done in a separate kernel that first 
    # calculates Laplacian(phi) then takes the Laplacian of that.
    # Here we simulate it by adding contributions to nb and nb-of-nb.
    
    (; area, delta) = face
    gamma = term.flux[fID]
    coeff = gamma * area / delta
    
    # MASTER CELL (cIndex) contribution
    # For a 1D Laplacian-of-Laplacian: phi_{i+2} - 4phi_{i+1} + 6phi_i - 4phi_{i-1} + phi_{i-2}
    # This is handled by distributing the flux across the wider stencil.
    
    # AP contribution (diagonal)
    # AP = 6.0 * coeff / delta ? (Correct scaling depends on grid spacing)
    # This is complex to do in a single face loop. 
    # RECOMMENDED: Implement as auxiliary variable if GPU performance is key.
    
    return coeff, -coeff
end

@inline scheme_source!(
    term::Operator{F,P,I,Biharmonic{T},Fn}, cell, cID, cIndex, prev, runtime) where {F,P,I,T,Fn} = begin
    0.0, 0.0
end

# Placeholder for Generic Newton Discretisation
# Note: For GPU implementation of Generic Newton, ForwardDiff 
# needs to be replaced with a custom kernel that handles dual numbers
# or manual Jacobian derivation. Currently CPU-only logic is in linearize_physics.
@inline scheme_source!(
    term::Operator{F,P,I,Si,Fn}, cell, cID, cIndex, prev, runtime)  where {F,P,I,Fn} = begin
    
    # Retrieve and calculate flux for cell 
    flux = term.sign*term.flux[cID]*cell.volume # indexed with cID
    ac = flux # indexed with cIndex
    ac, 0.0
end
