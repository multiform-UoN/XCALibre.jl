export volume_integral, weighted_volume_integral, volume_average, total_volume

"""
    volume_integral(phi::ScalarField) → Tf

Volume-weighted integral of a scalar field over the domain:

    ∫ phi dV

Returns the total integral (NOT the volume average).
Divide by `total_volume(mesh)` for the volume average.

# Example
    I = volume_integral(C)
    C_avg = I / total_volume(mesh)
"""
function volume_integral(phi::ScalarField)
    mesh = phi.mesh
    cells = mesh.cells
    s = zero(eltype(phi))
    for cID in eachindex(cells)
        s += phi[cID] * cells[cID].volume
    end
    return s
end

"""
    volume_integral(phi::VectorField) → SVector{3}

Volume-weighted integral of each component of a vector field.
Returns a 3-vector `[∫ux dV, ∫uy dV, ∫uz dV]`.
"""
function volume_integral(phi::VectorField)
    mesh = phi.mesh
    cells = mesh.cells
    Tf = eltype(phi.x)
    sx, sy, sz = zero(Tf), zero(Tf), zero(Tf)
    for cID in eachindex(cells)
        vol = cells[cID].volume
        v = phi[cID]
        sx += v[1] * vol
        sy += v[2] * vol
        sz += v[3] * vol
    end
    return [sx, sy, sz]
end

"""
    total_volume(mesh) → Tf

Sum of all cell volumes in the mesh domain.
"""
function total_volume(mesh)
    cells = mesh.cells
    s = zero(eltype(cells[1].volume))
    for cell in cells
        s += cell.volume
    end
    return s
end

"""
    weighted_volume_integral(phi::ScalarField, weight_func) → Tf

Volume-weighted integral of `phi` against an analytical weight function `w(x,y,z)`:

    ∫ phi(x) * w(x, y, z) dV

where the integral is approximated as a sum over cell centroids:

    Σ_i  phi_i * w(x_i, y_i, z_i) * V_i

# Arguments
- `phi`: scalar field to integrate
- `weight_func`: function `w(x, y, z) → scalar` evaluated at each cell centroid

# Example — effective dispersivity (homogenisation):
    D_eff_xx = (D_mol * vol + weighted_volume_integral(X_corrector, (x,y,z) -> v_pore)) / vol
"""
function weighted_volume_integral(phi::ScalarField, weight_func)
    mesh = phi.mesh
    cells = mesh.cells
    s = zero(eltype(phi))
    for cID in eachindex(cells)
        c = cells[cID].centre
        w = weight_func(c[1], c[2], c[3])
        s += phi[cID] * w * cells[cID].volume
    end
    return s
end

"""
    weighted_volume_integral(phi::VectorField, weight_func) → Vector{3}

Volume-weighted integral of each component of a vector field against `w(x,y,z)`.
Returns `[∫ux*w dV, ∫uy*w dV, ∫uz*w dV]`.
"""
function weighted_volume_integral(phi::VectorField, weight_func)
    mesh = phi.mesh
    cells = mesh.cells
    Tf = eltype(phi.x)
    sx, sy, sz = zero(Tf), zero(Tf), zero(Tf)
    for cID in eachindex(cells)
        c = cells[cID].centre
        w = weight_func(c[1], c[2], c[3])
        vol = cells[cID].volume
        v = phi[cID]
        sx += v[1] * w * vol
        sy += v[2] * w * vol
        sz += v[3] * w * vol
    end
    return [sx, sy, sz]
end

"""
    volume_average(phi::ScalarField) → Tf

Volume-averaged mean of a scalar field: `(∫ phi dV) / (∫ dV)`.
"""
function volume_average(phi::ScalarField)
    return volume_integral(phi) / total_volume(phi.mesh)
end

"""
    volume_average(phi::VectorField) → Vector{3}

Volume-averaged mean of a vector field.
"""
function volume_average(phi::VectorField)
    return volume_integral(phi) ./ total_volume(phi.mesh)
end
