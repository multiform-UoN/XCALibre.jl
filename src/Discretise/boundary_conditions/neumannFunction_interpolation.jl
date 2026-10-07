@inline function boundary_interpolation!(
    BC::NeumannFunction{T,Test,R}, phif::FaceScalarField, phi, boundary_cellsID, time, fID) where {T,Test<:Function,R}
    (; faces) = phi.mesh
    @inbounds begin
        i = fID - BC.IDs_range.start + 1
        cID = boundary_cellsID[fID]
        phif[fID] = phi[cID] + faces.delta[fID]*BC.value(faces.centre[fID], time, i)
    end
    nothing
end

@inline function boundary_interpolation!(
    BC::NeumannFunction{T,Test,R}, psif::FaceVectorField, psi, boundary_cellsID, time, fID) where {T,Test<:Function,R}
    (; faces) = psi.mesh
    @inbounds begin
        # face = faces[fID]
        # i = fID - BC.IDs_range.start + 1
        # psif[fID] = BC.value(face.centre, time, i)
        cID = boundary_cellsID[fID]
        psif[fID] = psi[cID]
    end
    nothing
end

@inline function boundary_interpolation!(
    BC::NeumannFunction{T,Test,R}, phif::FaceScalarField, phi, boundary_cellsID, time, fID) where {T,Test<:XCALibreUserFunctor,R}
    (; faces) = phi.mesh
    @inbounds begin
        # face = faces[fID]
        # i = fID - BC.IDs_range.start + 1
        # phif[fID] = BC.value(face.centre, time, i)
        cID = boundary_cellsID[fID]
        phif[fID] = phi[cID]
    end
    nothing
end

@inline function boundary_interpolation!(
    BC::NeumannFunction{T,Test,R}, psif::FaceVectorField, psi, boundary_cellsID, time, fID) where {T,Test<:XCALibreUserFunctor,R}
    (; faces) = psi.mesh
    @inbounds begin
        # face = faces[fID]
        # i = fID - BC.IDs_range.start + 1
        # psif[fID] = BC.value(face.centre, time, i)
        cID = boundary_cellsID[fID]
        psif[fID] = psi[cID]
    end
    nothing
end
