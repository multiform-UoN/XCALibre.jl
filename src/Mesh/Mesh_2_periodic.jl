# =============================================================================
# Topological Periodic Boundary Conditions — XCALibre.jl
# =============================================================================

export construct_periodic_topology

"""
    construct_periodic_topology(mesh, patch1::Symbol, patch2::Symbol, translation::AbstractVector; tol=1e-5)

Returns a new mesh where faces on `patch1` and `patch2` are rewired as internal faces.
The `translation` vector should point from `patch1` to `patch2`.

This implementation treats periodic boundaries as standard internal faces during
assembly, which naturally supports monolithic block-coupled systems.
"""
function construct_periodic_topology(mesh::Mesh2, patch1::Symbol, patch2::Symbol, translation::AbstractVector; tol=1e-5)
    _construct_periodic_topology_generic(mesh, patch1, patch2, translation, tol, Face2D, Mesh2)
end

function construct_periodic_topology(mesh::Mesh3, patch1::Symbol, patch2::Symbol, translation::AbstractVector; tol=1e-5)
    _construct_periodic_topology_generic(mesh, patch1, patch2, translation, tol, Face3D, Mesh3)
end

function _construct_periodic_topology_generic(mesh, patch1, patch2, translation, tol, FaceType, MeshType)
    TI = _get_int(mesh)
    TF = _get_float(mesh)

    # 1. Identify periodic face pairs
    b_idx1 = findfirst(x -> x.name == patch1, mesh.boundaries)
    b_idx2 = findfirst(x -> x.name == patch2, mesh.boundaries)
    @assert b_idx1 !== nothing && b_idx2 !== nothing "Periodic patches $patch1 or $patch2 not found."

    range1 = mesh.boundaries[b_idx1].IDs_range
    range2 = mesh.boundaries[b_idx2].IDs_range

    # Map from patch1 face ID to patch2 face ID
    # Note: translation is patch1 -> patch2
    periodic_map = Dict{Int, Int}()
    for f1 in range1
        c1 = mesh.faces[f1].centre
        expected_c2 = c1 + translation
        found = false
        for f2 in range2
            c2 = mesh.faces[f2].centre
            if norm(expected_c2 - c2) < tol
                periodic_map[f1] = f2
                found = true
                break
            end
        end
        @assert found "Could not find periodic match for face $f1 on patch $patch1"
    end

    # 2. Categorize all faces into: Kept Boundary, Original Internal, and New Periodic
    n_boundary_total = total_boundary_faces(mesh)
    boundary_mask = trues(n_boundary_total)
    for f1 in range1; boundary_mask[f1] = false; end
    for f2 in range2; boundary_mask[f2] = false; end

    kept_boundary_indices = findall(boundary_mask)
    original_internal_indices = (n_boundary_total + 1):length(mesh.faces)

    # 3. Build new faces array and ID mapping
    # New order: [kept_boundary_faces..., original_internal_faces..., new_periodic_faces...]
    new_faces = FaceType[]
    old_to_new_face = Dict{Int, Int}()

    # Add kept boundary faces
    for (i, old_f) in enumerate(kept_boundary_indices)
        push!(new_faces, mesh.faces[old_f])
        old_to_new_face[old_f] = i
    end

    # Add original internal faces
    offset = length(new_faces)
    for (i, old_f) in enumerate(original_internal_indices)
        push!(new_faces, mesh.faces[old_f])
        old_to_new_face[old_f] = offset + i
    end

    # 4. Map cells to their new periodic faces
    cell_to_periodic_faces = [Tuple{Int, Int, Int}[] for _ in 1:length(mesh.cells)]
    periodic_f1_list = sort(collect(keys(periodic_map)))
    for (i, f1) in enumerate(periodic_f1_list)
        f2 = periodic_map[f1]
        face1 = mesh.faces[f1]
        face2 = mesh.faces[f2]

        owner1 = face1.ownerCells[1]
        owner2 = face2.ownerCells[1]

        # Calculate periodic geometry using translated neighbor
        C1 = mesh.cells[owner1].centre
        C2_eff = mesh.cells[owner2].centre - translation

        C1C2 = C2_eff - C1
        weight, delta, e = if norm(C1C2) < tol
            # Degenerate case: cell centres coincide after translation shift
            # (happens when inlet/outlet have the same normal direction and uniform grid).
            # Fall back to face-normal stencil: place virtual C2 as mirror of C1 across face.
            d1 = abs(dot(face1.centre - C1, face1.normal))
            d2 = abs(dot(face2.centre - mesh.cells[owner2].centre, face1.normal))
            delta_fb = d1 + d2
            e_fb = normalize(face1.centre - C1)  # direction from C1 toward face
            TF(0.5), TF(delta_fb), e_fb
        else
            weight_delta_e(face1.centre - C1, face1.centre - C2_eff, C1C2, face1.normal)
        end

        new_periodic_face = FaceType(
            face1.nodes_range,
            SVector{2, TI}(owner1, owner2),
            face1.centre,
            face1.normal,
            e,
            face1.area,
            delta,
            weight
        )
        push!(new_faces, new_periodic_face)

        # The periodic face is appended after all copied boundary/internal faces.
        # Using `offset + i` here points into the copied internal-face block and
        # corrupts cell connectivity whenever original internal faces exist.
        # The ID must be the actual index of the newly appended periodic face.
        new_fID = length(new_faces)
        push!(cell_to_periodic_faces[owner1], (new_fID, owner2, 1))
        push!(cell_to_periodic_faces[owner2], (new_fID, owner1, -1))
    end

    # 5. Rebuild connectivity structures for all cells
    new_cell_faces = TI[]
    new_cell_neighbours = TI[]
    new_cell_nsign = TI[]
    new_cells = Cell{TF, SVector{3,TF}, UnitRange{TI}}[]

    for cID in 1:length(mesh.cells)
        cell = mesh.cells[cID]
        start_idx = length(new_cell_faces) + 1

        # Add original internal faces (mapped)
        for fi in cell.faces_range
            old_fID = mesh.cell_faces[fi]
            new_fID = old_to_new_face[old_fID]
            push!(new_cell_faces, new_fID)
            push!(new_cell_neighbours, mesh.cell_neighbours[fi])
            push!(new_cell_nsign, mesh.cell_nsign[fi])
        end

        # Add new periodic faces
        for (fID, nID, nsign) in cell_to_periodic_faces[cID]
            push!(new_cell_faces, fID)
            push!(new_cell_neighbours, nID)
            push!(new_cell_nsign, nsign)
        end

        stop_idx = length(new_cell_faces)
        push!(new_cells, Cell(cell.centre, cell.volume, cell.nodes_range, UnitRange{TI}(start_idx, stop_idx)))
    end

    # 6. Rebuild Boundary metadata
    new_boundaries = Boundary{Symbol, UnitRange{TI}}[]
    curr_b_idx = 1
    for b in mesh.boundaries
        if b.name == patch1 || b.name == patch2
            push!(new_boundaries, Boundary(b.name, UnitRange{TI}(0, -1))) # Empty range
        else
            old_range = b.IDs_range
            new_range = UnitRange{TI}(curr_b_idx, curr_b_idx + length(old_range) - 1)
            push!(new_boundaries, Boundary(b.name, new_range))
            curr_b_idx += length(old_range)
        end
    end

    # 7. Rebuild boundary_cellsID
    new_boundary_cellsID = TI[]
    for old_f in kept_boundary_indices
        push!(new_boundary_cellsID, mesh.boundary_cellsID[old_f])
    end

    # 8. Final Mesh object
    return MeshType(
        new_cells,
        mesh.cell_nodes,
        new_cell_faces,
        new_cell_neighbours,
        new_cell_nsign,
        new_faces,
        mesh.face_nodes,
        new_boundaries,
        mesh.nodes,
        mesh.node_cells,
        mesh.get_float,
        mesh.get_int,
        new_boundary_cellsID
    )
end
