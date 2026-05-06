export monolithic_discretise!, monolithic_apply_bcs!

# ---------------------------------------------------------------------------
# monolithic_discretise!
#
# Assembles interior-face contributions of a MonolithicSystem into a block-CSR
# matrix.  Boundary conditions are applied separately via monolithic_apply_bcs!.
#
# Design: each Operator carries term.phi (the field it acts on).  The column
# block is determined by field_to_idx[term.phi].  Cross-field terms (e.g.
# Laplacian(a12, C2) inside equation C1) are automatically routed to the
# off-diagonal block (i,j) — no special CoupledSi operator is needed.
#
# Reuses the existing single-equation scheme! / scheme_source! unchanged;
# only the sparse-matrix indices are shifted into the block layout.
# ---------------------------------------------------------------------------
function monolithic_discretise!(
    sys::MonolithicSystem,
    A_mono::SparseMatricesCSR.SparseMatrixCSR,
    b_mono::AbstractVector,
    config
)
    (; equations, phi_list, n_vars, n_cells, field_to_idx) = sys
    (; hardware, runtime) = config
    mesh = phi_list[1].mesh
    (; cells, cell_neighbours, faces, cell_faces, cell_nsign) = mesh

    A_mono.nzval .= 0
    b_mono .= 0

    for (i, model_eqn) in enumerate(equations)
        row_off = (i - 1) * n_cells

        for term in model_eqn.model.terms
            # Route to the correct column block via the operator's own phi.
            # Uses objectid(phi.values) as a stable mutable-array-based key.
            phi_j = term.phi   # for Operator; use linearize_physics first for NonlinearOperator
            j = field_to_idx[objectid(phi_j.values)]
            col_off = (j - 1) * n_cells
            # prev for this term = the field's current values (for time/explicit schemes)
            prev = phi_j.values

            for cID in 1:n_cells
                cell = cells[cID]
                row = row_off + cID
                cIndex_mono = spindex(A_mono.rowptr, A_mono.colval, row, col_off + cID)
                @assert cIndex_mono > 0 "Diagonal entry not found in monolithic matrix at row $row, col $(col_off + cID)"

                ac_sum = 0.0
                for fi in cell.faces_range
                    fID = cell_faces[fi]
                    ns  = cell_nsign[fi]
                    face = faces[fID]
                    nID = cell_neighbours[fi]
                    cellN = cells[nID]
                    col_nb = col_off + nID

                    nIndex_mono = spindex(A_mono.rowptr, A_mono.colval, row, col_nb)
                    @assert nIndex_mono > 0 "Neighbour entry not found in monolithic matrix at row $row, col $col_nb"

                    ac, an = scheme!(term, A_mono.nzval, cell, face, cellN, ns,
                                     cIndex_mono, nIndex_mono, fID, prev, runtime)
                    ac_sum += ac
                    # Use += so that two terms targeting the same block accumulate
                    A_mono.nzval[nIndex_mono] += an
                end

                ac_sc, b_c = scheme_source!(term, cell, cID, cIndex_mono, prev, runtime)
                A_mono.nzval[cIndex_mono] += ac_sum + ac_sc
                b_mono[row] += b_c
            end
        end

        # Explicit source terms (RHS only)
        for src in model_eqn.model.sources
            for cID in 1:n_cells
                row = row_off + cID
                cell = cells[cID]
                b_mono[row] += src.sign * src.field[cID] * cell.volume
            end
        end
    end
end

# ---------------------------------------------------------------------------
# monolithic_apply_bcs!
#
# Applies boundary conditions to the monolithic block-CSR matrix.
#
# For each equation i and each term in that equation (targeting column block j):
# - BCs_j = bcs_list[j] (the BCs of the field phi_j that this term acts on)
# - BC contributions are routed to block (i, j): diagonal and RHS
#
# This is physically correct: each Laplacian(a_ij, phi_j) uses phi_j's own BCs,
# not the equation's self-field BCs.  Cross-field BC contributions go to the
# off-diagonal block (i, j) at the diagonal position within that block.
# ---------------------------------------------------------------------------
function monolithic_apply_bcs!(
    sys::MonolithicSystem,
    A_mono::SparseMatricesCSR.SparseMatrixCSR,
    b_mono::AbstractVector,
    bcs_list,
    config,
    time=0.0
)
    (; equations, phi_list, n_cells, field_to_idx) = sys
    mesh = phi_list[1].mesh
    (; faces, cells, boundary_cellsID) = mesh
    nbfaces = length(boundary_cellsID)

    for (i, model_eqn) in enumerate(equations)
        row_off = (i - 1) * n_cells

        for term in model_eqn.model.terms
            phi_j = term.phi
            j = field_to_idx[objectid(phi_j.values)]
            col_off = (j - 1) * n_cells
            BCs_j = bcs_list[j]

            for fID in 1:nbfaces
                cellID = boundary_cellsID[fID]
                face   = faces[fID]
                cell   = cells[cellID]

                # Monolithic diagonal index in block (i, j)
                row      = row_off + cellID
                zcellID  = spindex(A_mono.rowptr, A_mono.colval, row, col_off + cellID)

                for BC in BCs_j
                    (; start, stop) = BC.IDs_range
                    if start <= fID <= stop
                        k = fID - start + 1
                        # cellID is LOCAL (1..n_cells) — needed by Extrapolated to
                        # access phi.values[cellID] correctly.
                        AP, BP = BC(term, A_mono.colval, A_mono.rowptr, A_mono.nzval,
                                    cellID, zcellID, cell, face, fID, k, nothing, time)
                        A_mono.nzval[zcellID] += AP
                        b_mono[row] += BP
                        break
                    end
                end
            end
        end
    end
end
