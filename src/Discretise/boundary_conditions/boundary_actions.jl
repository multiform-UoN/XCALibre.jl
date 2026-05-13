# =============================================================================
# DEPRECATED — superseded by Discretise_7_boundary_actions.jl
# =============================================================================
# This file defined a minimal "Clerk" action vocabulary (BoundaryAction, SetRow,
# AddDiagonal, AddSource, inject!, ResidualBC) that partially overlapped with the
# richer AbstractBoundaryAction hierarchy in Discretise_7_boundary_actions.jl.
#
# The file is retained for git history but is NO LONGER included in Discretise.jl.
# Use the Discretise_7 API instead:
#   SetEquationRow, SetNewtonRow, AddDiagonalEntry, AddRHSEntry, AddJacobianEntry
#   LocalScalarResidualBC, apply_boundary_action!, apply_boundary_actions!
# =============================================================================
