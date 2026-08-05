# W7-X stella same-state RHS replay

This compact fixture records solver RHS operators applied directly to traced
stella distribution and potential arrays. The external raw trace is not stored
in the repository. Both periodic spectral and open GKW-upwind parallel models
are reported on a 16×4 velocity grid contained inside the stella domain.

Status: `same_state_rhs_parity_failed`. Maximum RHS relative L2 error:
`2.056022` (tolerance `0.1`).
