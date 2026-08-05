# W7-X stella same-state RHS replay

This compact fixture records solver RHS operators applied directly to traced
stella distribution and potential arrays. The external raw trace is not stored
in the repository. Both periodic spectral and open GKW-upwind parallel models
are reported on a 16×4 velocity grid contained inside the stella domain. A
two additional discriminators apply source-derived stella mirror, drift, and
drive coefficients at 16×4 and an exact-node 32×4 velocity resolution without
changing the remaining production geometry conventions.

Status: `same_state_rhs_parity_failed`. Acceptance-case maximum RHS relative L2 error:
`0.39558905` (tolerance `0.1`).
