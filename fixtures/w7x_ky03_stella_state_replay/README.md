# W7-X stella same-state RHS replay

This compact fixture records solver RHS operators applied directly to traced
stella distribution and potential arrays. The external raw trace is not stored
in the repository. Both periodic spectral and open GKW-upwind parallel models
are reported on a 16×4 velocity grid contained inside the stella domain. Two
additional discriminators apply source-derived stella mirror, drift, and
drive coefficients at 16×4 and an exact-node 32×4 velocity resolution without
changing the remaining production geometry conventions.

The v4 trace also verifies the mirror operator directly on stella's native
256×32×8 grid before interpolation. Its maximum reconstruction error is
`0.0037034624`.

Status: `same_state_rhs_parity_failed`. Acceptance-case maximum RHS relative L2 error:
`0.24326839` (tolerance `0.1`).
