# Core algebra and Riccati model

`algebra_core.py` implements multiplication, conjugation, associators and the
moving frame. `switching_algebra_model.py` supplies another coordinate product,
the Riccati coefficient selection, vector field, tangent action and two
periodic protocols. Numerical multiplication is restricted to `0 <= gamma <= 1`.

Coordinate order is `(1, i, j, k, ell)`. The two product functions have different
argument order: `algebra_core.product(gamma, p, q)` and
`switching_algebra_model.product(p, q, gamma)`.

From `quaternions-to-pentarnions/` run `python run_audit.py` for the sampled algebra checks.
