# Shared numerical algebra

Canonical coordinates are `(scalar, i, j, k, ell)`. The fifth coordinate is the
**actual coefficient of ell**, not a reference coordinate to be multiplied by
gamma again. At gamma=0 the ambient algebra remains five-dimensional; the
quaternion subalgebra is the first four coordinates.

- `coordinate_product.product(p, q, gamma)` is the component-wise implementation
  used by the Riccati model.
- `algebra_core.product(gamma, p, q)` is the independent table-based implementation.
  Note its different argument order. Unit tests
  compare every basis product and seeded random examples between the two APIs.
- `algebra_core` also supplies conjugation, associators and moving frames.
- `reference_data` reads a CSV or its single-member ZIP without extraction or
  modification. Simulation drivers write their own output trees.

Independent symbolic programs retain their own exact multiplication tables so
that their checks do not merely repeat a call to the numerical implementation.
The two numerical implementations are small, explicitly cross-tested reference
forms, not separate copies of the full research code.
