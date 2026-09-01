"""Selection substages.

===================  =====================================================
``select.route``     manual vs AI; resolves the pool, lookup table and the
                     request-level inputs every category is judged against
``select.budget``    per-category photo and spread allowance
``select.pick``      the category loop; delegates to a category strategy
``select.publish``   narrow the photo table and finalise the outcome
===================  =====================================================

``select.pick`` is decomposed a second time internally: the shared scaffolding
(scoring, gating, temporal narrowing, colour top-up) lives in the driver, while
each content category's rule is a
:class:`~src.pipeline.select.strategies.base.CategoryStrategy`.

Importing this package registers every substage it defines.
"""

from src.pipeline.select import (  # noqa: F401  (imported for registration)
    budget,
    pick,
    publish,
    route,
)

__all__ = ["budget", "pick", "publish", "route"]
