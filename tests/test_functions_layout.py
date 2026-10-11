from invexapi import functions
from invexapi.functions import catalog


def test_catalog_filters_by_certificate():
    invex = set(catalog(invex=True))
    convex = set(catalog(convex=True))

    assert {functions.LogPenalty, functions.LogMinusRationalPenalty, functions.QuasinormInvexPenalty} <= invex
    assert convex == {functions.L1Penalty, functions.TikhonovPenalty, functions.PSDTracePenalty}
    assert convex <= invex  # every certified-convex function here is also certified invex
    assert set(catalog(invex=True, convex=False)) == invex - convex


def test_catalog_lists_only_functions():
    assert all(cls.__module__.startswith("invexapi.functions") for cls in catalog())
