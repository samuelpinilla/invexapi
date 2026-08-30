import json

from invexapi import metadata
from invexapi.penalties import Reference

_EXPECTED_NAMES = {
    "Loss",
    "Sum",
    "QuasinormInvexPenalty",
    "_QuasinormProx",
    "LogInvexPenalty",
    "TikhonovPenalty",
    "Solver",
    "FISTA",
    "NonlinearCG",
    "warn_if_unproven",
}


def test_dump_all_covers_every_documented_spot():
    names = {entry["name"] for entry in metadata.dump_all()}
    assert _EXPECTED_NAMES.issubset(names)


def test_dump_all_json_round_trips():
    parsed = json.loads(metadata.dump_all_json())
    assert isinstance(parsed, list)
    assert len(parsed) == len(metadata.dump_all())


def test_quasinorm_prox_has_provenance():
    entry = next(e for e in metadata.dump_all() if e["name"] == "_QuasinormProx")

    assert len(entry["provenance"]) >= 1
    assert entry["provenance"][0]["files"]


def test_tikhonov_certificates_appear_via_example():
    entry = next(e for e in metadata.dump_all() if e["name"] == "TikhonovPenalty")

    assert set(entry["certificates"]) == {"convex", "invex", "quasi_convex", "quasi_invex"}
    assert entry["certificates"]["convex"]["status"] == "assumed"


def test_reference_still_importable_from_penalties():
    # Reference moved to invexapi.metadata but must stay available from its
    # original location for backward compatibility.
    assert Reference is metadata.Reference
