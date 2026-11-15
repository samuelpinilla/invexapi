import json

from invexapi import metadata
from invexapi import Reference

_EXPECTED_NAMES = {
    "Loss",
    "Sum",
    "QuasinormInvexPenalty",
    "_QuasinormProx",
    "LogPenalty",
    "LogMinusRationalPenalty",
    "TikhonovPenalty",
    "Solver",
    "FISTA",
    "NonlinearCG",
    "warn_if_unproven",
    "FASTA",
    "frft",
    "FractionalFourier",
    "LiftedFrFTAmbiguity",
    "PSDTracePenalty",
    "SymmetricNuclearPenalty",
    "LiftedFROG",
    "LinearTilt",
    "SlackAmplitudeLeastSquares",
    "AmplitudeHingeSquared",
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
    assert entry["provenance"][0]["reference"]["title"]


def test_tikhonov_certificates_appear_via_example():
    entry = next(e for e in metadata.dump_all() if e["name"] == "TikhonovPenalty")

    assert set(entry["certificates"]) == {"convex", "invex", "quasi_convex", "quasi_invex"}
    assert entry["certificates"]["convex"]["status"] == "assumed"


def test_every_provenance_cites_a_published_reference():
    # provenance must be verifiable by a reader: a citation, never a file path
    for entry in metadata.dump_all():
        for prov in entry["provenance"]:
            assert "files" not in prov
            assert prov["reference"] is not None, entry["name"]
            assert prov["reference"]["title"] and prov["reference"]["year"]
