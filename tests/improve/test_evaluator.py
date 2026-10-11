from __future__ import annotations

import pytest

from ratchet.improve.evaluator import EvaluatorChangedError, evaluator_identity, require_same

from .fakes import FakeCase, fake_cases


def test_identity_is_stable_and_names_every_component() -> None:
    a = evaluator_identity(fake_cases(), {"model": "m", "temperature": 0}, "suite")
    b = evaluator_identity(list(reversed(fake_cases())), {"model": "m", "temperature": 0}, "suite")
    assert a == b and set(a["components"]) == {"cases", "graders", "faults", "model_config", "suite_file"}
    require_same(a, b)


def test_any_change_is_a_new_evaluator_version() -> None:
    a = evaluator_identity(fake_cases(), {"model": "m", "temperature": 0})
    hotter = evaluator_identity(fake_cases(), {"model": "m", "temperature": 1})
    relabelled = evaluator_identity(
        [*fake_cases()[:-1], FakeCase("plain-6", "train")], {"model": "m", "temperature": 0}
    )
    for other, part in ((hotter, "model_config"), (relabelled, "cases")):
        assert other["version"] != a["version"]
        with pytest.raises(EvaluatorChangedError, match=part):
            require_same(a, other)
