from ratchet.types import Guard, Harness


def _seed() -> Harness:
    return Harness.build(
        parent=None, system_prompt="p", procedures=[], guards=[], tool_allowlist=["b", "a"], created_by="seed"
    )


def test_build_is_content_addressed() -> None:
    a, b = _seed(), _seed()
    assert a.digest == b.digest and a.version == a.digest[:12]
    assert a.tool_allowlist == ["a", "b"]
    assert a.verify_digest()


def test_child_links_parent_and_changes_digest() -> None:
    seed = _seed()
    g = Guard(id="g1", description="d", source="def check(call, view, history):\n    return None\n")
    kid = seed.child(guards=[g], created_by="improver")
    assert kid.parent == seed.version and kid.digest != seed.digest and kid.verify_digest()


def test_tampered_harness_fails_verification() -> None:
    from dataclasses import replace

    seed = _seed()
    assert not replace(seed, system_prompt="other").verify_digest()
