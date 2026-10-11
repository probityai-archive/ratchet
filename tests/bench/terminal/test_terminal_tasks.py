from __future__ import annotations

from pathlib import Path

import pytest
from terminal_testkit import make_task

from ratchet.bench.terminal import tasks as T


def suite(root: Path) -> dict[str, T.TaskSpec]:
    for name in T.PHASE_1:
        make_task(root, name, timeout=900.0)
    for name in T.PHASE_2:
        make_task(root, name, timeout=1800.0)
    make_task(root, "long-hard", timeout=3600.0)
    make_task(root, "medium-one", difficulty="medium", timeout=900.0)
    return T.load_suite(root)


def test_load_task_reads_metadata_and_instruction(tmp_path: Path) -> None:
    task = T.load_task(make_task(tmp_path))
    assert task.name == "tiny-task"
    assert task.image == "local/tiny-task:1"
    assert (task.difficulty, task.category) == ("hard", "software-engineering")
    assert (task.agent_timeout_s, task.verifier_timeout_s) == (900.0, 60.0)
    assert (task.cpus, task.memory_mb) == (1.0, 1024)
    assert task.verifier_env == {"MODE": "strict"}
    assert task.instruction.startswith("Write hello")
    assert task.tests_dir == tmp_path / "tiny-task" / "tests"
    assert task.tests_digest == T.tree_digest(task.tests_dir) and len(task.tests_digest) == 64
    (task.tests_dir / "test.sh").write_text("changed", encoding="utf-8")
    assert T.tree_digest(task.tests_dir) != task.tests_digest


def test_load_task_refuses_a_task_without_an_image_or_tests(tmp_path: Path) -> None:
    with pytest.raises(T.SuiteError, match="docker_image"):
        T.load_task(make_task(tmp_path, "no-image", image=False))
    bare = make_task(tmp_path, "no-tests")
    (bare / "tests" / "test.sh").unlink()
    with pytest.raises(T.SuiteError, match=r"test\.sh"):
        T.load_task(bare)


def test_load_suite_refuses_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(T.SuiteError, match=r"no task\.toml"):
        T.load_suite(tmp_path)


def test_phases_are_derived_from_metadata_and_match_the_frozen_lists(tmp_path: Path) -> None:
    found = T.phases(suite(tmp_path))
    assert tuple(found[1]) == T.PHASE_1
    assert tuple(found[2]) == T.PHASE_2


def test_phases_refuse_a_suite_that_changed(tmp_path: Path) -> None:
    tasks = suite(tmp_path)
    make_task(tmp_path, "new-hard-task", timeout=900.0)
    with pytest.raises(T.SuiteError, match="phase 1 derived"):
        T.phases(T.load_suite(tmp_path))
    del tasks["dna-assembly"]
    with pytest.raises(T.SuiteError, match="phase 2 derived"):
        T.phases(tasks)


def test_the_committed_split_is_the_seeded_split() -> None:
    record = T.committed_record()
    assert record["digest"] == T.split_record(record["seed"])["digest"]
    assert record["phases"] == T.assignment(T.DEFAULT_SEED)
    for p, names in (("1", T.PHASE_1), ("2", T.PHASE_2)):
        sides = record["phases"][p]
        assert sorted(sides["train"] + sides["heldout"]) == sorted(names)
        assert len(sides["heldout"]) == T.HELDOUT_PER_PHASE[int(p)]


def test_the_split_changes_with_the_seed() -> None:
    assert T.split_record(1)["digest"] != T.split_record(2)["digest"]


def test_phase_one_cases(tmp_path: Path) -> None:
    cases = T.phase_cases(suite(tmp_path), 1)
    sides = T.assignment()["1"]
    assert [c.task.name for c in cases if c.split == "train"] == sides["train"]
    assert [c.task.name for c in cases if c.split == "heldout"] == sides["heldout"]
    assert {c.phase for c in cases} == {1}
    assert all(c.must_succeed and c.kind == "must_succeed" and c.family == "terminal" for c in cases)
    assert cases[0].id == T.case_id(cases[0].task.name)


def test_phase_two_keeps_phase_one_train_and_audits_phase_one_heldout(tmp_path: Path) -> None:
    cases = T.phase_cases(suite(tmp_path), 2)
    a = T.assignment()
    by_split = {s: [c.task.name for c in cases if c.split == s] for s in ("train", "heldout", "audit")}
    assert by_split["train"] == a["1"]["train"] + a["2"]["train"]
    assert by_split["heldout"] == a["2"]["heldout"]
    assert by_split["audit"] == a["1"]["heldout"]
    assert {c.phase for c in cases if c.split == "audit"} == {1}


def test_phase_cases_refuse_an_unknown_phase(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="phase must be 1 or 2"):
        T.phase_cases(suite(tmp_path), 3)


def test_task_cases(tmp_path: Path) -> None:
    tasks = suite(tmp_path)
    cases = T.task_cases(tasks, ["medium-one", "cancel-async-tasks"], "train")
    assert [(c.id, c.split, c.phase) for c in cases] == [
        ("term-medium-one", "train", 0),
        ("term-cancel-async-tasks", "train", 1),
    ]
    with pytest.raises(T.SuiteError, match="no task named"):
        T.task_cases(tasks, ["missing"])


def test_heldout_of_checks_its_size() -> None:
    assert T.heldout_of(["a", "b", "c"], 0, 1) == []
    with pytest.raises(ValueError, match="between 0 and 3"):
        T.heldout_of(["a", "b", "c"], 4, 1)


def test_phase_of(tmp_path: Path) -> None:
    tasks = suite(tmp_path)
    assert T.phase_of(tasks["long-hard"]) is None
    assert T.phase_of(tasks["medium-one"]) is None
    assert T.phase_of(tasks["dna-assembly"]) == 2


def test_the_wide_split_is_committed_and_keeps_the_first_one_unchanged(tmp_path: Path) -> None:
    record = T.committed_record(T.WIDE_SPLIT_FILE)
    assert record == T.split_record(T.DEFAULT_SEED, T.WIDE_HELDOUT)
    assert record["heldout_per_phase"] == {"1": 6, "2": 5}
    assert "heldout_per_phase" not in T.split_record()
    assert T.committed_record()["digest"] != record["digest"]
    cases = T.phase_cases(suite(tmp_path), 2, heldout=T.WIDE_HELDOUT)
    counts = {s: sum(1 for c in cases if c.split == s) for s in ("train", "heldout", "audit")}
    assert counts == {"train": 11, "heldout": 5, "audit": 6}
