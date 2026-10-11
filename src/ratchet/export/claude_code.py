"""Compile an admitted Ratchet guard into a Claude Code PreToolUse hook.

The hook is one self-contained Python file. It carries:
  - a header naming the guard id, the harness version and digest, and the SHA-256 of the signed decision
    receipt that admitted the harness, so provenance survives the file being copied anywhere;
  - `ratchet.export.hookrt`, pasted verbatim: it builds the call, view and history the same way transcript
    replay does, and speaks the Claude Code hook protocol;
  - the guard source and the Ratchet sandbox child (`ratchet/agent/_guard_child.py`) as string constants.

At each tool call the hook starts the sandbox child (`python -I -S`, empty environment, rlimits, network
namespace, seccomp, Landlock), hands it the guard source, and asks one question. It fails closed: a guard
error, a sandbox that cannot apply a required layer, a digest mismatch or unreadable input is a deny.

Only an admitted guard is exported: the receipt must verify under a trusted key, say `admitted`, and name
this exact harness by version and digest, and the harness must contain the guard.

    python -m ratchet.export.claude_code --harness harness.json --guard-id G --receipt receipt.json \
        --trusted-key <ed25519 public key hex> --out .claude/hooks/ratchet_G.py
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ratchet.agent.guards import CHILD_PATH, GuardLimits
from ratchet.export import hookrt
from ratchet.improve.receipt import ReceiptError, key_id, verify
from ratchet.types import Guard, Harness, Procedure, sha256_hex

RUNTIME_PATH = Path(hookrt.__file__)
RUNTIME_BEGIN = hookrt.RUNTIME_BEGIN
RUNTIME_END = hookrt.RUNTIME_END
HEADER_KEYS = (
    "guard_id",
    "guard_sha256",
    "harness_version",
    "harness_digest",
    "decision_receipt_sha256",
    "decision_signers",
    "decided_at",
    "sandbox_child_sha256",
    "hook_runtime_sha256",
    "sandbox_require",
)


class ExportError(ValueError):
    """The guard is not admitted, or a hook file does not match its own provenance."""


def text_sha256(text: str) -> str:
    return sha256_hex(text.encode("utf-8"))


@dataclass(frozen=True)
class Provenance:
    guard_id: str
    guard_sha256: str
    harness_version: str
    harness_digest: str
    decision_receipt_sha256: str
    decision_signers: str
    decided_at: str

    def header(self, child_sha256: str, runtime_sha256: str, require: Sequence[str]) -> dict[str, str]:
        return {
            **asdict(self),
            "sandbox_child_sha256": child_sha256,
            "hook_runtime_sha256": runtime_sha256,
            "sandbox_require": ",".join(require) or "none",
        }


def _signers(envelope: bytes, trusted: Mapping[str, Ed25519PublicKey]) -> list[str]:
    signed = []
    for keyid, key in sorted(trusted.items()):
        try:
            verify(envelope, {keyid: key})
        except ReceiptError:
            continue
        signed.append(keyid)
    return signed


def verify_admission(
    harness: Harness, envelope: bytes, trusted: Mapping[str, Ed25519PublicKey]
) -> dict[str, Any]:
    """The decision payload, or ExportError unless it verifies under a trusted key and admits exactly
    this harness."""
    try:
        decision = verify(envelope, trusted)
    except ReceiptError as exc:
        raise ExportError(f"decision receipt does not verify: {exc}") from exc
    if not harness.verify_digest():
        raise ExportError("harness content does not match its digest")
    if decision.get("admitted") is not True:
        raise ExportError("the decision did not admit the harness")
    if decision.get("candidate") != harness.version or decision.get("candidate_digest") != harness.digest:
        raise ExportError(
            f"the decision admits {decision.get('candidate')!r}, not harness {harness.version!r}"
        )
    return decision


def admitted_guard(
    harness: Harness, guard_id: str, envelope: bytes, trusted: Mapping[str, Ed25519PublicKey]
) -> tuple[Guard, Provenance]:
    """The guard and its provenance, or ExportError when the receipt does not admit this harness."""
    decision = verify_admission(harness, envelope, trusted)
    guard = next((g for g in harness.guards if g.id == guard_id), None)
    if guard is None:
        raise ExportError(f"harness {harness.version} has no guard {guard_id!r}")
    provenance = Provenance(
        guard_id=guard.id,
        guard_sha256=text_sha256(guard.source),
        harness_version=harness.version,
        harness_digest=harness.digest,
        decision_receipt_sha256=sha256_hex(envelope),
        decision_signers=",".join(_signers(envelope, trusted)),
        decided_at=str(decision.get("decided_at", "")),
    )
    return guard, provenance


def py_literal(text: str) -> str:
    """A Python literal for `text` that `ruff format` leaves as it is: a raw triple-quoted string when
    that is exact (double quotes first), else a double-quoted escaped string."""
    if "\n" in text:
        for quote in ('"""', "'''"):
            if quote not in text and not text.endswith(("\\", quote[0])):
                literal = "r" + quote + text + quote
                if ast.literal_eval(literal) == text:
                    return literal
    return json.dumps(text)


def tuple_literal(items: Sequence[str]) -> str:
    inner = ", ".join(py_literal(x) for x in items)
    return f"({inner},)" if len(items) == 1 else f"({inner})"


def render_hook(guard: Guard, provenance: Provenance, limits: GuardLimits | None = None) -> str:
    """The hook file's text."""
    limits = limits or GuardLimits()
    child = CHILD_PATH.read_text(encoding="utf-8")
    runtime = RUNTIME_PATH.read_text(encoding="utf-8")
    header = provenance.header(text_sha256(child), text_sha256(runtime), limits.require)
    lines = [
        "#!/usr/bin/env python3",
        "# Ratchet guard hook for Claude Code (PreToolUse). Generated by ratchet.export.claude_code.",
        *(f"# {key}: {header[key]}" for key in HEADER_KEYS),
        "# Install: see examples/claude-code/README.md in the Ratchet repository.",
        "# Any edit to the guard, the runtime or the sandbox child below makes the hook deny every call.",
        "# `python -m ratchet.export.claude_code --verify <this file>` checks every digest above.",
        "# The sandbox child is embedded byte for byte, so its long lines cannot be wrapped here:",
        "# ruff: noqa: E501",
        RUNTIME_BEGIN,
        runtime.rstrip("\n"),
        "",
        "",
        RUNTIME_END,
        "",
        f"GUARD_ID = {py_literal(guard.id)}",
        f"GUARD_SHA256 = {py_literal(provenance.guard_sha256)}",
        f"CHILD_SHA256 = {py_literal(header['sandbox_child_sha256'])}",
        f"RUNTIME_SHA256 = {py_literal(header['hook_runtime_sha256'])}",
        f"GUARD_SOURCE = {py_literal(guard.source)}",
        f"CHILD_SOURCE = {py_literal(child)}",
        "CONFIG = HookConfig(",
        "    spec=SandboxSpec(",
        "        guard_id=GUARD_ID,",
        "        guard_source=GUARD_SOURCE,",
        "        child_source=CHILD_SOURCE,",
        f"        cpu_s={limits.cpu_s!r},",
        f"        memory_mb={limits.memory_mb!r},",
        f"        call_timeout_s={limits.call_timeout_s!r},",
        f"        startup_timeout_s={limits.startup_timeout_s!r},",
        f"        require={tuple_literal(limits.require)},",
        "    ),",
        "    guard_sha256=GUARD_SHA256,",
        "    child_sha256=CHILD_SHA256,",
        "    runtime_sha256=RUNTIME_SHA256,",
        "    hook_file=__file__,",
        ")",
        "",
        'if __name__ == "__main__":',
        "    sys.exit(run_hook(CONFIG, sys.stdin, sys.stdout, sys.stderr))",
        "",
    ]
    return "\n".join(lines)


def read_header(text: str) -> dict[str, str]:
    header: dict[str, str] = {}
    for line in text.splitlines():
        if line == RUNTIME_BEGIN:
            break
        key, sep, value = line.removeprefix("# ").partition(": ")
        if line.startswith("# ") and sep and key in HEADER_KEYS:
            header[key] = value
    return header


def _constants(tree: ast.Module) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return out


def verify_hook(text: str) -> dict[str, str]:
    """Check that a hook file matches its own header: guard, sandbox child and runtime digests. Returns the
    header. This proves the file is the one the receipt names; verifying the receipt itself needs the
    envelope and the signer's public key."""
    header = read_header(text)
    missing = [k for k in HEADER_KEYS if k not in header]
    if missing:
        raise ExportError(f"hook header is missing {missing}")
    runtime = hookrt.embedded_runtime(text)
    if runtime is None:
        raise ExportError("hook runtime markers are missing")
    constants = _constants(ast.parse(text))
    actual = {
        "guard_id": constants.get("GUARD_ID"),
        "guard_sha256": text_sha256(str(constants.get("GUARD_SOURCE"))),
        "sandbox_child_sha256": text_sha256(str(constants.get("CHILD_SOURCE"))),
        "hook_runtime_sha256": text_sha256(runtime),
    }
    pinned = {
        "GUARD_SHA256": "guard_sha256",
        "CHILD_SHA256": "sandbox_child_sha256",
        "RUNTIME_SHA256": "hook_runtime_sha256",
    }
    wrong = sorted(k for k, v in actual.items() if header[k] != v)
    wrong += sorted(name for name, key in pinned.items() if constants.get(name) != header[key])
    if wrong:
        raise ExportError(f"hook does not match its header: {wrong}")
    return header


def settings_entry(command: str, hook_path: str, matcher: str = "*") -> dict[str, Any]:
    """The `.claude/settings.json` fragment that installs a hook. `onFailure: block` makes a hook that
    cannot even start deny the call."""
    handler = {"type": "command", "command": command, "args": [hook_path], "onFailure": "block"}
    return {"hooks": {"PreToolUse": [{"matcher": matcher, "hooks": [handler]}]}}


def export_hook(
    harness: Harness,
    guard_id: str,
    envelope: bytes,
    trusted: Mapping[str, Ed25519PublicKey],
    out: Path,
    limits: GuardLimits | None = None,
) -> Provenance:
    guard, provenance = admitted_guard(harness, guard_id, envelope, trusted)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_hook(guard, provenance, limits), encoding="utf-8")
    out.chmod(0o755)
    return provenance


# --- loading and the command line ---------------------------------------------------------------------------


def harness_from_json(doc: Mapping[str, Any]) -> Harness:
    return Harness(
        version=doc["version"],
        parent=doc["parent"],
        system_prompt=doc["system_prompt"],
        procedures=[Procedure(**p) for p in doc["procedures"]],
        guards=[Guard(**g) for g in doc["guards"]],
        tool_allowlist=list(doc["tool_allowlist"]),
        created_by=doc["created_by"],
        digest=doc["digest"],
    )


def public_key(hex_key: str) -> Ed25519PublicKey:
    try:
        return Ed25519PublicKey.from_public_bytes(bytes.fromhex(hex_key.strip()))
    except ValueError as exc:
        raise ExportError("a trusted key must be a 32-byte Ed25519 public key in hex") from exc


def trusted_keys(hex_keys: Sequence[str]) -> dict[str, Ed25519PublicKey]:
    keys = [public_key(k) for k in hex_keys]
    return {key_id(k): k for k in keys}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ratchet.export.claude_code", description=__doc__.split("\n")[0]
    )
    p.add_argument("--verify", type=Path, help="check a generated hook against its header and exit")
    p.add_argument("--harness", type=Path, help="harness JSON (a Harness as written by Ratchet)")
    p.add_argument("--guard-id")
    p.add_argument("--receipt", type=Path, help="the DSSE decision envelope that admitted the harness")
    p.add_argument("--trusted-key", action="append", default=[], help="Ed25519 public key, hex; repeatable")
    p.add_argument("--out", type=Path)
    p.add_argument(
        "--require",
        default=",".join(GuardLimits().require),
        help="sandbox layers that must take effect, comma separated ('' for none, e.g. on macOS)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.verify is not None:
            header = verify_hook(args.verify.read_text(encoding="utf-8"))
            print(json.dumps(header, indent=2))
            return 0
        if not (args.harness and args.guard_id and args.receipt and args.out and args.trusted_key):
            raise ExportError("--harness, --guard-id, --receipt, --trusted-key and --out are required")
        harness = harness_from_json(json.loads(args.harness.read_text(encoding="utf-8")))
        require = tuple(x for x in args.require.split(",") if x)
        provenance = export_hook(
            harness,
            args.guard_id,
            args.receipt.read_bytes(),
            trusted_keys(args.trusted_key),
            args.out,
            GuardLimits(require=require),
        )
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(provenance), indent=2))
    hook_path = str(args.out) if args.out.is_absolute() else "${CLAUDE_PROJECT_DIR}/" + args.out.as_posix()
    print(json.dumps(settings_entry("python3", hook_path), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
