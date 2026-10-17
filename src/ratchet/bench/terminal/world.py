"""A terminal task as a Ratchet world: two tools over one container, and a bounded state view for guards.

Tools.
  run_command(cmd, timeout_s)  runs `cmd` with bash (sh when the image has no bash) in the container. The
                               working directory and exported variables carry over from one call to the next,
                               as in a terminal. Output is stdout and stderr merged, cut to its first and last
                               bytes when long. A command still running at `timeout_s` is killed with its
                               process group; a process started in the background (`cmd &`) keeps running.
  write_file(path, content)    writes `content` to `path` (relative paths start at the working directory),
                               creating parent directories.

The state view a guard reads (`TerminalView`), bounded so every recorded trace stays small:
  cwd, last_exit_code, last_timed_out, calls, seconds_left   the session
  files_changed            files modified since the task started, outside package and cache trees, each with
                           size and SHA-256 (no hash above 1 MiB), at most `MAX_FILES` by path
  files_changed_total      how many such files there are
  bulk_changes             counts of changed files under package and cache trees, by top-level directory
  processes                processes started since the task started that are still running (pid, command)
The view is refreshed after every call by a probe script run in the same container.

The agent's time limit is the task's own agent timeout, counted in wall time from the world's start; a
command's timeout is cut to the time left.
"""

from __future__ import annotations

import copy
import hashlib
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ratchet_cases.types import ToolCall, ToolResult

from ratchet.bench.terminal.container import Container
from ratchet.canonical import canonical_json

RUN_COMMAND = "run_command"
WRITE_FILE = "write_file"
TOOL_NAMES = (RUN_COMMAND, WRITE_FILE)
STATE_DIR = "/tmp/.ratchet"
MAX_FILES = 40
MAX_PROCESSES = 20
MAX_TEXT = 240
HASH_LIMIT = 1 << 20
OUTPUT_HEAD = 2000
OUTPUT_TAIL = 8000
DEFAULT_TIMEOUT_S = 120.0
MAX_TIMEOUT_S = 1800.0
PROBE_TIMEOUT_S = 180.0

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": RUN_COMMAND,
            "description": (
                "Run a shell command (bash) in the task's Linux container and return its exit code and "
                "output. The working directory and exported variables persist between calls. Long output "
                "is cut in the middle. A command still running after timeout_s seconds is killed; start "
                "long-lived servers in the background with their output redirected to a file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "The command line to run."},
                    "timeout_s": {
                        "type": "number",
                        "description": f"Seconds until the command is killed (default {DEFAULT_TIMEOUT_S:g})",
                    },
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": WRITE_FILE,
            "description": (
                "Write text to a file in the container, replacing it if it exists and creating parent "
                "directories. Relative paths start at the current working directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The file to write."},
                    "content": {"type": "string", "description": "The full new content of the file."},
                },
                "required": ["path", "content"],
            },
        },
    },
]

# Changed files under these trees are counted, not listed: package installs touch thousands of them.
_BULK_PATTERNS = (
    "/usr/*|/var/*|/opt/*|/etc/*|/root/.cache/*|/root/.local/*|/root/.cargo/*|/root/.rustup/*|/root/.npm/*"
    "|*/site-packages/*|*/dist-packages/*|*/node_modules/*|*/__pycache__/*|*/.git/objects/*|*/.venv/*"
)

SETUP_SCRIPT = f"""d={STATE_DIR}
mkdir -p "$d"
sh_=$(command -v bash || command -v sh)
echo "$sh_" > "$d/shell"
pwd > "$d/cwd"
"$sh_" -c 'export -p' > "$d/env"
for p in /proc/[0-9]*; do echo "${{p#/proc/}}"; done > "$d/base_pids"
touch "$d/mark"
echo ready
"""

PROBE = f"""d={STATE_DIR}
echo "cwd $(cat "$d/cwd" 2>/dev/null)"
if command -v awk >/dev/null 2>&1; then
  find / -xdev \\( -path /proc -o -path /sys -o -path /dev -o -path /run -o -path "$d" \\) -prune \\
    -o -type f -newer "$d/mark" -print 2>/dev/null | awk '
      {{ p = $0 }}
      p ~ /\\t|\\r/ {{ next }}
      {{
        bulk = 0
        n = split("{_BULK_PATTERNS}", pats, "|")
        for (i = 1; i <= n; i++) {{
          g = pats[i]; gsub(/\\./, "\\\\.", g); gsub(/\\*/, ".*", g)
          if (p ~ ("^" g "$")) {{ bulk = 1; break }}
        }}
        if (bulk) {{ split(p, parts, "/"); c["/" parts[2]]++ }} else {{ print "L\\t" p }}
      }}
      END {{ for (k in c) print "B\\t" c[k] "\\t" k }}' > "$d/changed"
  grep -c '^L' "$d/changed" | sed 's/^/total /'
  grep '^B' "$d/changed"
  grep '^L' "$d/changed" | cut -f2 | sort | head -n {MAX_FILES} | while IFS= read -r f; do
    s=$(wc -c < "$f" 2>/dev/null | tr -d ' ')
    h=-
    if [ -n "$s" ] && [ "$s" -le {HASH_LIMIT} ] && command -v sha256sum >/dev/null 2>&1; then
      h=$(sha256sum < "$f" | cut -c1-64)
    fi
    printf 'F\\t%s\\t%s\\t%s\\n' "${{s:--1}}" "$h" "$f"
  done
else
  echo "nofind"
fi
self=$$
for p in /proc/[0-9]*; do
  pid=${{p#/proc/}}
  [ "$pid" = "$self" ] && continue
  grep -qx "$pid" "$d/base_pids" 2>/dev/null && continue
  c=$(tr '\\000' ' ' < "$p/cmdline" 2>/dev/null)
  case "$c" in ''|*"$d"*) continue ;; esac
  printf 'P\\t%s\\t%s\\n' "$pid" "$c"
done
"""


def command_script(timeout_s: float, marker: str) -> str:
    """Run the command stored in the state directory, persist cwd and env, cut its output, then probe."""
    t = max(1, int(timeout_s))
    return f"""d={STATE_DIR}
sh_=$(cat "$d/shell")
rm -f "$d/cwd.new" "$d/env.new"
inner='cd "$(cat {STATE_DIR}/cwd)" 2>/dev/null || cd /
. {STATE_DIR}/env 2>/dev/null
trap "pwd > {STATE_DIR}/cwd.new; export -p > {STATE_DIR}/env.new" EXIT
. {STATE_DIR}/cmd'
start=$(date +%s)
if command -v timeout >/dev/null 2>&1; then
  timeout -k 5 {t} "$sh_" -c "$inner" > "$d/out" 2>&1 < /dev/null
else
  "$sh_" -c "$inner" > "$d/out" 2>&1 < /dev/null
fi
rc=$?
elapsed=$(( $(date +%s) - start ))
[ -s "$d/cwd.new" ] && mv "$d/cwd.new" "$d/cwd"
[ -s "$d/env.new" ] && mv "$d/env.new" "$d/env"
size=$(wc -c < "$d/out" | tr -d ' ')
if [ "$size" -gt {OUTPUT_HEAD + OUTPUT_TAIL} ]; then
  head -c {OUTPUT_HEAD} "$d/out"
  printf '\\n[... %s bytes omitted ...]\\n' "$((size - {OUTPUT_HEAD + OUTPUT_TAIL}))"
  tail -c {OUTPUT_TAIL} "$d/out"
else
  cat "$d/out"
fi
printf '\\n%s\\n' "{marker}"
echo "rc $rc"
echo "elapsed $elapsed"
{PROBE}"""


def _clip(text: str, limit: int = MAX_TEXT) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass
class Probe:
    """What the probe script reported."""

    cwd: str = "/"
    files: list[dict[str, Any]] = field(default_factory=list)
    files_total: int = 0
    bulk: dict[str, int] = field(default_factory=dict)
    processes: list[dict[str, Any]] = field(default_factory=list)
    processes_total: int = 0
    exit_code: int | None = None
    elapsed_s: int = 0


def parse_probe(text: str) -> Probe:
    out = Probe()
    for line in text.splitlines():
        if line.startswith("cwd "):
            out.cwd = line[4:].strip() or "/"
        elif line.startswith("rc "):
            out.exit_code = int(line[3:].strip())
        elif line.startswith("elapsed "):
            out.elapsed_s = int(line[8:].strip() or 0)
        elif line.startswith("total "):
            out.files_total = int(line[6:].strip() or 0)
        elif line.startswith("B\t"):
            _, count, top = line.split("\t", 2)
            out.bulk[top] = int(count)
        elif line.startswith("F\t"):
            _, size, digest, path = line.split("\t", 3)
            out.files.append(
                {"path": _clip(path), "size": int(size), "sha256": None if digest == "-" else digest}
            )
        elif line.startswith("P\t"):
            _, pid, cmd = line.split("\t", 2)
            out.processes_total += 1
            if len(out.processes) < MAX_PROCESSES:
                out.processes.append({"pid": int(pid), "cmd": _clip(cmd.strip())})
    return out


@dataclass(frozen=True)
class TerminalView:
    data: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)


Clock = Callable[[], float]


class TerminalWorld:
    """One task's container in use by one agent run."""

    def __init__(
        self,
        container: Container,
        task_name: str,
        agent_timeout_s: float,
        *,
        clock: Clock = time.monotonic,
    ) -> None:
        self.container = container
        self.task_name = task_name
        self.agent_timeout_s = agent_timeout_s
        self.clock = clock
        self.calls = 0
        self.last_exit_code: int | None = None
        self.last_timed_out = False
        self.probe = Probe()
        self.probe_errors = 0
        setup = container.exec(SETUP_SCRIPT, PROBE_TIMEOUT_S)
        if setup.exit_code != 0 or "ready" not in setup.output:
            raise RuntimeError(f"world setup failed in {container.name}: {setup.output[-300:]}")
        self.started = clock()
        self._refresh()

    # -- the surface the worker loop uses --------------------------------------------------------------

    def tools(self) -> list[dict[str, Any]]:
        return copy.deepcopy(TOOL_SPECS)

    def tool_names(self) -> list[str]:
        return sorted(TOOL_NAMES)

    @property
    def seconds_left(self) -> float:
        return self.agent_timeout_s - (self.clock() - self.started)

    @property
    def expired(self) -> bool:
        return self.seconds_left <= 0

    def call(self, call: ToolCall) -> ToolResult:
        self.calls += 1
        if self.expired:
            return ToolResult(call.call_id, False, None, "the agent's time limit for this task has passed")
        if call.name == RUN_COMMAND:
            return self._run_command(call)
        if call.name == WRITE_FILE:
            return self._write_file(call)
        return ToolResult(call.call_id, False, None, f"unknown tool {call.name!r}")

    def view(self) -> TerminalView:
        p = self.probe
        return TerminalView(
            {
                "task": self.task_name,
                "cwd": p.cwd,
                "last_exit_code": self.last_exit_code,
                "last_timed_out": self.last_timed_out,
                "calls": self.calls,
                "seconds_left": max(0, int(self.seconds_left)),
                "files_changed": list(p.files),
                "files_changed_total": p.files_total,
                "bulk_changes": dict(sorted(p.bulk.items())),
                "processes": list(p.processes),
                "processes_total": p.processes_total,
            }
        )

    def digest(self) -> str:
        p = self.probe
        state = {"cwd": p.cwd, "files": p.files, "bulk": p.bulk, "processes": [q["cmd"] for q in p.processes]}
        return hashlib.sha256(canonical_json(state)).hexdigest()

    # -- tools --------------------------------------------------------------------------------------------

    def _run_command(self, call: ToolCall) -> ToolResult:
        cmd = call.args.get("cmd")
        if not isinstance(cmd, str) or not cmd.strip():
            return ToolResult(call.call_id, False, None, "run_command needs a non-empty string 'cmd'")
        raw = call.args.get("timeout_s", DEFAULT_TIMEOUT_S)
        try:
            asked = float(raw)
        except (TypeError, ValueError):
            asked = DEFAULT_TIMEOUT_S
        timeout = max(1.0, min(asked if asked > 0 else DEFAULT_TIMEOUT_S, MAX_TIMEOUT_S, self.seconds_left))
        stored = self.container.put_file(f"{STATE_DIR}/cmd", cmd.encode("utf-8"))
        if stored.exit_code != 0:
            return ToolResult(
                call.call_id, False, None, f"could not stage the command: {stored.output[-200:]}"
            )
        marker = f"@@ratchet-{uuid.uuid4().hex}@@"
        done = self.container.exec(command_script(timeout, marker), timeout + 30 + PROBE_TIMEOUT_S)
        output, sep, tail = done.output.partition("\n" + marker + "\n")
        if not sep:
            # The client was killed before the script finished; the command's output is what arrived.
            self.last_exit_code, self.last_timed_out = None, True
            self._refresh()
            data = {
                "exit_code": None,
                "timed_out": True,
                "output": output[-OUTPUT_TAIL:],
                "cwd": self.probe.cwd,
            }
            return ToolResult(call.call_id, False, data, "the command did not finish")
        probe = parse_probe(tail)
        self.probe = probe
        code = probe.exit_code
        # `timeout` exits 124 (TERM) or 137 (KILL after the grace period) when it fired.
        timed_out = code in (124, 137) and probe.elapsed_s >= int(timeout) - 1
        self.last_exit_code, self.last_timed_out = code, timed_out
        result: dict[str, Any] = {
            "exit_code": code,
            "timed_out": timed_out,
            "output": output,
            "cwd": probe.cwd,
        }
        if timed_out:
            return ToolResult(call.call_id, False, result, f"killed after {timeout:g} s")
        if code != 0:
            return ToolResult(call.call_id, False, result, f"exit code {code}")
        return ToolResult(call.call_id, True, result, None)

    def _write_file(self, call: ToolCall) -> ToolResult:
        path, content = call.args.get("path"), call.args.get("content")
        if not isinstance(path, str) or not path.strip() or not isinstance(content, str):
            return ToolResult(call.call_id, False, None, "write_file needs string 'path' and 'content'")
        target = path if path.startswith("/") else f"{self.probe.cwd.rstrip('/')}/{path}"
        data = content.encode("utf-8")
        done = self.container.put_file(target, data)
        self._refresh()
        if done.exit_code != 0:
            return ToolResult(call.call_id, False, None, f"could not write {target}: {done.output[-200:]}")
        digest = hashlib.sha256(data).hexdigest()
        return ToolResult(call.call_id, True, {"path": target, "bytes": len(data), "sha256": digest}, None)

    def _refresh(self) -> None:
        done = self.container.exec(PROBE, PROBE_TIMEOUT_S)
        if done.exit_code != 0 and not done.output:
            self.probe_errors += 1
            return
        exit_code = self.probe.exit_code
        self.probe = parse_probe(done.output)
        self.probe.exit_code = exit_code
