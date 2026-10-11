"""Ratchet guard: never force-push to, or delete, main or master.

Runs in the Ratchet guard sandbox: `check(call, view, history)` sees one tool call. A string blocks the call
and is shown to the agent; None leaves the call to the normal permission flow. Only modules the sandbox
preloads can be imported (`re`, `string`, ...); `shlex` is not one, so the shell words are split here.

A push targets main when a refspec names it (`main`, `HEAD:main`, `+x:refs/heads/main`), when it
pushes everything (`--all`, `--mirror`), or when it names no refspec and the checked-out branch,
`view["git_branch"]`, is main. It is forced by `-f`, `--force`, `--force-with-lease` or a `+` refspec.
Pushes inside `$(...)`, backticks, `sh -c`/`bash -c` scripts and `eval` are found too. Text that merely
mentions a push, in a commit message, an echo, a grep or single quotes, is not a push.
"""

from typing import Any

PROTECTED = ("main", "master")
FORCE_FLAGS = ("--force", "--force-with-lease")
DELETE_FLAGS = ("--delete", "-d")
EVERYTHING_FLAGS = ("--all", "--mirror", "--branches")
GIT_VALUE_OPTIONS = ("-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path")
PUSH_VALUE_OPTIONS = ("-o", "--push-option", "--repo", "--receive-pack", "--exec")
WRAPPERS = ("sudo", "command", "env", "nohup", "time", "exec", "nice")
SHELLS = ("sh", "bash", "zsh", "dash", "ksh")
SEPARATORS = ";|&\n()"
MAX_DEPTH = 8


def closing(text: str, start: int) -> int:
    """Index of the parenthesis that closes a `$(` whose body starts at `start`, or len(text)."""
    depth = 1
    for i in range(start, len(text)):
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        if depth == 0:
            return i
    return len(text)


class Scanner:
    """Shell words per simple command, plus the bodies of command substitutions. Quotes and backslashes
    are honoured: nothing inside single quotes runs; `$(...)` and backticks run inside double quotes."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.commands: list[list[str]] = []
        self.substitutions: list[str] = []
        self.words: list[str] = []
        self.word: list[str] = []
        self.started = False
        i = 0
        quote = ""
        while i < len(text):
            i, quote = self.step(i, quote)
        self.end_word()
        self.end_command()

    def end_word(self) -> None:
        if self.started:
            self.words.append("".join(self.word))
        self.word, self.started = [], False

    def end_command(self) -> None:
        if self.words:
            self.commands.append(self.words)
        self.words = []

    def take(self, c: str) -> None:
        self.word.append(c)
        self.started = True

    def step(self, i: int, quote: str) -> tuple[int, str]:
        text, c = self.text, self.text[i]
        if quote == "'":
            if c == "'":
                return i + 1, ""
            self.take(c)
            return i + 1, quote
        if text.startswith("$(", i):
            end = closing(text, i + 2)
            self.substitutions.append(text[i + 2 : end])
            self.started = True
            return end + 1, quote
        if c == "`":
            end = text.find("`", i + 1)
            end = len(text) if end < 0 else end
            self.substitutions.append(text[i + 1 : end])
            self.started = True
            return end + 1, quote
        if c == "\\" and i + 1 < len(text):
            self.take(text[i + 1])
            return i + 2, quote
        if quote == '"':
            if c == '"':
                return i + 1, ""
            self.take(c)
            return i + 1, quote
        if c in "'\"":
            self.started = True
            return i + 1, c
        if c.isspace() or c in SEPARATORS:
            self.end_word()
            if c in SEPARATORS:
                self.end_command()
            return i + 1, quote
        self.take(c)
        return i + 1, quote


def command_start(words: list[str]) -> int:
    """Index of the command word, after `VAR=value` assignments and wrappers such as `sudo`."""
    i = 0
    while i < len(words) and (("=" in words[i] and not words[i].startswith("-")) or words[i] in WRAPPERS):
        i += 1
    return i


def push_args(words: list[str]) -> list[str] | None:
    """The arguments after `git ... push`, or None when the command is not a git push."""
    i = command_start(words)
    if i >= len(words) or words[i].rsplit("/", 1)[-1] != "git":
        return None
    i += 1
    while i < len(words) and words[i].startswith("-"):
        i += 2 if words[i] in GIT_VALUE_OPTIONS else 1
    if i >= len(words) or words[i] != "push":
        return None
    return words[i + 1 :]


def nested_scripts(words: list[str]) -> list[str]:
    """Scripts a command runs as shell code: `sh -c <script>` (and `-lc` and the like), and `eval`."""
    i = command_start(words)
    if i >= len(words):
        return []
    name = words[i].rsplit("/", 1)[-1]
    if name == "eval":
        return [" ".join(words[i + 1 :])]
    if name not in SHELLS:
        return []
    for j in range(i + 1, len(words) - 1):
        flag = words[j]
        if flag.startswith("-") and not flag.startswith("--") and "c" in flag:
            return [words[j + 1]]
    return []


def destination(refspec: str, branch: str) -> str:
    dst = refspec.lstrip("+")
    if ":" in dst:
        dst = dst.split(":", 1)[1]
    if dst in ("HEAD", "@"):
        dst = branch
    return dst.removeprefix("refs/heads/")


def verdict(args: list[str], branch: str) -> str | None:
    forced = deleting = everything = False
    positional: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in PUSH_VALUE_OPTIONS:
            i += 2
            continue
        if a.split("=", 1)[0] in FORCE_FLAGS or (a.startswith("-") and not a.startswith("--") and "f" in a):
            forced = True
        elif a in DELETE_FLAGS:
            deleting = True
        elif a in EVERYTHING_FLAGS:
            everything = True
        elif not a.startswith("-"):
            positional.append(a)
        i += 1
    refspecs = positional[1:]
    if not refspecs and not everything:
        refspecs = [branch or "?"]
    hits = sorted({destination(r, branch) for r in refspecs} & set(PROTECTED)) or (
        list(PROTECTED) if everything else []
    )
    plus = any(r.startswith("+") and destination(r, branch) in PROTECTED for r in refspecs)
    empty_src = any(r.startswith(":") and destination(r, branch) in PROTECTED for r in refspecs)
    if hits and (forced or plus):
        return (
            f"force push to {hits[0]} rewrites shared history; push a branch and open a pull request instead"
        )
    if hits and (deleting or empty_src):
        return f"deleting {hits[0]} is not allowed"
    if forced and not branch and not positional[1:] and not everything:
        return "force push with no refspec and an unknown current branch; name the branch you mean to push"
    return None


def reason_for(text: str, branch: str, depth: int = 0) -> str | None:
    if depth > MAX_DEPTH:
        return "shell nesting too deep to check; run the push directly"
    scan = Scanner(text)
    for words in scan.commands:
        args = push_args(words)
        reason = verdict(args, branch) if args is not None else None
        for script in [] if reason else nested_scripts(words):
            reason = reason or reason_for(script, branch, depth + 1)
        if reason is not None:
            return reason
    for body in scan.substitutions:
        reason = reason_for(body, branch, depth + 1)
        if reason is not None:
            return reason
    return None


def check(call: Any, view: Any, history: Any) -> str | None:
    if call.name != "Bash":
        return None
    command = call.args.get("command")
    if not isinstance(command, str):
        return None
    return reason_for(command, str(view.get("git_branch") or ""))
