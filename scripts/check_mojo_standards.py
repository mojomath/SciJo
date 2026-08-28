#!/usr/bin/env python3
"""Check `.mojo` files against SciJo's file-layout standards.

Run this before opening a PR to get a clean, file-by-file list of every
place a file deviates from the standard: header format, module docstring
format, section-separator format, TODO/FIXME formatting, and — for every
`def`/`fn` — whether its docstring documents the parameters, args, raises,
and return value that its signature actually has.

Standards are documented in `docs/developer_guide.md`.

This is a *reporting* tool. It never rewrites files (use
`organize_mojo_imports.py` for the import-reorg autofix). Exit code is 0 if
clean, 1 if any file has findings, 2 on a usage/parse error.

Usage:
    python3 scripts/check_mojo_standards.py scijo
    python3 scripts/check_mojo_standards.py scijo/optimize/root_scalar.mojo
    python3 scripts/check_mojo_standards.py scijo --only headers,imports
    python3 scripts/check_mojo_standards.py scijo --quiet   # summary only
    python3 scripts/check_mojo_standards.py scijo --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ===----------------------------------------------------------------------=== #
# Constants
# ===----------------------------------------------------------------------=== #

SEPARATOR_WIDTH = 80  # "# ===" + 70 "-" + "=== #"
EXPECTED_SEPARATOR = "# ===" + "-" * 70 + "=== #"
# Line 2 (`# SciJo: <description>`) is per-file and checked separately.
# Line 3 is the fixed license line.
LICENSE_LINES = [
    "# Distributed under the Apache 2.0 License.",
]

# Files with a different, legitimate header (vendored/adapted from another
# project) — excluded from the header/license check entirely rather than
# flagged as non-conformant.
HEADER_EXEMPT_FILES: set[str] = set()

# Canonical docstring section order (developer_guide.md "Docstring Format").
# Sections not in this list are left alone (e.g. `Attributes:`, `Constants:`
# on struct docstrings).
SECTION_ORDER = [
    "Parameters",
    "Args",
    "Constraints",
    "Returns",
    "Raises",
    "NOTES",
    "References",
    "Examples",
]
# Singular/alternate spellings seen in the wild that should be the
# canonical form actually used across the codebase.
SECTION_CANONICAL = {
    "Parameter": "Parameters",
    "Parameters": "Parameters",
    "Arg": "Args",
    "Args": "Args",
    "Argument": "Args",
    "Arguments": "Args",
    "Raise": "Raises",
    "Raises": "Raises",
    "Return": "Returns",
    "Returns": "Returns",
    "Constraint": "Constraints",
    "Constraints": "Constraints",
    "Note": "NOTES",
    "Notes": "NOTES",
    "NOTES": "NOTES",
    "Reference": "References",
    "References": "References",
    "Example": "Examples",
    "Examples": "Examples",
}
SECTION_HEADER_RE = re.compile(
    r"^(?P<indent>\s*)(?P<name>[A-Za-z]+):\s*$"
)

TODO_RE = re.compile(r"#\s*(TODO|FIXME)\b")
TODO_OK_RE = re.compile(r"#\s*(TODO|FIXME):\s+[A-Z0-9`@]")

DEF_RE = re.compile(r"^(?P<indent>\s*)(?:@\w+(\(.*\))?\s*\n\s*)*(?P<kind>def|fn)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)")
STRUCT_RE = re.compile(r"^(?P<indent>\s*)(?:@\w+(\(.*\))?\s*\n\s*)*(?:struct|trait)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)")


# ===----------------------------------------------------------------------=== #
# Data model
# ===----------------------------------------------------------------------=== #


@dataclass
class Finding:
    line: int
    category: str
    message: str


@dataclass
class FileReport:
    path: Path
    findings: list[Finding] = field(default_factory=list)

    def add(self, line: int, category: str, message: str) -> None:
        self.findings.append(Finding(line, category, message))


# ===----------------------------------------------------------------------=== #
# Low-level helpers
# ===----------------------------------------------------------------------=== #


def strip_backticks(s: str) -> str:
    return s.replace("`", "")


def is_separator_line(line: str) -> bool:
    stripped = line.rstrip("\n").lstrip()
    return bool(re.match(r"^#\s*=+[-=]*=+\s*#?\s*$", stripped)) and "=" in stripped


def separator_indent(line: str) -> str:
    """Leading whitespace before the '#' of a separator line (separators
    inside a struct/trait body are indented to match; module-level ones are
    not)."""
    return line[: len(line) - len(line.lstrip())]


def docstring_spans(lines: list[str]) -> list[tuple[int, int]]:
    """Return (start, end) 0-indexed line ranges of every triple-quoted
    docstring/string block, inclusive of the quote lines."""
    spans = []
    i = 0
    n = len(lines)
    in_str = False
    start = None
    while i < n:
        line = lines[i]
        count = line.count('"""')
        if not in_str:
            if count >= 1:
                start = i
                if count >= 2:
                    spans.append((i, i))
                else:
                    in_str = True
        else:
            if count >= 1:
                spans.append((start, i))
                in_str = False
        i += 1
    return spans


# ===----------------------------------------------------------------------=== #
# Check: file header (developer_guide.md "File Header")
# ===----------------------------------------------------------------------=== #


def check_header(lines: list[str], report: FileReport) -> int:
    """Returns the 0-indexed line number right after the header block."""
    if not lines:
        report.add(1, "header", "File is empty.")
        return 0

    if not is_separator_line(lines[0]):
        report.add(1, "header", "File must start with the header separator line.")
        return 0
    if lines[0].rstrip("\n") != EXPECTED_SEPARATOR:
        report.add(
            1,
            "header",
            f"Header separator malformed (expected exactly 80 chars, "
            f"'# ===' + 70 dashes + '=== #'): {lines[0].rstrip()!r}",
        )

    if len(lines) < 2 or not re.match(r"^# SciJo: \S", lines[1]):
        report.add(
            2,
            "header",
            "Second header line should read '# SciJo: <short description>'.",
        )

    idx = 2
    for expected in LICENSE_LINES:
        if idx >= len(lines) or lines[idx].rstrip("\n") != expected:
            report.add(
                idx + 1,
                "header",
                f"Header line does not match expected license text: {expected!r}",
            )
        idx += 1

    if idx >= len(lines) or not is_separator_line(lines[idx]):
        report.add(idx + 1, "header", "Header must close with a separator line.")
    else:
        if lines[idx].rstrip("\n") != EXPECTED_SEPARATOR:
            report.add(
                idx + 1,
                "header",
                f"Closing header separator malformed: {lines[idx].rstrip()!r}",
            )
        idx += 1

    # No blank line between header and docstring.
    if idx < len(lines) and lines[idx].strip() == "":
        report.add(
            idx + 1,
            "header",
            "Blank line between header and module docstring (should be none).",
        )

    return idx


# ===----------------------------------------------------------------------=== #
# Check: module docstring (developer_guide.md "File Header" / "Docstring Format")
# ===----------------------------------------------------------------------=== #


def check_module_docstring(
    lines: list[str], header_end: int, report: FileReport
) -> int | None:
    """Returns the 0-indexed line number of the closing `\"\"\"`, or None if
    no module docstring was found at all (flagged separately).

    SciJo convention: the docstring opens with '\"\"\"Title (`scijo.a.b`)' on
    the *same* line as the opening quotes (unlike NuMojo/mojoBLAS, which put
    a bare '\"\"\"' then the title on the next line), followed by a bare
    '===...' underline matching the title's length.
    """
    idx = header_end
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx >= len(lines) or '"""' not in lines[idx]:
        report.add(
            header_end + 1, "docstring", "Missing module-level docstring."
        )
        return None

    doc_start = idx
    first_line = lines[idx].rstrip("\n")
    if first_line.strip() == '"""':
        report.add(
            idx + 1,
            "docstring",
            "Module docstring opens with a bare '\"\"\"' — SciJo convention "
            "puts the title on the same line: '\"\"\"Title (`scijo.a.b`)'.",
        )

    # Find closing line.
    close = None
    j = idx + 1
    while j < len(lines):
        if '"""' in lines[j]:
            close = j
            break
        j += 1
    if close is None:
        report.add(idx + 1, "docstring", "Module docstring is never closed.")
        return None

    if close == idx:
        report.add(idx + 1, "docstring", "Module docstring is empty.")
        return close

    title_text = first_line[3:] if first_line.startswith('"""') else first_line
    if not re.match(r"^[A-Za-z_].*\(`scijo(\.[a-z0-9_]+)*`\)$", title_text):
        report.add(
            idx + 1,
            "docstring",
            f"Title line should look like 'Title (`scijo.optimize.root_scalar`)': "
            f"{title_text!r}",
        )

    if close < idx + 2:
        report.add(
            idx + 2, "docstring", "Module docstring missing '===' underline."
        )
        return close

    # The underline's length matches the FULL first line, including the
    # leading '\"\"\"' (confirmed against the codebase) — not just the title
    # text after the quotes.
    underline_line = lines[idx + 1].rstrip("\n")
    if not re.match(r"^=+$", underline_line):
        report.add(
            idx + 2,
            "docstring",
            "Second line of module docstring should be a bare '===' underline.",
        )
    elif len(underline_line) != len(first_line):
        report.add(
            idx + 2,
            "docstring",
            f"Underline length ({len(underline_line)}) does not match the "
            f"title line's length ({len(first_line)}): {first_line!r}",
        )

    body = lines[idx : close + 1]
    body_text = "".join(body)
    if re.search(r"^\s*(TODO|FIXME):", body_text, re.M):
        report.add(
            idx + 1,
            "docstring",
            "TODO:/FIXME: item found inside module docstring — move it to a "
            "standalone comment after the closing '\"\"\"'.",
        )

    for label in ("Example:", "Note:", "Return:", "Parameter:", "Notes:"):
        if any(line.strip() == label for line in body):
            canon = SECTION_CANONICAL[label.rstrip(":")]
            report.add(
                idx + 1,
                "docstring",
                f"Module docstring uses '{label}' — canonical form is "
                f"'{canon}:'.",
            )

    return close


# ===----------------------------------------------------------------------=== #
# Check: section separators + TODO/FIXME
# ===----------------------------------------------------------------------=== #


def check_body(
    lines: list[str], body_start: int, report: FileReport
) -> None:
    doc_spans = set()
    for s, e in docstring_spans(lines[body_start:]):
        for i in range(s, e + 1):
            doc_spans.add(body_start + i)

    for i in range(body_start, len(lines)):
        if i in doc_spans:
            continue
        line = lines[i]
        stripped = line.rstrip("\n")

        if re.match(r"^\s*#\s*=", stripped) and "=" in stripped:
            if is_separator_line(line):
                indent = separator_indent(stripped)
                if stripped != indent + EXPECTED_SEPARATOR:
                    report.add(
                        i + 1,
                        "separator",
                        f"Section separator malformed (expected 80-char "
                        f"'# ===' + 70 dashes + '=== #', same indent as "
                        f"surrounding code): {stripped.strip()!r}",
                    )
            elif re.search(r"[-=]{4,}", stripped):
                # A "banner" comment that uses ===/--- decoration but isn't
                # a bare separator line (e.g. has a title embedded in the
                # same line) — not the standard 3-line
                # separator/title/separator block.
                report.add(
                    i + 1,
                    "separator",
                    f"Non-standard banner comment (should be a 3-line "
                    f"'# ===...=== #' / '# Title' / '# ===...=== #' block, "
                    f"not decoration on one line): {stripped!r}",
                )

        m = TODO_RE.search(stripped)
        if m and not stripped.lstrip().startswith(("# ===", "#===")):
            if not TODO_OK_RE.search(stripped):
                report.add(
                    i + 1,
                    "todo",
                    f"TODO/FIXME not formatted as '# TODO: Capitalized "
                    f"sentence.' : {stripped.strip()!r}",
                )


# ===----------------------------------------------------------------------=== #
# Check: imports (cross-project convention shared with NuMojo/mojoBLAS)
# ===----------------------------------------------------------------------=== #

# Every group of `from`/`import` statements must be preceded by a 3-line
# '# ===' / title / '# ===' banner (title one of IMPORT_GROUP_TITLES) —
# same convention as NuMojo and mojoBLAS. `organize_mojo_imports.py`
# rewrites files to match this; this check just reports drift.
IMPORT_GROUP_TITLES = {"Stdlib", "External", "Max", "SciJo"}
IMPORT_RE = re.compile(r"^(?P<indent>\s*)(?:from|import)\s+\S")


def check_imports(lines: list[str], body_start: int, report: FileReport) -> None:
    doc_spans = set()
    for s, e in docstring_spans(lines[body_start:]):
        for i in range(s, e + 1):
            doc_spans.add(body_start + i)

    def is_banner_at(idx: int) -> bool:
        return (
            idx >= 2
            and is_separator_line(lines[idx - 2])
            and lines[idx - 1].strip().lstrip("# ").rstrip()
            in IMPORT_GROUP_TITLES
            and is_separator_line(lines[idx])
        )

    i = body_start
    n = len(lines)
    prev_was_import = False
    while i < n:
        if i in doc_spans:
            prev_was_import = False
            i += 1
            continue
        if IMPORT_RE.match(lines[i]):
            # Only the FIRST line of a contiguous run of import statements
            # needs a banner directly above it — later imports in the same
            # run are preceded by another import line, not a banner.
            if not prev_was_import and not is_banner_at(i - 1):
                report.add(
                    i + 1,
                    "imports",
                    f"Import group not preceded by a '# ===' / title / "
                    f"'# ===' banner (title one of "
                    f"{sorted(IMPORT_GROUP_TITLES)}): {lines[i].strip()!r}",
                )
            # Skip to the end of this (possibly multi-line, parenthesized)
            # import statement so its continuation lines aren't re-checked.
            depth = lines[i].count("(") - lines[i].count(")")
            while depth > 0 and i + 1 < n:
                i += 1
                depth += lines[i].count("(") - lines[i].count(")")
            prev_was_import = True
            i += 1
            continue
        # Any other line — including a blank line — ends the current
        # contiguous import run; the next import line needs its own banner.
        prev_was_import = False
        i += 1


# ===----------------------------------------------------------------------=== #
# Check: error handling
# ===----------------------------------------------------------------------=== #

# SciJo raises plain `Error(...)` (no NumojoError-style wrapper type exists
# in this codebase) — nothing to enforce here beyond the docstring label
# check below.


def check_raises_docstring_labels(lines: list[str], report: FileReport) -> None:
    # Placeholder for future error-label conventions; SciJo currently has
    # no restricted set of Raises: labels to enforce.
    return


# ===----------------------------------------------------------------------=== #
# Check: function/method signature vs. docstring
# ===----------------------------------------------------------------------=== #


@dataclass
class Signature:
    name: str
    kind: str  # "def" or "fn"
    indent: int
    sig_start: int
    sig_end: int  # last line of the signature (the one ending in ':')
    params: list[str]
    args: list[str]
    raises: bool
    returns: bool
    is_private: bool
    is_dunder: bool
    self_raises: bool = False  # has an explicit `raise` statement in its own body


def parse_signature(lines: list[str], start: int) -> Signature | None:
    """Parse a def/fn signature starting at `start` (the line with the
    def/fn keyword, possibly after decorator lines already consumed by the
    caller). Returns None if this doesn't look like a real signature."""
    line0 = lines[start]
    m = re.match(r"^(?P<indent>\s*)(?P<kind>def|fn)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)", line0)
    if not m:
        return None
    indent = len(m.group("indent"))
    kind = m.group("kind")
    name = m.group("name")

    # Collect the full signature text until the line ending in ':' at
    # bracket-depth 0 (accounting for [] and ()).
    text_parts = [line0]
    depth = line0.count("[") + line0.count("(") - line0.count("]") - line0.count(")")
    end = start
    while True:
        stripped_end = text_parts[-1].rstrip("\n").rstrip()
        if depth <= 0 and stripped_end.endswith(":"):
            break
        end += 1
        if end >= len(lines):
            return None
        text_parts.append(lines[end])
        depth += (
            lines[end].count("[")
            + lines[end].count("(")
            - lines[end].count("]")
            - lines[end].count(")")
        )
        if end - start > 200:
            return None

    full = "".join(text_parts)

    # Extract the parameter list [...]  (generic/comptime params) — the
    # FIRST bracket group after the name.
    params: list[str] = []
    pm = re.search(r"\[", full)
    args_search_from = 0
    if pm and full[: pm.start()].rstrip().endswith(name):
        depth2 = 0
        j = pm.start()
        buf = []
        while j < len(full):
            c = full[j]
            if c == "[":
                depth2 += 1
                if depth2 == 1:
                    j += 1
                    continue
            if c == "]":
                depth2 -= 1
                if depth2 == 0:
                    j += 1
                    break
            buf.append(c)
            j += 1
        param_block = "".join(buf)
        params = split_top_level(param_block)
        args_search_from = j

    # Extract the argument list (...) — the paren group after params (or
    # after the name if no param brackets).
    args: list[str] = []
    am = re.search(r"\(", full[args_search_from:])
    if am:
        start_paren = args_search_from + am.start()
        depth3 = 0
        j = start_paren
        buf = []
        while j < len(full):
            c = full[j]
            if c == "(":
                depth3 += 1
                if depth3 == 1:
                    j += 1
                    continue
            if c == ")":
                depth3 -= 1
                if depth3 == 0:
                    j += 1
                    break
            buf.append(c)
            j += 1
        arg_block = "".join(buf)
        args = split_top_level(arg_block)

    tail = full[args_search_from:]
    # Everything after the closing paren of args, up to the final ':'.
    close_paren_idx = None
    depth4 = 0
    started = False
    for idx, c in enumerate(tail):
        if c == "(":
            depth4 += 1
            started = True
        elif c == ")":
            depth4 -= 1
            if started and depth4 == 0:
                close_paren_idx = idx
                break
    rest = tail[close_paren_idx + 1 :] if close_paren_idx is not None else ""

    raises = bool(re.search(r"\braises\b", rest))
    # Returns something other than None: a `->` present.
    returns = "->" in rest and not re.search(r"->\s*None\b", rest)

    is_dunder = name.startswith("__") and name.endswith("__")
    is_private = name.startswith("_") and not is_dunder

    return Signature(
        name=name,
        kind=kind,
        indent=indent,
        sig_start=start,
        sig_end=end,
        params=[p for p in params if p and not p.startswith("*")],
        args=[a for a in args if a],
        raises=raises,
        returns=returns,
        is_private=is_private,
        is_dunder=is_dunder,
        self_raises=body_has_explicit_raise(lines, end, indent),
    )


def body_has_explicit_raise(lines: list[str], sig_end: int, sig_indent: int) -> bool:
    """True if this function's own body (before the next sibling def/fn/
    struct at the same or shallower indent) contains an explicit `raise`
    statement — as opposed to merely being `raises`-annotated because it
    calls something else that can raise."""
    i = sig_end + 1
    while i < len(lines):
        line = lines[i]
        if line.strip() == "":
            i += 1
            continue
        this_indent = len(line) - len(line.lstrip())
        if this_indent <= sig_indent and re.match(
            r"^\s*(def|fn|struct|trait|@)\s*", line
        ):
            break
        if re.search(r"(?<![\w.])raise\b", line) and not line.lstrip().startswith("#"):
            return True
        i += 1
    return False


def split_top_level(block: str) -> list[str]:
    """Split a `[a, b: T, c: U = v]` or `(a, b: T)` block on top-level
    commas, stripping the outer brackets, and return just the bound names
    (before ':' or '=')."""
    if len(block) >= 2 and block[0] in "[(":
        block = block[1:-1]
    parts = []
    depth = 0
    cur = []
    for c in block:
        if c in "[(":
            depth += 1
        elif c in "])":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(c)
    if cur:
        parts.append("".join(cur))

    names = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Strip leading modifiers.
        p = re.sub(r"^(var|mut|ref|out|owned|read|\*\*|\*)\s*", "", p).strip()
        if not p or p in ("self", "*"):
            continue
        if p in ("self",):
            continue
        # name is up to the first ':' or '=' or whitespace.
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)", p)
        if m:
            nm = m.group(1)
            if nm == "self":
                continue
            names.append(nm)
    return names


def extract_docstring_for(lines: list[str], sig_end: int) -> tuple[int, int] | None:
    """Given the line index of the ':' closing a signature, find the
    function's docstring span (start, end) if the next non-blank line opens
    one. Returns None if there's no docstring immediately following."""
    i = sig_end + 1
    if i >= len(lines):
        return None
    if '"""' not in lines[i]:
        return None
    start = i
    if lines[i].count('"""') >= 2:
        return (start, i)
    j = i + 1
    while j < len(lines):
        if '"""' in lines[j]:
            return (start, j)
        j += 1
    return None


def is_stub_body(lines: list[str], sig_end: int) -> bool:
    """True if the line right after the signature is a bare '...' stub
    (trait method declaration with no implementation)."""
    i = sig_end + 1
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    return i < len(lines) and lines[i].strip() == "..."


def check_signature_docstring(
    sig: Signature, lines: list[str], report: FileReport
) -> None:
    if is_stub_body(lines, sig.sig_end):
        return  # trait method declaration, not a real implementation

    doc_span = extract_docstring_for(lines, sig.sig_end)
    if doc_span is None:
        if not sig.is_private:
            report.add(
                sig.sig_start + 1,
                "docstring",
                f"'{sig.kind} {sig.name}' has no docstring.",
            )
        return

    start, end = doc_span
    body = lines[start : end + 1]

    # Which canonical sections are present, and are they labeled with a
    # non-canonical (singular/alternate) spelling?
    present_canonical: set[str] = set()
    present_order: list[str] = []
    for line in body:
        m = SECTION_HEADER_RE.match(line.rstrip("\n"))
        if not m:
            continue
        raw = m.group("name")
        if raw not in SECTION_CANONICAL:
            continue
        canon = SECTION_CANONICAL[raw]
        present_canonical.add(canon)
        present_order.append(canon)
        if raw != canon:
            report.add(
                start + 1,
                "docstring",
                f"'{sig.name}': docstring section '{raw}:' should be "
                f"'{canon}:' (canonical form).",
            )

    # Order check.
    seen_order = [SECTION_ORDER.index(c) for c in present_order if c in SECTION_ORDER]
    if seen_order != sorted(seen_order):
        report.add(
            start + 1,
            "docstring",
            f"'{sig.name}': docstring sections out of order "
            f"({' -> '.join(present_order)}); expected order is "
            f"{' -> '.join(SECTION_ORDER)}.",
        )

    if sig.is_private:
        return  # one-line docstring is sufficient for private helpers

    # Documented parameter/arg names.
    documented_params = extract_documented_names(body, "Parameters")
    documented_args = extract_documented_names(body, "Args")

    missing_params = [p for p in sig.params if p not in documented_params]
    if missing_params and "Parameters" not in present_canonical and sig.params:
        report.add(
            start + 1,
            "docstring",
            f"'{sig.name}': has type parameter(s) {sig.params} but "
            f"docstring has no 'Parameters:' section.",
        )
    elif missing_params:
        report.add(
            start + 1,
            "docstring",
            f"'{sig.name}': type parameter(s) not documented in "
            f"Parameters: {missing_params}",
        )

    missing_args = [a for a in sig.args if a not in documented_args]
    if missing_args and "Args" not in present_canonical and sig.args:
        report.add(
            start + 1,
            "docstring",
            f"'{sig.name}': has argument(s) {sig.args} but docstring has "
            f"no 'Args:' section.",
        )
    elif missing_args:
        report.add(
            start + 1,
            "docstring",
            f"'{sig.name}': argument(s) not documented in Args: {missing_args}",
        )

    if sig.returns and "Returns" not in present_canonical:
        report.add(
            start + 1,
            "docstring",
            f"'{sig.name}': has a return type but docstring has no "
            f"'Returns:' section.",
        )

    if sig.self_raises and "Raises" not in present_canonical:
        # Only flagged when the function's own body has an explicit `raise`
        # statement (not merely `raises`-annotated because it calls other
        # raising functions) — don't add speculatively.
        report.add(
            start + 1,
            "docstring",
            f"'{sig.name}': body has an explicit 'raise' but docstring has "
            f"no 'Raises:' section.",
        )


def extract_documented_names(body: list[str], section: str) -> set[str]:
    names = set()
    in_section = False
    section_indent = None
    for line in body:
        stripped = line.rstrip("\n")
        m = SECTION_HEADER_RE.match(stripped)
        if m and SECTION_CANONICAL.get(m.group("name")) == section:
            in_section = True
            section_indent = len(m.group("indent"))
            continue
        if in_section:
            if stripped.strip() == "":
                continue
            this_indent = len(stripped) - len(stripped.lstrip())
            if section_indent is not None and this_indent <= section_indent:
                if SECTION_HEADER_RE.match(stripped):
                    in_section = False
                    continue
                if this_indent < section_indent:
                    in_section = False
                    continue
            nm = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", stripped)
            if nm:
                names.add(nm.group(1))
    return names


# ===----------------------------------------------------------------------=== #
# Driver: walk a file, find every def/fn, run the checks
# ===----------------------------------------------------------------------=== #


def find_top_level_defs(lines: list[str]) -> list[int]:
    """Line indices where a module-level or struct/trait-member def/fn
    signature begins. Excludes def/fn nested inside another function's body
    (local helper closures like a `@parameter def kernel[...]` defined
    inside a method) — those are implementation-detail plumbing, not public
    API, and don't need standalone docstrings."""
    starts = []
    i = 0
    n = len(lines)
    doc_spans = set()
    for s, e in docstring_spans(lines):
        for k in range(s, e + 1):
            doc_spans.add(k)

    # Track a stack of (indent, kind) for enclosing def/fn/struct/trait
    # blocks, based on indentation, to detect nesting inside a function body.
    block_stack: list[tuple[int, str]] = []
    while i < n:
        if i in doc_spans:
            i += 1
            continue
        line = lines[i]
        if line.strip() == "":
            i += 1
            continue
        indent = len(line) - len(line.lstrip())
        while block_stack and indent <= block_stack[-1][0]:
            block_stack.pop()

        m = re.match(r"^\s*(def|fn)\s+[A-Za-z_]", line)
        if m:
            nested_in_def = any(kind in ("def", "fn") for _, kind in block_stack)
            if not nested_in_def:
                starts.append(i)
            block_stack.append((indent, m.group(1)))
            # Skip over the rest of a (possibly multi-line) signature so its
            # continuation lines — which can dedent back to the def's own
            # indent on the closing ')...:' line — don't get misread as a
            # sibling/dedent that pops this def off the stack early.
            depth = (
                line.count("[")
                + line.count("(")
                - line.count("]")
                - line.count(")")
            )
            j = i
            while depth > 0 and j + 1 < n:
                j += 1
                depth += (
                    lines[j].count("[")
                    + lines[j].count("(")
                    - lines[j].count("]")
                    - lines[j].count(")")
                )
            i = j
        elif re.match(r"^\s*(struct|trait)\s+[A-Za-z_]", line):
            block_stack.append((indent, "struct"))
        i += 1
    return starts


def check_file(path: Path, only: set[str] | None) -> FileReport:
    report = FileReport(path=path)
    text = path.read_text()
    lines = text.splitlines(keepends=True)

    def enabled(cat: str) -> bool:
        return only is None or cat in only

    if enabled("headers") and path.name not in HEADER_EXEMPT_FILES:
        header_end = check_header(lines, report)
    else:
        # Best-effort: find end of header block without reporting.
        header_end = 0
        while header_end < len(lines) and not (
            header_end > 0 and is_separator_line(lines[header_end])
        ):
            header_end += 1
        header_end += 1

    doc_end = None
    if enabled("docstring"):
        doc_end = check_module_docstring(lines, header_end, report)
    if doc_end is None:
        # locate it anyway for downstream checks
        idx = header_end
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1
        if idx < len(lines) and '"""' in lines[idx]:
            j = idx + 1
            if lines[idx].count('"""') < 2:
                while j < len(lines) and '"""' not in lines[j]:
                    j += 1
            doc_end = j
        else:
            doc_end = header_end

    body_start = doc_end + 1 if doc_end is not None else header_end

    if enabled("separators") or enabled("todo"):
        check_body(lines, body_start, report)
    if enabled("imports"):
        check_imports(lines, body_start, report)
    if enabled("error-handling"):
        check_raises_docstring_labels(lines, report)

    if enabled("signatures"):
        for start in find_top_level_defs(lines):
            sig = parse_signature(lines, start)
            if sig is None:
                continue
            check_signature_docstring(sig, lines, report)

    return report


# ===----------------------------------------------------------------------=== #
# CLI
# ===----------------------------------------------------------------------=== #

ALL_CATEGORIES = [
    "headers",
    "docstring",
    "separators",
    "imports",
    "todo",
    "error-handling",
    "signatures",
]


def iter_mojo_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.mojo")))
        elif p.suffix == ".mojo":
            files.append(p)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=f"Comma-separated subset of checks to run: {', '.join(ALL_CATEGORIES)}",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Print only the summary line."
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Write full results as JSON."
    )
    args = parser.parse_args()

    only = set(args.only.split(",")) if args.only else None
    if only:
        unknown = only - set(ALL_CATEGORIES)
        if unknown:
            print(f"Unknown --only categories: {unknown}", file=sys.stderr)
            return 2

    files = iter_mojo_files(args.paths)
    reports: list[FileReport] = []
    total_findings = 0
    for f in files:
        try:
            report = check_file(f, only)
        except Exception as exc:  # keep going across a bad file
            report = FileReport(path=f)
            report.add(1, "internal", f"Checker error on this file: {exc!r}")
        if report.findings:
            reports.append(report)
            total_findings += len(report.findings)

    if args.json:
        payload = [
            {
                "file": str(r.path),
                "findings": [
                    {"line": f.line, "category": f.category, "message": f.message}
                    for f in r.findings
                ],
            }
            for r in reports
        ]
        args.json.write_text(json.dumps(payload, indent=2))

    if not args.quiet:
        for r in reports:
            print(f"\n{r.path}  ({len(r.findings)} finding(s))")
            for f in r.findings:
                print(f"  {r.path}:{f.line}: [{f.category}] {f.message}")

    print(
        f"\n{len(files)} file(s) checked, {len(reports)} file(s) with "
        f"findings, {total_findings} total finding(s)."
    )
    return 1 if total_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
