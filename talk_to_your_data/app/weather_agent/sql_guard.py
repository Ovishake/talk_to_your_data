"""Defense-in-depth check that LLM-generated SQL is a single read-only query.

The real security boundary is Unity Catalog: the app's service principal only has
SELECT on the curated schema. This guard gives the LLM a clear error it can fix.
"""

from __future__ import annotations

import re

_FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "merge", "drop", "alter", "create", "truncate",
    "grant", "revoke", "deny", "copy", "optimize", "vacuum", "refresh", "msck",
    "call", "set", "use", "reset", "cache", "uncache", "install", "restore",
    "clone", "analyze", "export", "import", "load", "unload", "execute",
)  # fmt: skip
_FORBIDDEN_FUNCTIONS = (
    r"ai_\w+", r"http_request", r"read_\w+", r"secret", r"try_secret",
    r"java_method", r"reflect", r"vector_search", r"remote_query",
)  # fmt: skip

_KEYWORD_RE = re.compile(r"\b(" + "|".join(_FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE)
_FUNCTION_RE = re.compile(r"\b(" + "|".join(_FORBIDDEN_FUNCTIONS) + r")\s*\(", re.IGNORECASE)
_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRING_OR_IDENT = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|`[^`]*`")


class UnsafeSQLError(ValueError):
    """Raised when a statement is not a single read-only query."""


def validate_sql(sql: str) -> str:
    """Return the cleaned statement, or raise UnsafeSQLError with a reason the LLM can act on."""
    if not sql or not sql.strip():
        raise UnsafeSQLError("Empty query.")

    cleaned = _LINE_COMMENT.sub(" ", _BLOCK_COMMENT.sub(" ", sql)).strip().rstrip(";").strip()
    masked = _STRING_OR_IDENT.sub("''", cleaned)  # literals can't trigger or hide keywords

    if ";" in masked:
        raise UnsafeSQLError("Only one statement is allowed. Remove extra ';' statements.")
    first_word = re.match(r"[\s(]*([A-Za-z]+)", masked)
    if not first_word or first_word.group(1).lower() not in {"select", "with"}:
        raise UnsafeSQLError("Only SELECT or WITH queries are allowed.")
    if keyword := _KEYWORD_RE.search(masked):
        raise UnsafeSQLError(f"Keyword '{keyword.group(1).upper()}' is not allowed (read-only).")
    if function := _FUNCTION_RE.search(masked):
        raise UnsafeSQLError(f"Function '{function.group(1)}' is not allowed in this app.")
    return cleaned
