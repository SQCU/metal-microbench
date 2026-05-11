"""Recursive-descent parser for the gemma chat-template tool-call body
DSL. Replaces a three-pass regex chain (atomic-quote → raw-quote →
bareword-key) + json.loads that the bridge used to do; that approach is
structurally inadequate because the DSL has nested braces, atomic-quoted
strings whose contents can contain arbitrary other syntax, and lists —
features no regex chain can disambiguate.

Grammar (matches chat_template.jinja's format_argument macro exactly):

    call          := "call:" name "{" pairs "}"
    pairs         := pair ("," pair)*   | ε
    pair          := key ":" value
    key           := atomic_string | bareword_token
    value         := atomic_string
                   | raw_string
                   | map
                   | list
                   | scalar
    map           := "{" pairs "}"
    list          := "[" list_items "]"
    list_items    := value ("," value)*   | ε
    atomic_string := "<|\"|>" content "<|\"|>"      (lazy match on close)
    raw_string    := "<|" D content D "|>"          (D ∉ {'<','>','|'})
    scalar        := bareword_token                  (parsed as int / float
                                                      / true / false / null
                                                      / string in that order)
    bareword_token := chars up to whitespace or any of ':,{}[]'

Whitespace is allowed between tokens. Inside string literals nothing is
escaped — the format itself has no escape mechanism; an atomic-quoted
string that contains a literal `<|"|>` substring is unparseable in
general (the close-delimiter for the OUTER string is ambiguous with
the close-delimiter for the inner content). We resolve such cases by
taking the FIRST `<|"|>` close after the open, matching the Jinja-side
non-greedy emit pattern. Pathologically nested atomic-quotes in real
model output have not been observed; if they appear they should be
treated as a training-data / template-emit bug, not parser ambiguity.
"""
from __future__ import annotations


class ToolCallParseError(Exception):
    """Raised when the body cannot be parsed. The bridge falls back to
    leaving the raw `<|tool_call>...<tool_call|>` block as visible
    content so the failure is observable rather than silent."""


_ATOMIC_OPEN = '<|"|>'
_ATOMIC_CLOSE = '<|"|>'
_BAREWORD_STOPS = set(' \t\n\r:,{}[]')


class _Parser:
    __slots__ = ('s', 'i')

    def __init__(self, src: str) -> None:
        self.s = src
        self.i = 0

    # ── primitives ───────────────────────────────────────────────────
    def _eof(self) -> bool:
        return self.i >= len(self.s)

    def _peek(self, n: int = 1) -> str:
        return self.s[self.i:self.i + n]

    def _consume(self, lit: str) -> bool:
        if self.s.startswith(lit, self.i):
            self.i += len(lit)
            return True
        return False

    def _expect(self, lit: str) -> None:
        if not self._consume(lit):
            ctx = self.s[self.i:self.i + 24]
            raise ToolCallParseError(
                f"expected {lit!r} at pos {self.i}; saw {ctx!r}")

    def _skip_ws(self) -> None:
        while self.i < len(self.s) and self.s[self.i] in ' \t\n\r':
            self.i += 1

    # ── nonterminals ─────────────────────────────────────────────────
    def parse_call(self) -> tuple[str, dict]:
        self._skip_ws()
        self._expect("call:")
        name = self._parse_name()
        self._skip_ws()
        args = self._parse_map()
        self._skip_ws()
        return name, args

    def _parse_name(self) -> str:
        start = self.i
        while self.i < len(self.s) and self.s[self.i] not in ' \t\n\r{':
            self.i += 1
        if start == self.i:
            raise ToolCallParseError(f"empty function name at pos {start}")
        return self.s[start:self.i]

    def _parse_map(self) -> dict:
        self._expect("{")
        out: dict = {}
        self._skip_ws()
        if self._peek() == "}":
            self.i += 1
            return out
        while True:
            self._skip_ws()
            key = self._parse_key()
            self._skip_ws()
            self._expect(":")
            self._skip_ws()
            value = self._parse_value()
            out[key] = value
            self._skip_ws()
            if self._consume(","):
                continue
            self._expect("}")
            return out

    def _parse_key(self):
        if self.s.startswith(_ATOMIC_OPEN, self.i):
            return self._parse_atomic_string()
        return self._parse_bareword_token()

    def _parse_value(self):
        if self.s.startswith(_ATOMIC_OPEN, self.i):
            return self._parse_atomic_string()
        # raw-string: `<|D...D|>` where D is not '<', '>', '|'. Must come
        # before the bareword fallback because a leading `<|` would also
        # be a valid bareword start otherwise.
        if (self.s.startswith("<|", self.i)
                and not self.s.startswith(_ATOMIC_OPEN, self.i)
                and self.i + 2 < len(self.s)
                and self.s[self.i + 2] not in '<>|'):
            return self._parse_raw_string()
        c = self._peek()
        if c == "{":
            return self._parse_map()
        if c == "[":
            return self._parse_list()
        return self._parse_scalar()

    def _parse_atomic_string(self) -> str:
        self._expect(_ATOMIC_OPEN)
        start = self.i
        end = self.s.find(_ATOMIC_CLOSE, self.i)
        if end < 0:
            raise ToolCallParseError(
                f"unclosed {_ATOMIC_OPEN}...{_ATOMIC_CLOSE} at pos {start}")
        content = self.s[start:end]
        self.i = end + len(_ATOMIC_CLOSE)
        return content

    def _parse_raw_string(self) -> str:
        self._expect("<|")
        if self._eof():
            raise ToolCallParseError("unterminated <|...|> at eof")
        delim = self.s[self.i]
        if delim in '<>|':
            raise ToolCallParseError(
                f"invalid raw-string delim {delim!r} at pos {self.i}")
        self.i += 1
        close = f"{delim}|>"
        end = self.s.find(close, self.i)
        if end < 0:
            raise ToolCallParseError(
                f"unclosed <|{delim}...{delim}|> at pos {self.i}")
        content = self.s[self.i:end]
        self.i = end + len(close)
        return content

    def _parse_list(self) -> list:
        self._expect("[")
        out: list = []
        self._skip_ws()
        if self._peek() == "]":
            self.i += 1
            return out
        while True:
            self._skip_ws()
            out.append(self._parse_value())
            self._skip_ws()
            if self._consume(","):
                continue
            self._expect("]")
            return out

    def _parse_scalar(self):
        tok = self._parse_bareword_token()
        if tok == "true":
            return True
        if tok == "false":
            return False
        if tok == "null" or tok == "None":
            return None
        # int → float → string, in that order. We don't attempt to coerce
        # strings that happen to parse as numbers if they came from a
        # different syntactic context — the bareword-token rule already
        # delimits scalars unambiguously.
        if tok:
            try:
                if "." not in tok and "e" not in tok and "E" not in tok:
                    return int(tok)
            except ValueError:
                pass
            try:
                return float(tok)
            except ValueError:
                pass
        return tok

    def _parse_bareword_token(self) -> str:
        start = self.i
        while self.i < len(self.s) and self.s[self.i] not in _BAREWORD_STOPS:
            self.i += 1
        if start == self.i:
            raise ToolCallParseError(f"empty bareword at pos {start}")
        return self.s[start:self.i]


def parse_tool_call_body(body: str) -> tuple[str, dict]:
    """Parse a `call:NAME{pairs}` body. Returns (function_name, args_dict).

    Args:
        body: the substring between `<|tool_call>` and `<tool_call|>`, with
            leading/trailing whitespace already stripped by the caller (or
            left as-is — the parser skips whitespace itself).
    Raises:
        ToolCallParseError on malformed input. Callers that need a None
        fallback (the bridge does, for graceful degradation) should catch.
    """
    p = _Parser(body)
    name, args = p.parse_call()
    if p.i != len(p.s):
        # Trailing junk after the closing `}`. Allow some whitespace then
        # error — trailing non-whitespace indicates a malformed body.
        tail = p.s[p.i:].strip()
        if tail:
            raise ToolCallParseError(
                f"unexpected trailing content after call: {tail[:48]!r}")
    return name, args
