"""Unit tests for the tool-call DSL parser.

Run: `cd server && .venv/bin/python -m pytest test_gemma_tool_call_parser.py -v`
or just `.venv/bin/python test_gemma_tool_call_parser.py` for a simple runner.

Cases include the dicemother python-exec body that broke the regex chain
(nested `{'C': (10, 10, 5)}` dict literal inside an atomic-quoted string
containing bareword-shaped keys after commas), plus the canonical happy
paths and the malformed-input fallbacks.
"""
from __future__ import annotations
import sys
from gemma_tool_call_parser import parse_tool_call_body, ToolCallParseError


def t(name, body, expected_fn=None, expected_args=None, *, should_fail=False):
    """Run a single test. Print result; return True on pass."""
    try:
        fn, args = parse_tool_call_body(body)
        if should_fail:
            print(f"  FAIL  [{name}] expected parse error, got "
                  f"({fn!r}, {args!r})")
            return False
        if fn != expected_fn:
            print(f"  FAIL  [{name}] fn: expected {expected_fn!r} got {fn!r}")
            return False
        if args != expected_args:
            print(f"  FAIL  [{name}] args:\n    expected {expected_args!r}\n"
                  f"    got      {args!r}")
            return False
        print(f"  PASS  [{name}]")
        return True
    except ToolCallParseError as e:
        if should_fail:
            print(f"  PASS  [{name}] (raised as expected: {e})")
            return True
        print(f"  FAIL  [{name}] raised {e}")
        return False


def main():
    results = []

    # ── canonical happy paths ────────────────────────────────────────
    results.append(t(
        "empty args",
        'call:foo{}',
        "foo", {}))
    results.append(t(
        "single string arg",
        'call:render-visual__render{description:<|"|>a duck<|"|>}',
        "render-visual__render", {"description": "a duck"}))
    results.append(t(
        "two string args",
        'call:greet{name:<|"|>alice<|"|>,greeting:<|"|>hi<|"|>}',
        "greet", {"name": "alice", "greeting": "hi"}))
    results.append(t(
        "boolean + int + float",
        'call:settings{flag:true,n:42,pi:3.14}',
        "settings", {"flag": True, "n": 42, "pi": 3.14}))
    results.append(t(
        "nested map",
        'call:nested{outer:{inner:<|"|>x<|"|>,k:1}}',
        "nested", {"outer": {"inner": "x", "k": 1}}))
    results.append(t(
        "list of strings",
        'call:random-choice__uniform{items:[<|"|>a<|"|>,<|"|>b<|"|>],n:2}',
        "random-choice__uniform", {"items": ["a", "b"], "n": 2}))
    results.append(t(
        "list of mixed scalars",
        'call:f{xs:[1,2.5,true,false]}',
        "f", {"xs": [1, 2.5, True, False]}))
    results.append(t(
        "null value",
        'call:f{x:null}',
        "f", {"x": None}))
    results.append(t(
        "atomic-quoted key",
        'call:f{<|"|>weird-key<|"|>:1}',
        "f", {"weird-key": 1}))
    results.append(t(
        "empty list",
        'call:f{xs:[]}',
        "f", {"xs": []}))

    # ── cases the regex chain broke on (the actual reported bug) ─────
    results.append(t(
        "python script with nested dict literal — dicemother repro",
        'call:python-exec__run{task:<|"|>import random\n'
        "# Define budgets for each tier\n"
        "# Tier: (Attribute, Equipment, Skill)\n"
        "budgets = {\n"
        "    'C': (10, 10, 5),\n"
        "    'U': (25, 20, 15),\n"
        "    'R': (50, 40, 30),\n"
        "    'SR': (100, 80, 60),\n"
        "    'SSR': (250, 200, 150)\n"
        "}\n"
        "\n"
        "tiers = list(budgets.keys())\n"
        "chosen_tier = random.choice(tiers)\n"
        'print(chosen_tier)<|"|>}',
        "python-exec__run", {
            "task": "import random\n"
                    "# Define budgets for each tier\n"
                    "# Tier: (Attribute, Equipment, Skill)\n"
                    "budgets = {\n"
                    "    'C': (10, 10, 5),\n"
                    "    'U': (25, 20, 15),\n"
                    "    'R': (50, 40, 30),\n"
                    "    'SR': (100, 80, 60),\n"
                    "    'SSR': (250, 200, 150)\n"
                    "}\n"
                    "\n"
                    "tiers = list(budgets.keys())\n"
                    "chosen_tier = random.choice(tiers)\n"
                    "print(chosen_tier)",
        }))
    results.append(t(
        "json-shaped string content with bareword-keys-after-commas — would've tripped the bareword regex",
        'call:f{payload:<|"|>{"k1":1,k2:2,k3:3}<|"|>}',
        "f", {"payload": '{"k1":1,k2:2,k3:3}'}))
    results.append(t(
        "negative number",
        'call:f{n:-42,m:-3.14e2}',
        "f", {"n": -42, "m": -314.0}))

    # ── raw-string form (`<|D...D|>` with non-quote delim) ───────────
    results.append(t(
        "raw-string with backtick delim",
        'call:f{x:<|`<|"|>literal quote inside`|>}',
        "f", {"x": '<|"|>literal quote inside'}))

    # ── malformed input → should raise ───────────────────────────────
    results.append(t(
        "missing close brace",
        'call:f{x:1', None, None, should_fail=True))
    results.append(t(
        "unclosed atomic string",
        'call:f{x:<|"|>not closed}', None, None, should_fail=True))
    results.append(t(
        "missing call: prefix",
        'foo{x:1}', None, None, should_fail=True))
    results.append(t(
        "trailing junk",
        'call:f{x:1}garbage', None, None, should_fail=True))

    # ── whitespace tolerance ─────────────────────────────────────────
    results.append(t(
        "whitespace around tokens",
        'call:f { x : 1 , y : <|"|>hi<|"|> }',
        "f", {"x": 1, "y": "hi"}))

    failed = sum(1 for r in results if not r)
    total = len(results)
    print()
    print(f"=== {total - failed}/{total} tests passed ===")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
