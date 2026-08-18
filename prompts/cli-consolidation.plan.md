# CLI Consolidation Plan

## Goal

Reduce 8 entry points to **1**, replacing all existing entry points with a
single `magnetrun` dispatcher.  `fetch` and `config` become subcommands rather
than separate executables — they are in the same package, so there is no reason
to keep them separate.  Deprecated aliases are kept for one release cycle.

## Target entry points

```toml
[project.scripts]
magnetrun = "python_magnetrun.main:main"   # single entry point

# deprecated aliases — keep for one release cycle, then remove
python-magnetrun               = "python_magnetrun.cli:main"
srvdata-to-magnetrun           = "python_magnetrun.requests.cli:main"
magnetrun-fetch                = "python_magnetrun.requests.cli:main"
magnetrun-analysis             = "python_magnetrun.analysis.cli:main"
magnetrun-processing           = "python_magnetrun.processing.cli:main"
hybrid-magnetrun               = "python_magnetrun.hybrid.cli:main"
magnetrun-config               = "python_magnetrun.config_cli:main"
magnetrun-pigbrother-logparser = "python_magnetrun.tdms.log_parser:main"
```

## `magnetrun` subcommands

All subcommands follow the pattern `magnetrun <subcommand> [args] [options]`.
The subcommand comes **first** — files are positional arguments of each
subcommand parser, not the top-level parser.

```
magnetrun info       files... [base-opts]
magnetrun add        files... [base-opts] --formula ...
magnetrun plot       files... [base-opts] --vs_time ... --key_vs_key ...
magnetrun select     files... [base-opts] --output_key ... --output_timerange ...
magnetrun stats      files... [base-opts] --plateau --localmax --detect_bkpts --keys ...
magnetrun signature  files... [base-opts] --key Field --threshold 1e-3   # new
magnetrun analysis   files... [analysis-opts]
magnetrun processing file    filter|smooth|lag [processing-opts]
magnetrun hybrid     [hybrid-opts]
magnetrun logparser  [log-opts]
magnetrun compare    [compare-opts]                # new; see cross-domain-comparison.prompt.md Phase F
magnetrun fetch      [fetch-opts]                  # was srvdata-to-magnetrun / magnetrun-fetch
magnetrun config     <domain> [config-opts]        # was magnetrun-config (already has sub-subparsers for domain)
```

## Key structural gain: remove `_normalize_argv`

The current `python-magnetrun` places `input_file` (nargs="+") at the top-level
parser, then the subcommand after the files:

```
python-magnetrun file1 file2 --housing M9 plot --vs_time ...
```

This requires the `_normalize_argv` hack in `cli.py` to work around argparse.
Moving `input_file` into each subcommand parser puts the subcommand first and
eliminates the hack entirely:

```
magnetrun plot file1 file2 --housing M9 --vs_time ...
```

## `register()` pattern

Each module exposes a `register(subparsers)` function alongside its `main()`.
This is the same pattern already used by `magnetrun-config` / `config_cli.py`.

```python
# Example pattern
def register(sub: "argparse._SubParsersAction") -> None:
    p = sub.add_parser("plot", parents=[base_parser, plot_parser],
                       help="plot run data")
    p.add_argument("input_file", nargs="+", ...)
    p.set_defaults(_handler=_run)

def _run(args: argparse.Namespace) -> int:
    ...
    return 0

def main() -> None:          # kept for backward-compat alias
    import argparse, sys
    sub_mock = argparse.ArgumentParser()._subparsers._group_actions  # not used
    # or just build a standalone parser reusing _run
```

## New `signature` subcommand

Promote the logic in `tests/test-signature.py` (`__main__` block) to a proper
module at `python_magnetrun/commands/signature.py`.

Key API: `Signature.from_mdata(mdata, key, tkey, threshold)` (already exists in
`python_magnetrun/signature.py`).

Arguments:
- `input_file` (positional, nargs="+")
- Base args inherited from `create_base_parser()` (housing, site, insert, log-level, datadirs)
- `--key` (default: "Field")
- `--threshold` (default: 1e-3)
- `--window` (default: 10)
- `--save` (store_true — save regime CSV)

Output: `<basename>-<key>.csv` with columns `[time, key]` at regime transitions.

## New unified dispatcher

New file: `python_magnetrun/main.py`

```python
"""Unified `magnetrun` entry point."""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="magnetrun",
        description="View, analyse, and process magnet run data.",
    )
    sub = parser.add_subparsers(dest="command", required=True,
                                metavar="subcommand")

    # data commands (operate on run files)
    from .commands.info      import register as _r_info;      _r_info(sub)
    from .commands.add       import register as _r_add;       _r_add(sub)
    from .commands.plot      import register as _r_plot;      _r_plot(sub)
    from .commands.select    import register as _r_select;    _r_select(sub)
    from .commands.stats     import register as _r_stats;     _r_stats(sub)
    from .commands.signature import register as _r_sig;       _r_sig(sub)

    # workflow commands
    from .analysis.cli    import register as _r_ana;    _r_ana(sub)
    from .processing.cli  import register as _r_proc;   _r_proc(sub)
    from .hybrid.cli      import register as _r_hyb;    _r_hyb(sub)
    from .tdms.log_parser import register as _r_log;    _r_log(sub)
    from .comparison.cli  import register as _r_cmp;    _r_cmp(sub)

    # infrastructure commands (were separate executables)
    from .requests.cli import register as _r_fetch;  _r_fetch(sub)
    from .config_cli   import register as _r_cfg;    _r_cfg(sub)

    args = parser.parse_args()
    sys.exit(args._handler(args))


if __name__ == "__main__":
    main()
```

### `magnetrun fetch` — adding `register()` to `requests/cli.py`

`requests/cli.py` has a flat `main()` with a single `ArgumentParser`.  Adding
`register()` is straightforward:

```python
def register(sub: "argparse._SubParsersAction") -> None:
    p = sub.add_parser("fetch", help="fetch run data from server")
    # move all add_argument calls from main() to here
    p.set_defaults(_handler=_run)

def _run(args: argparse.Namespace) -> int:
    ...   # body of current main()
    return 0

def main() -> None:  # deprecated alias
    import argparse, sys
    p = argparse.ArgumentParser()
    # rebuild standalone parser (or call register on a temporary subparsers)
    sys.exit(_run(p.parse_args()))
```

### `magnetrun config` — adding `register()` to `config_cli.py`

`config_cli.py` already has internal sub-subparsers for `domain`.  Wrap them:

```python
def register(sub: "argparse._SubParsersAction") -> None:
    p = sub.add_parser("config", help="manage housing and insert configuration")
    domain_sub = p.add_subparsers(dest="domain", required=True)
    _register_domain_subparsers(domain_sub)   # extract current body of main()
    p.set_defaults(_handler=_run)

def main() -> None:  # deprecated alias — kept for one release cycle
    import argparse, sys
    parser = argparse.ArgumentParser(prog="magnetrun-config", ...)
    domain_sub = parser.add_subparsers(dest="domain", required=True)
    _register_domain_subparsers(domain_sub)
    args = parser.parse_args()
    sys.exit(_run(args))
```

## Implementation order

1. **`commands/signature.py`** — self-contained, good template for the `register()` pattern
2. **`commands/{info,add,plot,select,stats}.py`** — add `register()` to existing modules;
   move `input_file` positional from top-level to each subcommand parser
3. **`analysis/cli.py`**, **`processing/cli.py`**, **`hybrid/cli.py`**,
   **`tdms/log_parser.py`** — add `register()` to each
4. **`requests/cli.py`** — extract `_run()` + add `register()`; keep `main()` as deprecated shim
5. **`config_cli.py`** — extract `_register_domain_subparsers()` + add `register()`; keep `main()` as deprecated shim
6. **`python_magnetrun/main.py`** — write the unified dispatcher (imports all `register()` fns)
7. **`pyproject.toml`** — replace all scripts with `magnetrun` only; keep old names as deprecated aliases
8. **`cli.py`** — remove `_normalize_argv`; reduce to a deprecated shim or delete

## Files touched

| File | Action |
|---|---|
| `python_magnetrun/main.py` | Create |
| `python_magnetrun/commands/signature.py` | Create |
| `python_magnetrun/commands/info.py` | Add `register()` |
| `python_magnetrun/commands/add.py` | Add `register()` |
| `python_magnetrun/commands/plot.py` | Add `register()` |
| `python_magnetrun/commands/select.py` | Add `register()` |
| `python_magnetrun/commands/stats.py` | Add `register()` |
| `python_magnetrun/analysis/cli.py` | Add `register()` + decompose `main()` (coordinate with `analysis-subpackage-refactoring.plan.md` Phase 5.3 — single branch) |
| `python_magnetrun/processing/cli.py` | Add `register()` |
| `python_magnetrun/hybrid/cli.py` | Add `register()` |
| `python_magnetrun/tdms/log_parser.py` | Add `register()` |
| `python_magnetrun/comparison/cli.py` | Create with `register()` (see `cross-domain-comparison.prompt.md` Phase F — no standalone entry point) |
| `python_magnetrun/requests/cli.py` | Extract `_run()` + add `register("fetch")`; keep `main()` as deprecated shim |
| `python_magnetrun/config_cli.py` | Extract `_register_domain_subparsers()` + add `register("config")`; keep `main()` as deprecated shim |
| `python_magnetrun/cli.py` | Remove `_normalize_argv`; deprecate or remove `main()` |
| `python_magnetrun/args.py` | Split: move subcommand parsers into individual `register()` fns |
| `pyproject.toml` | Single `magnetrun` entry point; all old names as deprecated aliases |
| `tests/test-signature.py` | Keep test; `__main__` block can be removed once CLI is live |
