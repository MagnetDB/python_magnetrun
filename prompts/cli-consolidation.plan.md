# CLI Consolidation Plan

## Goal

Reduce 8 entry points to 3, replacing `python-magnetrun`, `magnetrun-analysis`,
`magnetrun-processing`, `hybrid-magnetrun`, and `magnetrun-pigbrother-logparser`
with a single `magnetrun` dispatcher; rename `srvdata-to-magnetrun` to
`magnetrun-fetch`; leave `magnetrun-config` unchanged.

## Target entry points

```toml
[project.scripts]
magnetrun        = "python_magnetrun.main:main"          # new unified dispatcher
magnetrun-fetch  = "python_magnetrun.requests.cli:main"  # renamed (was srvdata-to-magnetrun)
magnetrun-config = "python_magnetrun.config_cli:main"    # unchanged

# deprecated aliases — keep for one release cycle, then remove
python-magnetrun             = "python_magnetrun.cli:main"
srvdata-to-magnetrun         = "python_magnetrun.requests.cli:main"
magnetrun-analysis           = "python_magnetrun.analysis.cli:main"
magnetrun-processing         = "python_magnetrun.processing.cli:main"
hybrid-magnetrun             = "python_magnetrun.hybrid.cli:main"
magnetrun-pigbrother-logparser = "python_magnetrun.tdms.log_parser:main"
```

## `magnetrun` subcommands

All subcommands follow the pattern `magnetrun <subcommand> files... [options]`.
The subcommand comes **first** — files are positional arguments of each subcommand
parser, not the top-level parser.

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

    args = parser.parse_args()
    sys.exit(args._handler(args))


if __name__ == "__main__":
    main()
```

## Implementation order

1. **`commands/signature.py`** — self-contained, good template for the `register()` pattern
2. **`commands/{info,add,plot,select,stats}.py`** — add `register()` to existing modules;
   move `input_file` positional from top-level to each subcommand parser
3. **`analysis/cli.py`**, **`processing/cli.py`**, **`hybrid/cli.py`**,
   **`tdms/log_parser.py`** — add `register()` to each
4. **`python_magnetrun/main.py`** — write the unified dispatcher
5. **`pyproject.toml`** — add `magnetrun` and `magnetrun-fetch`, keep deprecated aliases
6. **`cli.py`** — remove `_normalize_argv`; reduce to a deprecated shim or delete

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
| `python_magnetrun/cli.py` | Remove `_normalize_argv`; deprecate or remove `main()` |
| `python_magnetrun/args.py` | Split: move subcommand parsers into individual `register()` fns |
| `pyproject.toml` | Add `magnetrun`, `magnetrun-fetch`; mark old names deprecated |
| `tests/test-signature.py` | Keep test; `__main__` block can be removed once CLI is live |
