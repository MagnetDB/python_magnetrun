## Task: Refactor logging/print in a Python module

Refactor the logging and print usage in the Python module below to follow the two-channel model:

- `print` → stdout for **data output** (results, tables, computed values the user consumes)
- `logging` → stderr/file for **diagnostics** (errors, warnings, progress, debug trace)

### General rules (always apply)

1. Add a `setup_logging(verbose=False, logfile=None)` function if none exists, wiring:
   - A `StreamHandler(sys.stderr)` at WARNING by default, DEBUG if verbose=True
   - An optional `FileHandler` at DEBUG level if logfile is provided
2. Use `logging.getLogger(__name__)` at module level.
3. Do NOT convert print statements that already output computed results or structured
   data — keep those as `print`.
4. If logging is already present, preserve existing log levels and messages unless
   instructed otherwise by the options below.
5. Do not alter any logic, only logging/print wiring and setup.

### Option A — tag-based revert (apply if the code contains `# @data` tags)

Convert every `logger.*` line tagged with `# @data` back to a plain `print` statement.
Remove the tag comment. Leave all other logger calls untouched.

Example:
  BEFORE: logger.info("ΔT = %.3f K", delta_t)  # @data
  AFTER:  print(f"ΔT = {delta_t:.3f} K")

### Option B — pattern-based revert (apply if I describe a pattern below)

Convert `logger.*` calls back to `print` when they match the pattern I describe.
Leave all other logger calls untouched.

Pattern: [FILL IN — e.g. "any logger.info that formats a physical quantity",
          or "any logger call whose message starts with 'Result:' or 'Output:'"]

### Instructions

- Apply Option A if `# @data` tags are present in the code.
- Apply Option B if a pattern is specified above.
- Apply both if both are provided.
- If neither is provided, only apply the general rules.

### Module

[PASTE YOUR MODULE HERE]
