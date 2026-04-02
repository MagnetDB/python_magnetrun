Refactor the logging/print setup in the Python module below to follow the two-channel model:
- `print` → stdout for **data output** (results, tables, computed values the user consumes)
- `logging` → stderr/file for **diagnostics** (errors, warnings, progress, debug trace)

Rules:
1. Add a `setup_logging(verbose=False, logfile=None)` function if none exists.
2. Use `logging.getLogger(__name__)` at module level.
3. `WARNING` and above always go to stderr (visible in terminal).
4. `DEBUG`/`INFO` go to stderr only if `--verbose` is set, otherwise file only.
5. Do NOT convert print statements that output computed results or structured data — keep those as `print`.
6. If logging is already present, preserve existing log levels and messages unless instructed otherwise.

[paste your module here]
