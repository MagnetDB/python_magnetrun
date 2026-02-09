# Enhanced Error Logging - Quick Reference

## Import the utilities

```python
from python_magnetrun.hybrid.utils import log_exception, format_exception_location
```

## Common Patterns

### Pattern 1: Critical Error (Full Traceback)

Use when an error prevents the program from continuing:

```python
try:
    data = HybridData(base_dir, date, fepc_system="FEPC-LNCMI")
except Exception as e:
    log_exception("Error creating HybridData", e, use_print=True, include_traceback=True)
    return  # Exit or handle appropriately
```

**Output:**
```
Error creating HybridData: FileNotFoundError: [Errno 2] No such file or directory: '/path/to/config.cfg'
Traceback (most recent call last):
  File "/path/to/cli.py", line 327, in main
    data = HybridData(...)
  File "/path/to/hybrid_data.py", line 150, in __init__
    self._load_fepc_data()
  ...
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/config.cfg'
```

### Pattern 2: Warning (Location Only)

Use for recoverable errors or warnings:

```python
try:
    plot_data(values)
except Exception as e:
    log_exception("Warning: Could not plot data", e, use_print=True, include_traceback=False)
```

**Output:**
```
Warning: Could not plot data: ValueError: empty data array
  File: plotting.py
  Line: 245
  Function: plot_data
```

### Pattern 3: Inline Error Location

Use for simple error messages:

```python
try:
    result = validate_input(value)
except ValueError as e:
    print(f"Validation error at {format_exception_location()}: {e}")
    return
```

**Output:**
```
Validation error at validator.py:42:validate_input: Value must be positive
```

### Pattern 4: With Logger (Production Code)

Use in library code with proper logging:

```python
import logging
from python_magnetrun.hybrid.utils import log_exception

logger = logging.getLogger(__name__)

try:
    result = complex_operation()
except Exception as e:
    log_exception("Operation failed", e, logger_instance=logger, include_traceback=True)
```

## Decision Tree

```
Is this a critical error?
├─ YES: Use log_exception(..., include_traceback=True)
└─ NO: Is it recoverable/warning?
    ├─ YES: Use log_exception(..., include_traceback=False)
    └─ NO: Use format_exception_location() for inline message
```

## Parameters Quick Reference

### log_exception()

| Parameter           | Type      | Default  | Description                                |
| ------------------- | --------- | -------- | ------------------------------------------ |
| `message`           | str       | required | Custom error message                       |
| `exception`         | Exception | required | The caught exception                       |
| `logger_instance`   | Logger    | None     | Logger to use (None = print/module logger) |
| `use_print`         | bool      | False    | Use print() instead of logger              |
| `include_traceback` | bool      | True     | Include full traceback                     |

### format_exception_location()

| Parameter   | Type      | Default | Description                        |
| ----------- | --------- | ------- | ---------------------------------- |
| `exception` | Exception | None    | Not used, kept for API consistency |

**Returns:** `str` like `"filename.py:123:function_name"`

## Testing

Run the test suite to see examples:

```bash
python examples/test_error_logging.py
```

## Best Practices

✅ **DO:**
- Provide meaningful context in error messages
- Use full traceback for critical errors
- Use concise location for warnings
- Include variable values in context when helpful

❌ **DON'T:**
- Use generic messages like "Error" or "Failed"
- Always use full traceback (it can be verbose)
- Ignore exceptions silently
- Print traceback for expected/common errors

## Examples in Codebase

See these files for real-world usage:
- `python_magnetrun/hybrid/cli.py` - CLI error handling
- `examples/plot_hybrid_minimal.py` - Script error handling
- `python_magnetrun/python_magnetrun.py` - File loading errors
