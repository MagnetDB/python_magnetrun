# Enhanced Error Logging and Traceback Information

This document explains the enhanced error logging capabilities added to the codebase to provide better debugging information when errors occur.

## Overview

The codebase now includes utilities for better error reporting that show:
- Full traceback information (optional)
- File, line number, and function name where the error occurred
- Detailed exception messages
- Support for both logging and print-based output

## Utility Functions

### `log_exception()`

Located in `hybrid/utils.py`, this function provides comprehensive error logging:

```python
def log_exception(
    message: str,
    exception: Exception,
    logger_instance: Optional[logging.Logger] = None,
    use_print: bool = False,
    include_traceback: bool = True,
) -> None:
    """
    Log exception with traceback information
    
    Parameters
    ----------
    message : str
        Custom error message to display
    exception : Exception
        The exception that was caught
    logger_instance : logging.Logger, optional
        Logger instance to use. If None, uses print or module logger
    use_print : bool
        If True and logger_instance is None, uses print instead of logger
    include_traceback : bool
        If True, includes full traceback. Otherwise just file, line, and function
    """
```

### `format_exception_location()`

Returns a concise string showing where the exception occurred:

```python
def format_exception_location(exception: Exception = None) -> str:
    """
    Get a concise string with file:line:function where exception occurred
    
    Returns
    -------
    str
        Formatted string like "file.py:123:function_name"
    """
```

## Usage Examples

### Example 1: Full Traceback with Print

For critical errors where you want to see the complete stack trace:

```python
from hybrid.utils import log_exception

try:
    # Some risky operation
    result = process_data(file_path)
except Exception as e:
    log_exception("Failed to process data", e, use_print=True, include_traceback=True)
    return
```

Output:
```
Failed to process data: FileNotFoundError: [Errno 2] No such file or directory: 'data.txt'
Traceback (most recent call last):
  File "/path/to/hybrid/cli.py", line 327, in main
    data = HybridData(
  File "/path/to/hybrid/hybrid_data.py", line 150, in __init__
    self._load_fepc_data()
  File "/path/to/hybrid/hybrid_data.py", line 342, in _load_fepc_data
    with open(cfg_path) as f:
FileNotFoundError: [Errno 2] No such file or directory: 'data.txt'
```

### Example 2: Concise Error Location

For warnings or non-critical errors where full traceback is too verbose:

```python
from hybrid.utils import log_exception, format_exception_location

try:
    # Some operation
    plot_data(values)
except Exception as e:
    log_exception("Warning: Could not plot data", e, use_print=True, include_traceback=False)
    print(f"  Error at {format_exception_location()}: {e}")
```

Output:
```
Warning: Could not plot data: ValueError: invalid value
  File: plotting.py
  Line: 245
  Function: plot_data
  Error at plotting.py:245:plot_data: ValueError: invalid value
```

### Example 3: Using with Logger

For production code with proper logging setup:

```python
import logging
from hybrid.utils import log_exception

logger = logging.getLogger(__name__)

try:
    # Some operation
    result = complex_calculation()
except Exception as e:
    log_exception("Calculation failed", e, logger_instance=logger, include_traceback=True)
```

### Example 4: Simple Location in Error Messages

For quick inline error messages:

```python
from hybrid.utils import format_exception_location

try:
    # Some operation
    data = load_file(path)
except ValueError as e:
    print(f"Value error at {format_exception_location()}: {e}")
    return
```

Output:
```
Value error at loader.py:89:load_file: invalid data format
```

## Updated Error Handling Patterns

### Before

```python
except Exception as e:
    print(f"Error: {e}")
```

### After (Critical Errors)

```python
except Exception as e:
    log_exception("Error creating HybridData", e, use_print=True, include_traceback=True)
```

### After (Warnings/Non-Critical)

```python
except Exception as e:
    log_exception("Warning: Could not load data", e, use_print=True, include_traceback=False)
    print(f"  Error at {format_exception_location()}: {e}")
```

## Files Updated

The following files have been updated with enhanced error logging:

1. **hybrid/utils.py** - Added utility functions
2. **hybrid/cli.py** - Updated all exception handlers
3. **examples/plot_hybrid_minimal.py** - Updated all exception handlers
4. **examples/plot_hybrid_with_pupitre_tdms.py** - Updated all exception handlers
5. **python_magnetrun/python_magnetrun.py** - Updated file loading error handler

## Best Practices

1. **Use full traceback for critical errors**: When an error prevents the program from continuing or completing its main task, use `include_traceback=True`

2. **Use concise location for warnings**: For recoverable errors or warnings, use `include_traceback=False` and add location with `format_exception_location()`

3. **Prefer logging in libraries**: Use `logger_instance` parameter when writing library code

4. **Use print for scripts**: Use `use_print=True` for CLI scripts and examples

5. **Context matters**: Always provide meaningful context in the error message:
   - ❌ `log_exception("Error", e, ...)`
   - ✅ `log_exception("Failed to load FEPC configuration", e, ...)`

## Configuration

### Logging Level

The logging level can be controlled via command-line arguments in `hybrid/cli.py`:

```bash
python -m hybrid.cli --log-level DEBUG
```

This affects what gets logged via the logger but not what's printed via `use_print=True`.

## Backwards Compatibility

The changes are backward compatible:
- Existing error handling continues to work
- New utilities are optional and can be adopted gradually
- No changes to public APIs or function signatures

## Future Improvements

Potential enhancements:
1. Add colored output for better visibility in terminals
2. Support for logging to files with rotation
3. Structured logging (JSON format) for production systems
4. Integration with external error tracking services (e.g., Sentry)
