# Enhanced Error Logging Implementation - Summary

## Overview

This implementation adds comprehensive error logging capabilities to the `python_magnetrun` codebase, providing detailed traceback information and error context when exceptions occur.

## What Was Added

### 1. Utility Functions in `python_magnetrun/hybrid/utils.py`

Two new utility functions for better error reporting:

#### `log_exception()`
- Logs exceptions with optional full traceback
- Supports both logging and print output
- Can show just file:line:function or full stack trace
- Configurable message and output method

#### `format_exception_location()`
- Returns concise "file.py:line:function" string
- Useful for inline error messages
- Lightweight alternative to full traceback

### 2. Updated Error Handlers

The following files have been updated with enhanced error handling:

#### `python_magnetrun/hybrid/cli.py`
- All exception handlers now use `log_exception()` or `format_exception_location()`
- Critical errors show full traceback
- Warnings show concise location

#### `examples/plot_hybrid_minimal.py`
- Added imports for error utilities
- Updated all exception handlers for plotting functions

#### `examples/plot_hybrid_with_pupitre_tdms.py`
- Added imports for error utilities
- Updated all exception handlers for data loading and plotting

#### `python_magnetrun/python_magnetrun.py`
- File loading errors now show full traceback
- Added helper function for exception location

## Usage Examples

### Before
```python
except Exception as e:
    print(f"Error: {e}")
```

### After (Critical Error)
```python
except Exception as e:
    log_exception("Error creating HybridData", e, use_print=True, include_traceback=True)
```

### After (Warning)
```python
except Exception as e:
    log_exception("Warning: Could not load data", e, use_print=True, include_traceback=False)
    print(f"  Error at {format_exception_location()}: {e}")
```

## Benefits

1. **Better Debugging**: Full traceback shows complete call stack
2. **Precise Location**: File, line, and function name for quick navigation
3. **Flexible Output**: Choose between full traceback or concise location
4. **Context Preservation**: Custom messages provide operation context
5. **Production Ready**: Supports both print and logging frameworks

## Documentation

Three documentation files have been created:

1. **`docs/ERROR_LOGGING.md`** - Comprehensive guide with examples and best practices
2. **`docs/ERROR_LOGGING_QUICK_REF.md`** - Quick reference for common patterns
3. **`examples/test_error_logging.py`** - Test script demonstrating all features

## Testing

Run the test script to see examples of error logging:

```bash
python examples/test_error_logging.py
```

This will demonstrate:
- Full traceback logging
- Concise location reporting
- Format exception location utility
- Nested function calls
- Different exception types
- Error logging with context

## Backwards Compatibility

✅ **Fully backward compatible**
- Existing error handling continues to work
- New utilities are optional
- No changes to public APIs
- Can be adopted incrementally

## Files Modified

1. `python_magnetrun/hybrid/utils.py` - Added error logging utilities
2. `python_magnetrun/hybrid/cli.py` - Updated 5 exception handlers
3. `examples/plot_hybrid_minimal.py` - Updated 3 exception handlers
4. `examples/plot_hybrid_with_pupitre_tdms.py` - Updated 6 exception handlers
5. `python_magnetrun/python_magnetrun.py` - Updated 1 exception handler

## Files Created

1. `docs/ERROR_LOGGING.md` - Full documentation
2. `docs/ERROR_LOGGING_QUICK_REF.md` - Quick reference guide
3. `examples/test_error_logging.py` - Test and demonstration script

## Example Output

### Full Traceback
```
Error creating HybridData: FileNotFoundError: [Errno 2] No such file or directory: 'config.cfg'
Traceback (most recent call last):
  File "/path/to/python_magnetrun/hybrid/cli.py", line 327, in main
    data = HybridData(
  File "/path/to/python_magnetrun/hybrid/hybrid_data.py", line 150, in __init__
    self._load_fepc_data()
  File "/path/to/python_magnetrun/hybrid/hybrid_data.py", line 342, in _load_fepc_data
    with open(cfg_path) as f:
FileNotFoundError: [Errno 2] No such file or directory: 'config.cfg'
```

### Concise Location
```
Warning: Could not plot data: ValueError: empty array
  File: plotting.py
  Line: 245
  Function: plot_data
  Error at plotting.py:245:plot_data: ValueError: empty array
```

## Best Practices

1. **Use full traceback for critical errors** - When program cannot continue
2. **Use concise location for warnings** - For recoverable issues
3. **Provide meaningful context** - Include operation details in message
4. **Consider your audience** - Scripts use print, libraries use logging
5. **Don't over-use traceback** - Full stack trace can be verbose

## Next Steps

To further enhance error reporting, you could:

1. Add colored terminal output for better visibility
2. Implement structured logging (JSON) for production
3. Add error categorization (user error vs system error)
4. Integrate with error tracking services (Sentry, etc.)
5. Add performance metrics (timing information)

## Questions?

See the full documentation in `docs/ERROR_LOGGING.md` or run the test script in `examples/test_error_logging.py`.
