# Enhanced Error Logging - Implementation Checklist

## ✅ What's Done

### Core Implementation
- [x] Added `log_exception()` utility function to `hybrid/utils.py`
- [x] Added `format_exception_location()` utility function to `hybrid/utils.py`
- [x] Both functions fully documented with docstrings and examples

### File Updates
- [x] `hybrid/cli.py` - 5 exception handlers updated
- [x] `examples/plot_hybrid_minimal.py` - 3 exception handlers updated
- [x] `examples/plot_hybrid_with_pupitre_tdms.py` - 6 exception handlers updated
- [x] `python_magnetrun/python_magnetrun.py` - 1 exception handler updated

### Documentation
- [x] `docs/ERROR_LOGGING.md` - Comprehensive guide
- [x] `docs/ERROR_LOGGING_QUICK_REF.md` - Quick reference
- [x] `docs/ERROR_LOGGING_BEFORE_AFTER.md` - Visual comparisons
- [x] `docs/ERROR_LOGGING_ARCHITECTURE.md` - Technical details
- [x] `ENHANCED_ERROR_LOGGING_SUMMARY.md` - Implementation summary

### Testing & Examples
- [x] `examples/test_error_logging.py` - Test suite and demos
- [x] Test script is executable

## 🚀 How to Start Using It

### 1. For New Code

When writing new error handlers:

```python
from python_magnetrun.hybrid.utils import log_exception, format_exception_location

# Critical errors
try:
    critical_operation()
except Exception as e:
    log_exception("Critical operation failed", e, use_print=True, include_traceback=True)
    sys.exit(1)

# Warnings
try:
    optional_operation()
except Exception as e:
    log_exception("Warning: Optional operation failed", e, use_print=True, include_traceback=False)
    # Continue execution

# Inline
try:
    validate_input(value)
except ValueError as e:
    print(f"Validation error at {format_exception_location()}: {e}")
    return None
```

### 2. For Existing Code

Gradually update existing error handlers:

#### Step 1: Add imports
```python
from python_magnetrun.hybrid.utils import log_exception, format_exception_location
```

#### Step 2: Replace simple error handlers
```python
# OLD
except Exception as e:
    print(f"Error: {e}")

# NEW
except Exception as e:
    log_exception("Error loading data", e, use_print=True, include_traceback=True)
```

#### Step 3: Test the changes
Run your code and verify error messages are more informative.

### 3. Quick Test

Run the test script to see examples:

```bash
cd /home/LNCMI-G/christophe.trophime/github/python_magnetrun
python examples/test_error_logging.py
```

## 📋 Adoption Roadmap

### Phase 1: High-Priority Files (Already Done ✅)
- [x] `hybrid/cli.py` - Main CLI interface
- [x] `examples/plot_hybrid_minimal.py` - Common example
- [x] `examples/plot_hybrid_with_pupitre_tdms.py` - Complex example
- [x] `python_magnetrun/python_magnetrun.py` - Core module

### Phase 2: Next Priority (Optional)
Files with exception handlers that could benefit:

1. **hybrid/** module
   - [ ] `hybrid/plotting.py` - Has 4 exception handlers
   - [ ] `hybrid/hybrid_run.py` - Error handling exists
   - [ ] `hybrid/outliers.py` - Has error handling

2. **python_magnetrun/** module
   - [ ] `python_magnetrun/analysis/loaders.py` - Has 2 exception handlers
   - [ ] `python_magnetrun/analysis/processing.py` - Has 2 exception handlers
   - [ ] `python_magnetrun/processing/correlations.py` - Has exception handlers
   - [ ] `python_magnetrun/utils/files.py` - Has exception handlers

3. **Other examples/**
   - [ ] Various example scripts with error handling

### Phase 3: Complete Coverage
- [ ] Add to all new code as standard practice
- [ ] Document in contributing guidelines
- [ ] Add to code review checklist

## 📖 Documentation Reference

| Document                             | Purpose                      |
| ------------------------------------ | ---------------------------- |
| `docs/ERROR_LOGGING.md`              | Full guide with all features |
| `docs/ERROR_LOGGING_QUICK_REF.md`    | Quick copy-paste patterns    |
| `docs/ERROR_LOGGING_BEFORE_AFTER.md` | Visual comparisons           |
| `docs/ERROR_LOGGING_ARCHITECTURE.md` | Technical implementation     |
| `examples/test_error_logging.py`     | Working examples             |

## 🔍 Finding Candidates for Updates

Search for exception handlers that could be improved:

```bash
# Find simple error handlers
grep -r "except.*as.*:" --include="*.py" | grep -v "log_exception"

# Find print statements with errors
grep -r "print.*[Ee]rror" --include="*.py"

# Find exception handlers
grep -r "except Exception" --include="*.py"
```

## 💡 Tips for Good Error Messages

### ✅ Do
```python
log_exception("Failed to load FEPC configuration from /path/to/config.cfg", e, ...)
log_exception(f"Could not plot variable '{var_name}' for system '{system}'", e, ...)
log_exception("Error parsing RMS data file", e, ...)
```

### ❌ Don't
```python
log_exception("Error", e, ...)  # Too generic
log_exception("Failed", e, ...)  # No context
log_exception("Something went wrong", e, ...)  # Unhelpful
```

## 🎯 Decision Guide

| Situation              | Tool                        | Traceback?         |
| ---------------------- | --------------------------- | ------------------ |
| App crashes            | `log_exception`             | Yes (Full)         |
| Feature unavailable    | `log_exception`             | No (Location only) |
| User input invalid     | `format_exception_location` | N/A                |
| Optional feature fails | `log_exception`             | No (Location only) |
| Debugging              | `log_exception`             | Yes (Full)         |
| Production warning     | `log_exception`             | No (Location only) |

## 🐛 Troubleshooting

### Problem: Traceback shows wrong location

**Cause:** Exception re-raised or wrapped

**Solution:** Use `include_traceback=True` to see full call stack

### Problem: Too much output

**Cause:** Using `include_traceback=True` for common errors

**Solution:** Use `include_traceback=False` for warnings

### Problem: Import error

**Cause:** Trying to import from wrong location

**Solution:** 
```python
# From anywhere in the project (after hybrid is in python_magnetrun/)
from python_magnetrun.hybrid.utils import log_exception, format_exception_location

# If in hybrid/ module itself (using relative imports)
from .utils import log_exception, format_exception_location
```

## 📞 Questions or Issues?

1. Check the documentation in `docs/ERROR_LOGGING*.md`
2. Run `examples/test_error_logging.py` to see working examples
3. Look at updated files for real-world usage patterns

## 🎉 Benefits Summary

✨ **Better Debugging**
- See exact file, line, and function where errors occur
- Full call stack shows how you got there

✨ **Faster Development**
- Jump directly to problem code
- No more guessing where errors come from

✨ **Better User Experience**
- More informative error messages
- Easier to report bugs with context

✨ **Production Ready**
- Supports both print and logging
- Configurable verbosity
- Thread-safe implementation
