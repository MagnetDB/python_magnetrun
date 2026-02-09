# Before and After: Error Logging Comparison

## Scenario 1: File Loading Error

### Before
```python
try:
    data = HybridData(base_dir, date, fepc_system="FEPC-LNCMI")
except Exception as e:
    print(f"Error creating HybridData: {e}")
    return
```

**Output:**
```
Error creating HybridData: [Errno 2] No such file or directory: '/data/hybrid/kHz/2025-01-06/FEPC-LNCMI.cfg'
```

**Problem:** You know there's an error, but not WHERE in the code it occurred or HOW you got there.

---

### After
```python
from python_magnetrun.hybrid.utils import log_exception

try:
    data = HybridData(base_dir, date, fepc_system="FEPC-LNCMI")
except Exception as e:
    log_exception("Error creating HybridData", e, use_print=True, include_traceback=True)
    return
```

**Output:**
```
Error creating HybridData: FileNotFoundError: [Errno 2] No such file or directory: '/data/hybrid/kHz/2025-01-06/FEPC-LNCMI.cfg'
Traceback (most recent call last):
  File "/home/.../hybrid/cli.py", line 327, in main
    data = HybridData(
           ^^^^^^^^^^^
  File "/home/.../hybrid/hybrid_data.py", line 150, in __init__
    self._load_fepc_data()
  File "/home/.../hybrid/hybrid_data.py", line 342, in _load_fepc_data
    config = parse_cfg_file(cfg_path)
             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/.../hybrid/kHz/fepc_reader.py", line 156, in parse_cfg_file
    with open(cfg_path, 'r') as f:
         ^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/data/hybrid/kHz/2025-01-06/FEPC-LNCMI.cfg'
```

**Benefit:** Now you can see:
- The exact file: `fepc_reader.py`
- The exact line: `156`
- The exact function: `parse_cfg_file`
- The complete call stack showing how you got there

---

## Scenario 2: Plotting Warning

### Before
```python
try:
    mdata = pupitre_data.getMData()
    values = mdata.getData(pupitre_field)
    ax.plot(time, values)
except Exception as e:
    print(f"Could not plot pupitre data: {e}")
```

**Output:**
```
Could not plot pupitre data: 'NoneType' object has no attribute 'getData'
```

**Problem:** What's None? Where did this happen? Hard to debug.

---

### After
```python
from python_magnetrun.hybrid.utils import log_exception, format_exception_location

try:
    mdata = pupitre_data.getMData()
    values = mdata.getData(pupitre_field)
    ax.plot(time, values)
except Exception as e:
    log_exception("Could not plot pupitre data", e, use_print=True, include_traceback=False)
    print(f"  Error at {format_exception_location()}: {e}")
```

**Output:**
```
Could not plot pupitre data: AttributeError: 'NoneType' object has no attribute 'getData'
  File: plot_hybrid_minimal.py
  Line: 142
  Function: plot_comparison
  Error at plot_hybrid_minimal.py:142:plot_comparison: 'NoneType' object has no attribute 'getData'
```

**Benefit:** Now you know:
- Error is in `plot_hybrid_minimal.py` line 142
- It's in function `plot_comparison`
- You can go directly to that line to fix it
- No verbose traceback for a simple warning

---

## Scenario 3: Quick Inline Error

### Before
```python
try:
    result = validate_input(value)
except ValueError as e:
    print(f"Validation error: {e}")
    return
```

**Output:**
```
Validation error: Value must be between 0 and 100
```

**Problem:** Where did validation fail? Which file? Which function?

---

### After
```python
from python_magnetrun.hybrid.utils import format_exception_location

try:
    result = validate_input(value)
except ValueError as e:
    print(f"Validation error at {format_exception_location()}: {e}")
    return
```

**Output:**
```
Validation error at validator.py:42:validate_input: Value must be between 0 and 100
```

**Benefit:** 
- Minimal change to code
- Still concise output
- But now includes location information
- Easy to jump to the exact line

---

## Scenario 4: Nested Function Calls

### Before
```python
def load_data():
    return process_file(path)

def process_file(path):
    return parse_content(read_file(path))

try:
    data = load_data()
except Exception as e:
    print(f"Error: {e}")
```

**Output:**
```
Error: invalid literal for int() with base 10: 'abc'
```

**Problem:** Which function failed? load_data? process_file? parse_content? read_file? Hard to tell.

---

### After
```python
from python_magnetrun.hybrid.utils import log_exception

def load_data():
    return process_file(path)

def process_file(path):
    return parse_content(read_file(path))

try:
    data = load_data()
except Exception as e:
    log_exception("Error loading data", e, use_print=True, include_traceback=True)
```

**Output:**
```
Error loading data: ValueError: invalid literal for int() with base 10: 'abc'
Traceback (most recent call last):
  File "script.py", line 10, in <module>
    data = load_data()
           ^^^^^^^^^^^
  File "script.py", line 3, in load_data
    return process_file(path)
           ^^^^^^^^^^^^^^^^^^
  File "script.py", line 6, in process_file
    return parse_content(read_file(path))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "parser.py", line 45, in parse_content
    value = int(line.strip())
            ^^^^^^^^^^^^^^^^^
ValueError: invalid literal for int() with base 10: 'abc'
```

**Benefit:**
- Complete call stack visible
- Shows progression: script.py → load_data → process_file → parser.py → parse_content
- Exact line in parser.py (line 45) where conversion failed
- Can trace data flow backwards

---

## Summary: When to Use What

| Situation                | Tool                                          | Example                               |
| ------------------------ | --------------------------------------------- | ------------------------------------- |
| **Critical errors**      | `log_exception(..., include_traceback=True)`  | Failed to load required configuration |
| **Warnings/recoverable** | `log_exception(..., include_traceback=False)` | Optional data not available           |
| **Quick inline errors**  | `format_exception_location()`                 | Validation errors, simple checks      |
| **Library code**         | `log_exception(..., logger_instance=logger)`  | Reusable components                   |
| **Scripts/CLI**          | `log_exception(..., use_print=True)`          | Command-line tools                    |

## Key Takeaways

✅ **Always provide context** - Don't just say "Error", explain what operation failed

✅ **Choose appropriate detail** - Full traceback for critical errors, location for warnings

✅ **Make debugging easy** - Include file:line:function so developers can jump to the code

✅ **Consider your audience** - Scripts use print, libraries use logging

✅ **Be consistent** - Use the same patterns throughout the codebase
