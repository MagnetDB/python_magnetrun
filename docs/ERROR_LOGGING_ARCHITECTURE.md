# Enhanced Error Logging - Architecture

## Information Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      Exception Occurs                            │
│                      (try/except block)                          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Call Error Utility                              │
│                                                                  │
│  Option 1: log_exception(msg, e, ...)                          │
│  Option 2: format_exception_location()                          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              Extract Exception Information                       │
│                                                                  │
│  • Exception type (ValueError, FileNotFoundError, etc.)         │
│  • Exception message                                            │
│  • Traceback frames (file, line, function)                      │
│  • Call stack                                                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
┌──────────────────┐          ┌──────────────────┐
│ Full Traceback   │          │ Concise Location │
│ (include_        │          │ (include_        │
│  traceback=True) │          │  traceback=False)│
└────────┬─────────┘          └────────┬─────────┘
         │                             │
         ▼                             ▼
┌──────────────────┐          ┌──────────────────┐
│ Format Output:   │          │ Format Output:   │
│                  │          │                  │
│ Message: ...     │          │ Message: ...     │
│ Traceback:       │          │   File: ...      │
│   File "x.py"    │          │   Line: 123      │
│     line 10      │          │   Function: foo  │
│   File "y.py"    │          │                  │
│     line 25      │          │                  │
│   ...            │          │                  │
└────────┬─────────┘          └────────┬─────────┘
         │                             │
         └──────────────┬──────────────┘
                        │
                        ▼
          ┌────────────────────────────┐
          │     Output Destination     │
          │                            │
          │  • Print to stdout         │
          │  • Log to logger           │
          │  • Return string           │
          └────────────────────────────┘
```

## Component Details

### 1. Exception Capture
```python
try:
    risky_operation()
except Exception as e:  # ← Exception captured here
    # Handle error
```

### 2. Error Utilities

#### log_exception()
```
Input:
  ├─ message: str           → Custom context message
  ├─ exception: Exception   → The caught exception
  ├─ logger_instance: Logger → Where to log (optional)
  ├─ use_print: bool        → Use print vs logger
  └─ include_traceback: bool → Full trace vs location only

Processing:
  ├─ Extract exception info via sys.exc_info()
  ├─ Format traceback via traceback module
  └─ Generate output string

Output:
  ├─ To logger (if logger_instance provided)
  ├─ To print (if use_print=True)
  └─ To module logger (default)
```

#### format_exception_location()
```
Input:
  └─ (uses sys.exc_info() internally)

Processing:
  ├─ Extract traceback
  ├─ Get last frame (where error occurred)
  └─ Format as "file:line:function"

Output:
  └─ Returns string: "filename.py:123:function_name"
```

### 3. Usage Patterns

```
Pattern A: Critical Error
├─ Use: log_exception(..., include_traceback=True)
├─ Shows: Full call stack
└─ When: Program cannot continue

Pattern B: Warning
├─ Use: log_exception(..., include_traceback=False)
├─ Shows: File, line, function only
└─ When: Recoverable error

Pattern C: Inline
├─ Use: format_exception_location()
├─ Shows: "file:line:function"
└─ When: Simple error message needed
```

## Integration Points

```
┌─────────────────────────────────────────────────────────────────┐
│                         Codebase                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  hybrid/                                                         │
│  ├─ utils.py          ← Error utilities defined here            │
│  ├─ cli.py            ← Import and use in CLI                   │
│  ├─ hybrid_data.py    ← Available for use                       │
│  └─ plotting.py       ← Available for use                       │
│                                                                  │
│  examples/                                                       │
│  ├─ plot_hybrid_minimal.py        ← Uses utilities             │
│  ├─ plot_hybrid_with_pupitre.py   ← Uses utilities             │
│  └─ test_error_logging.py         ← Demo/test script           │
│                                                                  │
│  python_magnetrun/                                               │
│  └─ python_magnetrun.py ← Uses utilities (or inline version)   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Dependency Graph

```
       ┌──────────────┐
       │   Python     │
       │   stdlib     │
       └──────┬───────┘
              │
              │ (imports: sys, traceback, logging)
              │
              ▼
       ┌──────────────┐
       │ hybrid/      │
       │ utils.py     │
       └──────┬───────┘
              │
              │ (imports utilities)
              │
       ┌──────┴───────┬────────────┬──────────────┐
       ▼              ▼            ▼              ▼
  ┌─────────┐   ┌─────────┐  ┌──────────┐  ┌──────────┐
  │ cli.py  │   │examples/│  │plotting  │  │   ...    │
  │         │   │ *.py    │  │  .py     │  │          │
  └─────────┘   └─────────┘  └──────────┘  └──────────┘
```

## Error Information Layers

```
Layer 1: Exception Object
├─ Type: ValueError, FileNotFoundError, etc.
├─ Message: "File not found: data.txt"
└─ Context: Built-in Python exception

Layer 2: Traceback
├─ Call Stack: List of frames
├─ Each Frame:
│  ├─ filename: "/path/to/file.py"
│  ├─ lineno: 123
│  ├─ name: "function_name"
│  └─ line: "    result = process(data)"
└─ Context: Where and how error occurred

Layer 3: Custom Context
├─ User Message: "Failed to load FEPC configuration"
├─ Operation Context: What was being attempted
└─ Context: Business logic / user intent

Final Output = Layer 3 + Layer 2 + Layer 1
```

## Example Data Flow

```
1. Exception raised:
   raise FileNotFoundError("config.cfg not found")

2. Captured in except:
   except Exception as e:

3. sys.exc_info() returns:
   (
     <class 'FileNotFoundError'>,
     FileNotFoundError("config.cfg not found"),
     <traceback object>
   )

4. traceback.extract_tb() gives:
   [
     FrameSummary(
       filename='/path/cli.py',
       lineno=327,
       name='main'
     ),
     FrameSummary(
       filename='/path/hybrid_data.py',
       lineno=150,
       name='__init__'
     ),
     ...
   ]

5. Formatted output:
   "Error creating HybridData: FileNotFoundError: config.cfg not found
    Traceback (most recent call last):
      File '/path/cli.py', line 327, in main
        data = HybridData(...)
      File '/path/hybrid_data.py', line 150, in __init__
        self._load_fepc_data()
      ...
    FileNotFoundError: config.cfg not found"
```

## Performance Considerations

```
| Operation                      | Time Complexity | Notes                      |
| ------------------------------ | --------------- | -------------------------- |
| format_exception_location()    | O(n)            | n = traceback depth        |
| log_exception(traceback=False) | O(n)            | n = traceback depth        |
| log_exception(traceback=True)  | O(n*m)          | n = depth, m = line length |
```

**Recommendation:** Use `include_traceback=False` for high-frequency errors.

## Thread Safety

The utilities use `sys.exc_info()` which is thread-local:
- ✅ Safe to use in multi-threaded applications
- ✅ Each thread gets its own exception info
- ✅ No race conditions
