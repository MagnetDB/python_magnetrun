# Profiling python_magnetrun

## Setup

```bash
source magnetrun-env/bin/activate
pip install -e ".[profiling]"
```

## 1. cProfile — call-level timing

Best first step: generates a `.prof` file for any entry point.

```bash
# analysis CLI
python -m cProfile -o cli.prof magnetrun-analysis [args]

# processing CLI
python -m cProfile -o processing.prof magnetrun-processing [args]

# or run a script directly
python -m cProfile -o out.prof -s cumulative python_magnetrun/analysis/processing.py
```

View results in the terminal:

```bash
python -m pstats cli.prof
# inside pstats: sort cumulative → stats 20
```

Or visually (opens a browser flamegraph):

```bash
snakeviz cli.prof
```

A `cli.prof` snapshot already exists in the project root.

## 2. pyinstrument — statistical profiler, no code changes

Lowest overhead, best for a quick overview:

```bash
pyinstrument -o profile.html --html -- magnetrun-analysis [args]
# open profile.html in a browser
```

Or wrap a Python script:

```bash
pyinstrument python_magnetrun/analysis/processing.py
```

## 3. line_profiler — line-by-line timing

For drilling into a specific function after cProfile/pyinstrument points at it.

Decorate the function(s) of interest:

```python
@profile  # added temporarily; remove before committing
def my_slow_function(...):
    ...
```

Run:

```bash
kernprof -l -v python_magnetrun/analysis/processing.py
# outputs processing.py.lprof and prints the table
```

## 4. memory_profiler — memory usage

Use when allocation/GC is suspected (e.g. large DataFrames, TDMS loading).

Decorate the function:

```python
@profile
def load_data(...):
    ...
```

Run:

```bash
python -m memory_profiler python_magnetrun/magnetdata_tdms.py
```

Or plot memory over time:

```bash
mprof run python_magnetrun/magnetdata_tdms.py
mprof plot
```

## Typical workflow

1. Run `pyinstrument` or `cProfile` + `snakeviz` to find the top offenders.
2. Narrow to a function with `line_profiler`.
3. If memory-bound, confirm with `memory_profiler`.
4. Fix, re-profile to verify.

## Known hot spots (update as discovered)

| Area | File | Symptom |
|------|------|---------|
| TDMS loading | `python_magnetrun/magnetdata_tdms.py` | slow on large files |
| Analysis processing | `python_magnetrun/analysis/processing.py` | TBD |
