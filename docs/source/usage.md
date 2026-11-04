# Usage

(installation)=
## Installation

Install from source using pip:

```console
git clone https://github.com/MagnetDB/python_magnetrun
cd python_magnetrun
pip install -e .
```

To include optional signal-processing extras:

```console
pip install -e ".[signal]"
```

To include all development tools:

```console
pip install -e ".[dev]"
```

## Quick Start

### Loading a magnet run from a text file

```python
from python_magnetrun.MagnetRun import MagnetRun

mrun = MagnetRun.fromtxt(housing="M9", site="mysite", filename="run.txt")

# List available data keys
print(mrun.getKeys())

# Access the underlying pandas DataFrame
df = mrun.getData()
print(df.describe())
```

### Loading from a TDMS file (PigBrother)

```python
mrun = MagnetRun.fromtdms(site="mysite", insert="M9", filename="run.tdms")
```

### Working with MagnetData directly

```python
from python_magnetrun.magnetdata import MagnetData

data = MagnetData.fromtxt("run.txt")

# Basic statistics for a field
stats = data.stats("IH")
print(stats)

# Add a derived quantity
data.addData("IH_ref", "IH_ref = Idcct1 + Idcct2")
```

### Filtering spikes

```python
from python_magnetrun.processing.filters import filterpikes

mrun = filterpikes(
    mrun,
    key="IH",
    inplace=True,
    threshold=5.0,
    twindows=10,
    debug=False,
    show=False,
    input_file="run.txt",
)
```

## JSON configuration files

The package bundles default JSON configuration files for field definitions and
per-housing sensor role assignments. These files are installed alongside the
package and are always available after `pip install`, without needing to know
the installation path.

### Field definitions (`*-defs.json`)

Three bundled files map channel names to physical metadata and cross-format
aliases:

- `pupitre-defs.json` — Pupitre `.txt` column names
- `pigbrother-defs.json` — PigBrother `Group/Channel` keys
- `hybrid-defs.json` — Hybrid `FEPC_system/variable` keys

Bare filenames are resolved automatically:

```python
from python_magnetrun.field_defs import load_defs, resolve_defs_file

# Works after installation — no need for a full path
defs = load_defs("pupitre-defs.json")

# See where it resolves to
print(resolve_defs_file("pupitre-defs.json"))
```

**Resolution order** for a bare filename:

1. Absolute path — used directly.
2. Relative path that exists in the current directory.
3. `~/.config/magnetrun/<filename>` — user override.
4. File bundled with the installed package.

To permanently override a bundled file for your site, place an edited copy in
`~/.config/magnetrun/`:

```bash
cp /path/to/my-pupitre-defs.json ~/.config/magnetrun/pupitre-defs.json
```

### Housing configuration (`<Housing>-housing-config.json`)

Per-housing JSON files (`M5-housing-config.json`, `M7-housing-config.json`,
`M8-housing-config.json`, `M9-housing-config.json`, `M10-housing-config.json`)
are bundled as read-only templates. `get_housing_config()` resolves in this order:

1. Explicit `json_file` argument.
2. `~/.config/magnetrun/<Housing>-housing-config.json` — persistent user override.
3. Hardcoded built-in default.
4. Runtime `overrides` (e.g. GR1/GR2 swapped for an atypical run) applied on top.

```python
from python_magnetrun.housing_config import (
    get_housing_config,
    get_bundled_housing_config_path,
    get_user_housing_config_path,
)

# Always works — falls back to built-in if no user override exists
cfg = get_housing_config("M9")

# Path to the read-only bundled template
get_bundled_housing_config_path("M9")

# Path to the user-writable config (directory is created if needed)
get_user_housing_config_path("M9")
```

To create a persistent user config from a bundled template:

```bash
magnetrun config housing M9-housing-config.json create M9 --from-builtin M9
cp M9-housing-config.json ~/.config/magnetrun/M9-housing-config.json
```

## Command-Line Interface

The package installs several CLI commands:

`python-magnetrun`
: Main entry point for viewing and analysing runs.

  ```console
  python-magnetrun --help
  ```

`magnetrun-analysis`
: Advanced signal analysis.

  ```console
  magnetrun-analysis --help
  ```

`magnetrun sessions`
: List the files recorded during each EXPERIENCES_LOG session of a housing
  (overview, archive, pupitre, incident, stats, hybrid and runlog files),
  whether or not an Overview file exists; files claimed by no session are
  grouped into unlogged records. Writes `sessions.json` in `--output-dir`.

  ```console
  magnetrun sessions --log EXPERIENCES_LOG.csv --housing M9 \
      --from 2018-01-01 --to 2019-01-01 --output-dir /tmp/M9_2018
  ```

`hybrid-magnetrun`
: Process hybrid magnet data (kHz, RMS, trigger).

  ```console
  hybrid-magnetrun --help
  ```

`srvdata-to-magnetrun`
: Download runs from the control/monitoring server.

  ```console
  srvdata-to-magnetrun --help
  ```

`magnetrun-pigbrother-logparser`
: Parse PigBrother TDMS log files.

  ```console
  magnetrun-pigbrother-logparser --help
  ```
