# Python MagnetRun Requests CLI

This module provides command-line tools to connect to the Control/Monitoring site, retrieve magnet data, parts information (helices and rings), and manage experiment records.

## Overview

The CLI tool allows you to:
- Connect to srv-data server and authenticate
- List all parts (helices and rings) stored in the database
- Filter parts by type (helix, ring, or all)
- Retrieve magnet configurations and records
- Download configuration files
- Load data from cirrus monitoring system
- Perform sanity checks on records

## Installation

Make sure you have the package installed:

```bash
pip install -e .
```

## Basic Usage

### Authentication

All commands require authentication. You can specify your username with `--user`:

```bash
python -m python_magnetrun.requests.cli --user your.email@lncmi.cnrs.fr
```

The tool will prompt for your password. For non-interactive use, you can pipe the password:

```bash
echo "your_password" | python -m python_magnetrun.requests.cli --user your.email@lncmi.cnrs.fr
```

### Specify Server

By default, the tool connects to `https://srv-data-install.lncmi.cnrs.fr/`. To use a different server:

```bash
python -m python_magnetrun.requests.cli --user your.email@lncmi.cnrs.fr \
    --server https://your-server.domain.com/
```

## List Parts (Helices and Rings)

### List All Parts

Display all helices and rings from the srv-data database:

```bash
python -m python_magnetrun.requests.cli --user your.email@lncmi.cnrs.fr --list-parts
```

**Output example:**
```
================================================================================
Parts from srv-data server: https://srv-data-install.lncmi.cnrs.fr/
================================================================================

================================================================================
HELICES (14 found)
================================================================================
Name                 CAD Ref              Material        Geometry
--------------------------------------------------------------------------------
HL-31-H1            HL-31                CuAg0.1         HL-31.d
HL-31-H2            HL-31                CuAg0.1         HL-31.d
...

================================================================================
RINGS (13 found)
================================================================================
Name                 CAD Ref              Material        Geometry
--------------------------------------------------------------------------------
BR-31-R1            BR-31                CuAg0.1         BR-31.d
BR-31-R2            BR-31                CuAg0.1         BR-31.d
...

================================================================================
Total: 14 helices, 13 rings
================================================================================
```

### List Only Helices

Filter to show only helical parts:

```bash
python -m python_magnetrun.requests.cli --user your.email@lncmi.cnrs.fr \
    --list-parts --part-type helix
```

### List Only Rings

Filter to show only ring parts:

```bash
python -m python_magnetrun.requests.cli --user your.email@lncmi.cnrs.fr \
    --list-parts --part-type ring
```

## Data Management

### Save Configuration Files

Download and save configuration files to a local directory:

```bash
python -m python_magnetrun.requests.cli --user your.email@lncmi.cnrs.fr \
    --save --datadir ./magnet_configs
```

This will:
- Create the `./magnet_configs` directory if it doesn't exist
- Download magnet configuration files
- Save JSON files with magnet, part, and material data

### Perform Sanity Check

Check record consistency without saving:

```bash
python -m python_magnetrun.requests.cli --user your.email@lncmi.cnrs.fr --check
```

### Check and Save

Combine sanity check with file saving:

```bash
python -m python_magnetrun.requests.cli --user your.email@lncmi.cnrs.fr \
    --check --save --datadir ./data
```

## Load Data from Cirrus

### Load Default Feed (A1)

Load logs and XMLs from the cirrus.php monitoring system:

```bash
python -m python_magnetrun.requests.cli --user your.email@lncmi.cnrs.fr \
    --load-cirrus
```

### Load Specific Feed

Specify a different cirrus feed (A1, A2, A3, A4, etc.):

```bash
python -m python_magnetrun.requests.cli --user your.email@lncmi.cnrs.fr \
    --load-cirrus --cirrus-feed A3
```

## Logging and Debug

### Set Log Level

Control the verbosity of output:

```bash
# Debug mode - very verbose
python -m python_magnetrun.requests.cli --user your.email@lncmi.cnrs.fr \
    --list-parts --log-level DEBUG

# Info mode - standard output (default)
python -m python_magnetrun.requests.cli --user your.email@lncmi.cnrs.fr \
    --list-parts --log-level INFO

# Warning mode - only warnings and errors
python -m python_magnetrun.requests.cli --user your.email@lncmi.cnrs.fr \
    --list-parts --log-level WARNING

# Error mode - only errors
python -m python_magnetrun.requests.cli --user your.email@lncmi.cnrs.fr \
    --list-parts --log-level ERROR
```

## Combined Examples

### List Parts with Debug Logging

```bash
python -m python_magnetrun.requests.cli --user your.email@lncmi.cnrs.fr \
    --list-parts --part-type all --log-level DEBUG
```

### Check, Save, and List Helices

```bash
python -m python_magnetrun.requests.cli --user your.email@lncmi.cnrs.fr \
    --check --save --datadir ./output --list-parts --part-type helix
```

### Load Cirrus Data and Save

```bash
python -m python_magnetrun.requests.cli --user your.email@lncmi.cnrs.fr \
    --load-cirrus --cirrus-feed A2 --save --datadir ./cirrus_data
```

## Command-Line Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--user` | string | required | User email for authentication |
| `--server` | string | `https://srv-data-install.lncmi.cnrs.fr/` | Server URL |
| `--list-parts` | flag | - | List all parts (helices and rings) |
| `--part-type` | choice | `all` | Filter parts: `helix`, `ring`, or `all` |
| `--check` | flag | - | Perform sanity check for records |
| `--save` | flag | - | Save files to disk |
| `--datadir` | string | `.` | Directory for saved files |
| `--load-cirrus` | flag | - | Load logs and XMLs from cirrus.php |
| `--cirrus-feed` | string | `A1` | Cirrus feed identifier (A1, A2, A3, A4, etc.) |
| `--log-level` | choice | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

## Exit Behavior

- When using `--list-parts` alone, the program exits after displaying the parts list
- When combining `--list-parts` with `--check` or `--save`, the program continues to perform those operations after listing
- Use `Ctrl+C` to interrupt long-running operations

## Troubleshooting

### Connection Errors

If you see connection errors:
1. Check your network connection
2. Verify the server URL is correct with `--server`
3. Ensure you have VPN access if required

### Authentication Failures

If authentication fails:
1. Verify your username with `--user`
2. Check your password is correct
3. Contact your system administrator for account issues

### Empty Parts List

If `--list-parts` shows no results:
1. Check you're connected to the correct server
2. Verify your account has proper permissions
3. Use `--log-level DEBUG` to see detailed connection information

## Examples for Common Tasks

### Quick Parts Inventory

```bash
# Get a quick count of available parts
python -m python_magnetrun.requests.cli --user your.email@lncmi.cnrs.fr --list-parts
```

### Export All Data

```bash
# Download everything to a timestamped directory
python -m python_magnetrun.requests.cli --user your.email@lncmi.cnrs.fr \
    --check --save --datadir ./export_$(date +%Y%m%d)
```

### Verify Database Integrity

```bash
# Run checks with debug output
python -m python_magnetrun.requests.cli --user your.email@lncmi.cnrs.fr \
    --check --log-level DEBUG
```

### Monitor Specific Feed

```bash
# Load and save data from a specific monitoring feed
python -m python_magnetrun.requests.cli --user your.email@lncmi.cnrs.fr \
    --load-cirrus --cirrus-feed A4 --save --datadir ./monitoring/A4
```

## See Also

- Main project README: `../../README.md`
- Web scraping module: `webscrapping.py`
- Connection utilities: `connect.py`
