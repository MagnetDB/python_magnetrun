# Legacy Test Scripts

These standalone scripts predate the `magnetfs` package and are kept for reference.
For new usage, see [README.md](README.md) and the `magnetfs` CLI.

## Scripts

| Script | Purpose |
|---|---|
| `test_conversion.py` | Convert `.txt` files to Parquet and upload to RustFS |
| `test_read.py` | List and read Parquet files from RustFS |
| `test_plot.py` | Plot columns from Parquet files |
| `test_upload.py` | Simple CSV-to-Parquet upload pipeline (minimal version) |

## Usage

### test_conversion.py

```bash
# List available .txt files in your data directory
python test_conversion.py --datadir /path/to/data list --ext txt

# Convert a .txt file to Parquet and upload to RustFS
python test_conversion.py --datadir /path/to/data convert M10_2020.10.23---20_10_41.txt
```

The script handles the magnetrun `.txt` format (whitespace-separated, first row skipped)
and combines `Date` and `Time` columns into a `timestamp` column.

### test_read.py

```bash
# List files in the bucket
python test_read.py list

# Read and preview a Parquet file
python test_read.py read M10_2020.10.23---20_10_41.parquet
```

### test_plot.py

```bash
# List columns in a file
python test_plot.py M10_2020.10.23---20_10_41.parquet --keys

# Plot a specific column against time
python test_plot.py M10_2020.10.23---20_10_41.parquet --x timestamp --y Tin1

# Interactive mode (prompts for file and column selection)
python test_plot.py

# Save plot to a PNG file instead of displaying it
python test_plot.py M10_2020.10.23---20_10_41.parquet --x timestamp --y Tin1 --output plot.png
```

#### Displaying plots from inside the Docker container (X11 forwarding)

`plt.show()` requires a display server. On a Linux host, you can forward the X11 socket
into the container:

**1. Allow the container to connect to your X server (run once per session):**

```bash
xhost +local:docker
```

**2. Make sure `$DISPLAY` is set on the host** (usually `:0` or `:1`):

```bash
echo $DISPLAY
```

**3. Start the containers** — `docker-compose.yml` already passes `$DISPLAY` and mounts
`/tmp/.X11-unix`:

```bash
docker compose up -d
```

**4. Run the plot script:**

```bash
docker compose exec tester python test_plot.py M10_2020.10.23---20_10_41.parquet --x timestamp --y Tin1
```

> **Note:** The Dockerfile installs `libgl1-mesa-glx`, `libglib2.0-0`, and `python3-tk`
> which are required for X11 rendering. Rebuild the image after pulling changes:
> ```bash
> docker compose build tester
> ```

If X11 forwarding is not available (e.g. remote server, CI), use `--output plot.png` to
save the figure to a file instead.

## Running scripts inside Docker

```bash
# Build the tester image and start all services
docker compose up -d

# Run a script inside the tester container
docker compose exec tester python test_conversion.py --datadir /mnt/sshfs_data list
```

The `docker-compose.yml` maps `../nsrvdata` as a read-only volume at `/mnt/sshfs_data`
inside the tester container. Adjust this path to match your actual SSHFS mount point.
