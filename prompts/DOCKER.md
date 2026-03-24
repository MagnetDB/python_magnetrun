# Docker Setup for python_magnetrun

This document describes how to use the Docker Compose setup for JupyterLab with python_magnetrun.

## Prerequisites

- Docker (version 20.10 or later)
- Docker Compose (version 1.29 or later)

## Quick Start

1. **Build and start the JupyterLab container:**

   ```bash
   docker-compose up --build
   ```

   Or run in detached mode:

   ```bash
   docker-compose up -d --build
   ```

2. **Access JupyterLab:**

   Open your browser and navigate to:

   ```
   http://localhost:8888
   ```

   No token is required (configured for development ease).

3. **Stop the container:**

   ```bash
   docker-compose down
   ```

## Directory Structure

The following directories are mounted into the container:

- `./python_magnetrun` → `/workspace/python_magnetrun` (live code editing)
- `./python_magnetcooling` → `/workspace/python_magnetcooling` (dependency)
- `./notebooks` → `/workspace/notebooks` (your notebooks, persistent)
- `./data` → `/workspace/data` (read-only sample data)
- `./srvdata` → `/workspace/srvdata` (read-only server data)
- `./examples` → `/workspace/examples` (read-only example scripts)

## Working with the Setup

### Creating Notebooks

Notebooks created in JupyterLab will be saved in the `./notebooks/` directory on your host machine, so they persist even if you rebuild the container.

### Editing Code

Changes to Python files in `./python_magnetrun/` are immediately reflected in the container. You may need to restart the Python kernel in your notebook to pick up changes:

- In JupyterLab: `Kernel` → `Restart Kernel`

### Installing Additional Packages

To install additional Python packages:

1. **Temporary** (lost on container restart):

   ```bash
   docker-compose exec jupyterlab pip install package-name
   ```

2. **Permanent** (modify Dockerfile.jupyter):

   Add the package to `Dockerfile.jupyter` and rebuild:

   ```bash
   docker-compose up --build
   ```

### Accessing Container Shell

```bash
docker-compose exec jupyterlab bash
```

## Configuration

### Changing the Port

Edit `docker-compose.yml` and change the port mapping:

```yaml
ports:
  - "9999:8888"  # Access on http://localhost:9999
```

### Adding Authentication

For production use, add a token or password. Edit the `CMD` in `Dockerfile.jupyter`:

```dockerfile
# Generate token with: jupyter lab password
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

### GPU Support

Uncomment the `deploy` section in `docker-compose.yml` if you need GPU support.

## Troubleshooting

### Container won't start

Check logs:
```bash
docker-compose logs jupyterlab
```

### Permission issues

If you encounter permission issues with mounted volumes:

```bash
docker-compose exec jupyterlab chown -R root:root /workspace/notebooks
```

### Rebuilding from scratch

Remove all containers and rebuild:

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up
```

## Example Notebook

Here's a minimal example to verify the setup:

```python
# Test python_magnetrun installation
import python_magnetrun
print(f"python_magnetrun version: {python_magnetrun.__version__}")

# List available data files
from pathlib import Path
data_dir = Path('/workspace/data')
print(f"Data files: {list(data_dir.glob('*'))}")

# List server data
srvdata_dir = Path('/workspace/srvdata')
print(f"Server data files: {len(list(srvdata_dir.glob('*.json')))} JSON files")
```
