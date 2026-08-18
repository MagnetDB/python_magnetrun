# B-Profile Files

## Overview

B-profile files contain the axial magnetic field profile (B vs Z) of a magnet, normalized as a percentage, at one or more operating conditions. They are used for field homogeneity analysis and simulation post-processing.

- **Format**: ASCII CSV (comma-separated)
- **Naming convention**: `<site>_profiles.txt` (e.g. `M9_profiles.txt`)

## File Structure

```
Index,Position (mm),Profile at Tr (%),Profile at max (%)
0,299.7,-62.548..., -51.836...
1,299.8,-62.515..., -51.807...
...
```

### Columns

| Column | Description |
|--------|-------------|
| `Index` | Row index (0-based integer) |
| `Position (mm)` | Axial position along the magnet bore (mm) |
| `Profile at Tr (%)` | Normalized field at transition current (% of central field) |
| `Profile at max (%)` | Normalized field at maximum current (% of central field) |

> [!NOTE]
> Profile values are expressed as a percentage of the on-axis central field.
> Negative values indicate that the measurement point is off-centre (the profile
> is typically recorded over the full useful bore length, e.g. 299.7–599.5 mm for M9).

## Example

From `data/M9_profiles.txt` (M9 housing, 2999 points, Z = 299.7 → 599.5 mm, step 0.1 mm):

```
Index,Position (mm),Profile at Tr (%),Profile at max (%)
0,299.7,-62.54857446812466,-51.83603303013679
1,299.8,-62.51520612064338,-51.80767721498118
...
2998,599.5,-62.59870460073809,-51.87891871634039
```

## Python Interface

B-profile files are loaded with `MagnetData.frombprofile()`, which returns a `BProfileMagnetData` object (Type 0). The underlying data is a `pandas.DataFrame` with the four columns above.

```python
from python_magnetrun.magnetdata import MagnetData

mdata = MagnetData.frombprofile("data/M9_profiles.txt")
print(mdata.Keys)
# ['Index', 'Position (mm)', 'Profile at Tr (%)', 'Profile at max (%)']

df = mdata.getData("Position (mm)")
```

See [python_magnetrun/magnetdata.py](../python_magnetrun/magnetdata.py) and
[python_magnetrun/magnetdata_pandas.py](../python_magnetrun/magnetdata_pandas.py) for the implementation.
