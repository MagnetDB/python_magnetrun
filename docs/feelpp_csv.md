# FeelPP CSV Format

## Overview

FeelPP CSV files are post-processing files created by [FeelPP](https://feelpp.org).

- **Format**: ASCII, comma-separated
- **Role**: Provide a list of output fields defined in the json description of the problem to be solved

> [!NOTE]
> The units attached to the saved fields are not displayed in the csv file. They depends on the problem which has been solved.
> This information shall be given in an extra file.
> For the moment , we assume SI units for the fields.
