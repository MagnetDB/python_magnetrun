#!/usr/bin/env python3
import os
import sys

import numpy as np


def calculate_stats(directory, extension):
    # List all files in the directory with the given extension
    files = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(f".{extension}")
    ]

    if not files:
        print(f"No files with extension '.{extension}' found in '{directory}'.")
        return

    # Get file sizes in bytes
    sizes = [os.path.getsize(f) for f in files]

    # Calculate statistics
    stats = {
        "Number of files": len(sizes),
        "Average size (bytes)": np.mean(sizes),
        "Minimum size (bytes)": np.min(sizes),
        "Maximum size (bytes)": np.max(sizes),
        "Median size (bytes)": np.median(sizes),
        "First quartile (Q1) (bytes)": np.percentile(sizes, 25),
        "Third quartile (Q3) (bytes)": np.percentile(sizes, 75),
    }

    # Print results
    print(f"Statistics for .{extension} files in '{directory}':")
    print("-----------------------------------------------")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 file_stats.py <directory> <extension>")
        print("Example: python3 file_stats.py /path/to/dir txt")
        sys.exit(1)

    directory = sys.argv[1]
    extension = sys.argv[2]

    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)

    calculate_stats(directory, extension)
