import numpy as np
from scipy import stats
import pandas as pd
from typing import List, Tuple

def detect_breakpoints(data: np.ndarray, min_size: int = 30, significance: float = 0.05) -> List[Tuple[int, float]]:
    """
    Detect breakpoints in time series data using statistical tests.
    
    Parameters:
    -----------
    data : np.ndarray
        1D array of time series data
    min_size : int
        Minimum segment size to consider (default: 30)
    significance : float
        Statistical significance level (default: 0.05)
        
    Returns:
    --------
    List[Tuple[int, float]]
        List of tuples containing (breakpoint_index, p_value)
    """
    
    def _test_breakpoint(segment: np.ndarray, point: int) -> float:
        """
        Test if a point is a breakpoint using t-test.
        """
        if point < min_size or len(segment) - point < min_size:
            return 1.0
            
        segment1 = segment[:point]
        segment2 = segment[point:]
        
        # Perform two-sample t-test
        _, p_value = stats.ttest_ind(segment1, segment2)
        return p_value
    
    breakpoints = []
    n = len(data)
    
    # Scan through potential breakpoints
    for i in range(min_size, n - min_size):
        p_value = _test_breakpoint(data, i)
        print(i, p_value)
        
        if p_value < significance:
            # Check if this is a local minimum of p-values
            left_p = _test_breakpoint(data, i-1)
            right_p = _test_breakpoint(data, i+1)
            
            if p_value < left_p and p_value < right_p:
                breakpoints.append((i, p_value))
    
    return sorted(breakpoints, key=lambda x: x[1])  # Sort by significance

def analyze_segments(data: np.ndarray, breakpoints: List[int]) -> pd.DataFrame:
    """
    Analyze statistics for segments between breakpoints.
    
    Parameters:
    -----------
    data : np.ndarray
        Original time series data
    breakpoints : List[int]
        List of breakpoint indices
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing segment statistics
    """
    segments = []
    start_idx = 0
    
    for bp in sorted(breakpoints):
        segment = data[start_idx:bp]
        
        segment_stats = {
            'start_index': start_idx,
            'end_index': bp,
            'length': len(segment),
            'mean': np.mean(segment),
            'std': np.std(segment),
            'trend': np.polyfit(np.arange(len(segment)), segment, 1)[0]
        }
        
        segments.append(segment_stats)
        start_idx = bp
    
    # Add final segment
    final_segment = data[start_idx:]
    segments.append({
        'start_index': start_idx,
        'end_index': len(data),
        'length': len(final_segment),
        'mean': np.mean(final_segment),
        'std': np.std(final_segment),
        'trend': np.polyfit(np.arange(len(final_segment)), final_segment, 1)[0]
    })
    
    return pd.DataFrame(segments)# Example usage
    
"""
import numpy as np

# Generate sample data with known breakpoints
np.random.seed(42)
n_points = 300
data = np.concatenate([
    np.random.normal(0, 1, 100),
    np.random.normal(3, 1, 100),
    np.random.normal(1, 1, 100)
])
"""

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="enter input file")
parser.add_argument(
        "--site", help="specify a site (ex. M8, M9,...)", default="M9"
    )  # use housing instead
args = parser.parse_args()
print(f"args: {args}", flush=True)

file = args.input_file
filename = os.path.basename(file)
index = filename.index("_")
site = filename[0:index]
print(f"site detected: {site}")

insert = "tututu"

from python_magnetrun.MagnetRun import MagnetRun
mrun = MagnetRun.fromtxt(site, insert, file)

data = mrun.getData('Field').to_numpy().reshape(-1)

# Detect breakpoints
breakpoints = detect_breakpoints(data, min_size=10)
print("Detected breakpoints:", breakpoints)

# Analyze segments
segment_analysis = analyze_segments(data, [bp[0] for bp in breakpoints])
print("\nSegment analysis:")
print(segment_analysis)

import matplotlib.pyplot as plt
plt.plot(data, label='data')
for (x, y) in breakpoints:
    plt.axvline(x=x, color="green")

plt.grid()
plt.show()

