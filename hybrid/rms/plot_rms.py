"""
Quick RMS Data Plotter
======================

A simple utility to quickly plot variables from RMS files.
"""

import sys
from pathlib import Path
from rms_reader import RMSFileReader
import matplotlib.pyplot as plt
import numpy as np


def plot_rms_overview(filepath, output_file=None):
    """
    Create an overview plot with key variables from an RMS file.
    """
    reader = RMSFileReader(filepath)
    df = reader.read()
    var_info = reader.get_variable_info()
    
    # Separate analog and digital
    analog_vars = var_info[var_info['type'] == 'float32']['name'].tolist()
    digital_vars = var_info[var_info['type'] == 'bit']['name'].tolist()
    
    # Select interesting variables for overview
    # Temperatures
    temp_vars = [v for v in analog_vars if v.startswith('TT') and not v.endswith(('_D1', '_D2'))][:4]
    # Pressures
    pressure_vars = [v for v in analog_vars if v.startswith('PT')][:2]
    # Voltages
    voltage_vars = [v for v in analog_vars if v.startswith('PH_V')][:3]
    # Currents
    current_vars = [v for v in analog_vars if 'ALIM' in v][:2]
    
    # Alarms
    alarm_vars = [v for v in digital_vars if v.endswith('_D2')][:5]
    
    # Create figure
    n_plots = 0
    plot_vars = []
    
    if temp_vars:
        n_plots += 1
        plot_vars.append(('Temperatures (K)', temp_vars))
    if pressure_vars:
        n_plots += 1
        plot_vars.append(('Pressure (mBar)', pressure_vars))
    if voltage_vars:
        n_plots += 1
        plot_vars.append(('Voltages (V)', voltage_vars))
    if current_vars:
        n_plots += 1
        plot_vars.append(('Currents (A)', current_vars))
    if alarm_vars:
        n_plots += 1
        plot_vars.append(('Alarms', alarm_vars))
    
    if n_plots == 0:
        print("No recognizable variables found for overview plot")
        return
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots))
    if n_plots == 1:
        axes = [axes]
    
    fig.suptitle(f'RMS File Overview: {Path(filepath).name}', fontsize=14, fontweight='bold')
    
    for ax, (title, vars_to_plot) in zip(axes, plot_vars):
        for var in vars_to_plot:
            if var in df.columns:
                if title == 'Alarms':
                    # Plot digital signals as step functions
                    ax.step(df.index, df[var], where='post', label=var, linewidth=1.5)
                else:
                    ax.plot(df.index, df[var], label=var, linewidth=1)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel(title)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    return fig


def plot_variables(filepath, variable_names, output_file=None, same_plot=False):
    """
    Plot specific variables from an RMS file.
    
    Parameters:
    -----------
    filepath : str
        Path to RMS file
    variable_names : list
        List of variable names to plot
    output_file : str, optional
        Output filename for saving plot
    same_plot : bool
        If True, plot all variables on same axes; if False, create subplots
    """
    reader = RMSFileReader(filepath)
    df = reader.read()
    
    # Filter valid variables
    valid_vars = [v for v in variable_names if v in df.columns]
    
    if not valid_vars:
        print(f"Error: None of the requested variables found in file")
        print(f"Requested: {variable_names}")
        print(f"Available variables: {df.columns.tolist()}")
        return
    
    if same_plot:
        # Single plot with multiple y-axes if needed
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(valid_vars)))
        
        for i, var in enumerate(valid_vars):
            if i == 0:
                ax = ax1
                color = colors[i]
            else:
                ax = ax1.twinx()
                ax.spines['right'].set_position(('outward', 60 * (i - 1)))
                color = colors[i]
            
            ax.plot(df.index, df[var], color=color, label=var, linewidth=1.5)
            ax.set_ylabel(var, color=color)
            ax.tick_params(axis='y', labelcolor=color)
        
        ax1.set_xlabel('Time')
        ax1.grid(True, alpha=0.3)
        fig.suptitle(f'Variables: {", ".join(valid_vars)}', fontsize=12)
        
    else:
        # Separate subplots
        n_plots = len(valid_vars)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2.5*n_plots))
        if n_plots == 1:
            axes = [axes]
        
        fig.suptitle(f'RMS File: {Path(filepath).name}', fontsize=14, fontweight='bold')
        
        for ax, var in zip(axes, valid_vars):
            ax.plot(df.index, df[var], linewidth=1.5)
            ax.set_ylabel(var)
            ax.set_xlabel('Time')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = df[var].mean()
            std_val = df[var].std()
            min_val = df[var].min()
            max_val = df[var].max()
            
            stats_text = f'μ={mean_val:.3f}, σ={std_val:.3f}, min={min_val:.3f}, max={max_val:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    return fig


def plot_all_temperatures(filepath, output_file=None):
    """Plot all temperature sensors from an RMS file."""
    reader = RMSFileReader(filepath)
    df = reader.read()
    
    # Find all temperature variables
    temp_vars = [col for col in df.columns if col.startswith('TT') and not col.endswith(('_D1', '_D2'))]
    
    if not temp_vars:
        print("No temperature variables found")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for var in temp_vars:
        ax.plot(df.index, df[var], label=var, alpha=0.7, linewidth=1)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_title(f'All Temperature Sensors - {Path(filepath).name}', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    return fig


def plot_digital_timeline(filepath, output_file=None):
    """Create a timeline plot of all digital signals."""
    reader = RMSFileReader(filepath)
    df = reader.read()
    var_info = reader.get_variable_info()
    
    # Get digital variables
    digital_vars = var_info[var_info['type'] == 'bit']['name'].tolist()
    digital_vars = [v for v in digital_vars if v in df.columns]
    
    if not digital_vars:
        print("No digital variables found")
        return
    
    # Filter out always-off signals for clarity
    active_digital = []
    for var in digital_vars:
        if df[var].sum() > 0:  # Signal was ON at some point
            active_digital.append(var)
    
    if not active_digital:
        print("No active digital signals found")
        return
    
    fig, ax = plt.subplots(figsize=(14, max(6, len(active_digital) * 0.3)))
    
    # Create timeline
    y_positions = range(len(active_digital))
    
    for i, var in enumerate(active_digital):
        # Find periods where signal is ON
        signal = df[var]
        changes = signal.diff()
        
        # Get ON periods
        on_starts = df.index[changes == 1]
        on_ends = df.index[changes == -1]
        
        # Handle edge cases
        if signal.iloc[0] == 1:
            on_starts = [df.index[0]] + list(on_starts)
        if signal.iloc[-1] == 1:
            on_ends = list(on_ends) + [df.index[-1]]
        
        # Plot ON periods as horizontal bars
        for start, end in zip(on_starts, on_ends):
            ax.barh(i, (end - start).total_seconds(), left=start, height=0.8,
                   color='red', alpha=0.6)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(active_digital, fontsize=8)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_title(f'Digital Signal Timeline - {Path(filepath).name}', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    return fig


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
Quick RMS Plotter

Usage:
  python plot_rms.py overview <file> [output.png]
  python plot_rms.py vars <file> <var1> <var2> ... [--same-plot] [-o output.png]
  python plot_rms.py temps <file> [output.png]
  python plot_rms.py digital <file> [output.png]

Examples:
  python plot_rms.py overview data.rms
  python plot_rms.py vars data.rms PT205 TT200A -o myplot.png
  python plot_rms.py vars data.rms PT205 TT200A PH_V11 --same-plot
  python plot_rms.py temps data.rms temperatures.png
  python plot_rms.py digital data.rms alarms.png
        """)
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        if command == "overview":
            filepath = sys.argv[2]
            output = sys.argv[3] if len(sys.argv) > 3 else None
            plot_rms_overview(filepath, output)
        
        elif command == "vars":
            filepath = sys.argv[2]
            
            # Parse arguments
            same_plot = '--same-plot' in sys.argv
            output = None
            
            if '-o' in sys.argv:
                o_idx = sys.argv.index('-o')
                output = sys.argv[o_idx + 1]
            
            # Get variable names
            vars_to_plot = []
            for arg in sys.argv[3:]:
                if arg not in ['--same-plot', '-o'] and not arg.endswith('.png'):
                    vars_to_plot.append(arg)
            
            if not vars_to_plot:
                print("Error: No variables specified")
                sys.exit(1)
            
            plot_variables(filepath, vars_to_plot, output, same_plot)
        
        elif command == "temps":
            filepath = sys.argv[2]
            output = sys.argv[3] if len(sys.argv) > 3 else None
            plot_all_temperatures(filepath, output)
        
        elif command == "digital":
            filepath = sys.argv[2]
            output = sys.argv[3] if len(sys.argv) > 3 else None
            plot_digital_timeline(filepath, output)
        
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
