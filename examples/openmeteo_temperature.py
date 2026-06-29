"""Fetch current or historical temperature at the machine's location using Open-Meteo."""

import argparse
from datetime import date
from pathlib import Path

import geocoder
import matplotlib.pyplot as plt
import pandas as pd
import requests
from rich.console import Console
from rich.table import Table


def _get_location() -> tuple[float, float, str, str]:
    """Return (lat, lon, city, country) from the machine's public IP.

    Returns
    -------
    tuple[float, float, str, str]
        Latitude [°], longitude [°], city name, country code.

    Raises
    ------
    RuntimeError
        If the IP geolocation lookup fails.
    """
    g = geocoder.ip("me")
    if not g.ok:
        raise RuntimeError("Could not determine location from IP address.")
    lat, lon = g.latlng
    return lat, lon, g.city or "unknown city", g.country or "unknown country"


def get_current_temperature() -> None:
    """Print the current temperature at the machine's public IP location.

    Uses `geocoder` for IP-based geolocation and the Open-Meteo forecast API
    (no API key required).
    """
    lat, lon, city, country = _get_location()
    print(f"Location: {city}, {country} ({lat:.4f}, {lon:.4f})")

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "current": "temperature_2m"},
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()
    temp = data["current"]["temperature_2m"]
    unit = data["current_units"]["temperature_2m"]
    print(f"Current temperature: {temp} {unit}")


def fetch_historical_temperature(start: date, end: date) -> tuple[pd.DataFrame, str, str, str]:
    """Fetch hourly temperatures for a date range at the machine's location.

    Parameters
    ----------
    start : date
        First day of the requested range (inclusive).
    end : date
        Last day of the requested range (inclusive).

    Returns
    -------
    df : :class:`~pandas.DataFrame`
        Columns ``time`` (datetime) and ``temperature_2m`` [°C].
    unit : str
        Temperature unit string (e.g. ``"°C"``).
    city : str
        City name from geolocation.
    country : str
        Country code from geolocation.
    """
    lat, lon, city, country = _get_location()

    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": "temperature_2m",
        },
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    unit = data["hourly_units"]["temperature_2m"]
    df = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temperature_2m": data["hourly"]["temperature_2m"],
    })
    return df, unit, city, country


def print_historical(df: pd.DataFrame, unit: str, city: str, country: str,
                     start: date, end: date) -> None:
    """Print a formatted table of historical temperatures.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        Output of :func:`fetch_historical_temperature`.
    unit : str
        Temperature unit string.
    city : str
        City name.
    country : str
        Country code.
    start : date
        Start of the period.
    end : date
        End of the period.
    """
    lat_info = ""
    print(f"Location: {city}, {country}")
    print(f"Period: {start} → {end}\n")
    print(f"{'Time':<20} {'Temperature':>15}")
    print("-" * 36)
    for _, row in df.iterrows():
        temp_str = f"{row['temperature_2m']} {unit}" if pd.notna(row["temperature_2m"]) else "N/A"
        print(f"{str(row['time']):<20} {temp_str:>15}")


def _temp_color(temp: float) -> str:
    """Return a Rich color name based on temperature value [°C]."""
    if temp < 10:
        return "bright_blue"
    if temp < 20:
        return "cyan"
    if temp < 30:
        return "green"
    if temp < 35:
        return "yellow"
    return "red"


def print_rich_table(df: pd.DataFrame, unit: str, city: str, country: str,
                     start: date, end: date) -> None:
    """Render historical temperatures as a styled Rich table.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        Output of :func:`fetch_historical_temperature`.
    unit : str
        Temperature unit string shown in the column header.
    city : str
        City name shown in the table title.
    country : str
        Country code shown in the table title.
    start : date
        Start of the period shown in the table title.
    end : date
        End of the period shown in the table title.
    """
    console = Console()
    table = Table(
        title=f"Hourly temperature — {city}, {country}  ({start} → {end})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Time", style="dim", min_width=20)
    table.add_column(f"Temperature [{unit}]", justify="right", min_width=18)

    for _, row in df.iterrows():
        temp = row["temperature_2m"]
        if pd.isna(temp):
            table.add_row(str(row["time"]), "N/A")
        else:
            color = _temp_color(temp)
            table.add_row(str(row["time"]), f"[{color}]{temp} {unit}[/{color}]")

    console.print(table)


def export_data(df: pd.DataFrame, unit: str, output: Path, fmt: str, force: bool) -> None:
    """Write historical temperature data to a file.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        Output of :func:`fetch_historical_temperature`.
    unit : str
        Temperature unit string, written in column headers where applicable.
    output : :class:`~pathlib.Path`
        Destination file path.
    fmt : str
        Output format: ``"csv"``, ``"xlsx"``, or ``"duckdb"``.
    force : bool
        If ``False`` and *output* already exists, raise :exc:`FileExistsError`.

    Raises
    ------
    FileExistsError
        If *output* exists and *force* is ``False``.
    """
    if output.exists() and not force:
        raise FileExistsError(
            f"{output} already exists. Use --force to overwrite."
        )

    out_df = df.rename(columns={"temperature_2m": f"temperature_2m [{unit}]"})

    if fmt == "csv":
        out_df.to_csv(output, index=False)
    elif fmt == "xlsx":
        out_df.to_excel(output, index=False, engine="openpyxl")
    elif fmt == "duckdb":
        import duckdb
        con = duckdb.connect(str(output))
        con.execute("DROP TABLE IF EXISTS temperature")
        con.execute("CREATE TABLE temperature AS SELECT * FROM out_df")
        con.close()

    print(f"Data saved to {output} ({fmt})")


def plot_temperature(df: pd.DataFrame, unit: str, city: str, country: str,
                     start: date, end: date) -> None:
    """Plot hourly temperature as a time series.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        Output of :func:`fetch_historical_temperature`.
    unit : str
        Temperature unit string used for the y-axis label.
    city : str
        City name used in the plot title.
    country : str
        Country code used in the plot title.
    start : date
        Start of the period.
    end : date
        End of the period.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["time"], df["temperature_2m"], linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Temperature [{unit}]")
    ax.set_title(f"Hourly temperature — {city}, {country}  ({start} → {end})")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Parse arguments and dispatch to current or historical temperature fetch."""
    parser = argparse.ArgumentParser(
        description="Get temperature at current location via Open-Meteo."
    )
    parser.add_argument("--start", metavar="YYYY-MM-DD", help="Start date for historical range.")
    parser.add_argument("--end", metavar="YYYY-MM-DD", help="End date for historical range.")
    parser.add_argument("--output", metavar="FILE", help="Export historical data to a file.")
    parser.add_argument(
        "--format", dest="fmt", choices=["csv", "xlsx", "duckdb"], default="csv",
        help="Output format for --output (default: csv).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output file if it exists.")
    parser.add_argument("--plot", action="store_true", help="Plot historical temperature.")
    parser.add_argument("--table", action="store_true", help="Display historical data as a Rich table.")
    args = parser.parse_args()

    if bool(args.start) != bool(args.end):
        parser.error("--start and --end must be provided together.")

    if (args.output or args.plot or args.table) and not args.start:
        parser.error("--output, --plot and --table require --start and --end.")

    if args.force and not args.output:
        parser.error("--force requires --output.")

    if args.start:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        df, unit, city, country = fetch_historical_temperature(start, end)
        if args.table:
            print_rich_table(df, unit, city, country, start, end)
        else:
            print_historical(df, unit, city, country, start, end)
        if args.output:
            try:
                export_data(df, unit, Path(args.output), args.fmt, args.force)
            except FileExistsError as exc:
                print(f"Error: {exc}")
                raise SystemExit(1) from exc
        if args.plot:
            plot_temperature(df, unit, city, country, start, end)
    else:
        get_current_temperature()


if __name__ == "__main__":
    main()
