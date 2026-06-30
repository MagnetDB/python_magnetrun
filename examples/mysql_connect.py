"""
mysql_connect.py
================
Connect DuckDB to a remote MySQL server.

Four modes are available:

live
    Attach the MySQL database via the ``mysql_scanner`` extension and print the
    full schema (tables + column names/types).  No local copy is made.  Use
    this mode first to inspect an unknown MySQL database before deciding on an
    export or poll strategy.

export
    Copy tables from MySQL into a local file.  Three output formats are
    supported (``--format``):

    csv      One CSV file per table in ``--output-dir``.
    parquet  One Parquet file per table in ``--output-dir``.
    duckdb   All tables in a single DuckDB file (``--output``).
             The file is created if it does not exist.
             Existing tables are replaced (``CREATE OR REPLACE``).

view
    Query a MySQL table (or custom SQL) once and render the result as a
    styled table in the terminal using ``rich``.  Use ``--table`` or
    ``--query``, optionally filtered with ``--where``.  ``--limit`` caps
    the number of rows returned (default: 200; use ``--limit 0`` for all rows).

plot
    Query a MySQL table (or custom SQL) once and display all selected data as
    a static chart.  Supports the same ``--plot``, ``--fields``, ``--x-field``,
    ``--where``, ``--query``, and ``--plot-options`` flags as ``poll``.
    ``--limit`` defaults to 0 (all rows).  ``--plot textual`` is not supported.

    matplotlib   Static chart in a native window (blocks until closed).
    plotly       Writes a static HTML file and opens it in the browser.
    dash         Interactive web app served at http://127.0.0.1:8050; data
                 loaded once on page open with full zoom/pan support.

poll
    Repeatedly query a single MySQL table at a fixed interval and display
    selected columns as a live plot.  Backend is selected via ``--plot``:

    table        Rich terminal table updated in-place every interval seconds.
                 All queried columns are shown; Ctrl-C to stop.table
    matplotlib   Live-updating chart in a native window (default).
    plotly       Writes an auto-refreshing HTML file (default: poll_output.html)
                 and opens it in the default browser on the first poll.
    textual      Full-screen TUI in the terminal.  One scatter plot per group,
                 rendered via plotext (pip install plotext).
                 Updated every interval seconds.  Press q to quit, p to pause.
    dash         Interactive web app served at http://127.0.0.1:8050.
                 Uses dcc.Interval for live updates — no page refresh.
                 Pause button, zoom/pan, and hover tooltips included.
                 Host and port are set via --dash-host / --dash-port.

    Two-source mode: add ``--table2`` (or ``--query2``) with optional
    ``--fields2`` and ``--where2`` to poll a second table simultaneously.
    Fields from each table are displayed in separate subplots that share the
    same x-axis, so zooming or panning in one subplot mirrors the other.
    Supported backends: matplotlib, plotly, dash.

    --plot-options accepts a JSON object with any of:
        type      "line" | "scatter" | "bar"        (default "line")
        layout    "subplots" | "overlay" | "groups" (default "subplots")
                    subplots  one subplot per field
                    overlay   all fields on a single shared axes
                    groups    user-defined groups; also set "groups" key
        groups    list of field-name lists, e.g. [["Icoil","Ucoil"],["tsb","teb"]]
                    required when layout="groups"
        figsize   [width, height] in inches         (default [12, 6], matplotlib only)
        colors    list of color strings, one per y-field
        font      font family name, e.g. "DejaVu Sans" / "Arial" / "monospace"
        fontsize  base font size in points, e.g. 12

Usage
-----
    # Inspect schema
    python mysql_connect.py --mode live \\
        --host myhost --user myuser --password mypw --database mydb

    # Export to CSV
    python mysql_connect.py --mode export --format csv --output-dir ./out \\
        --host myhost --user myuser --password mypw --database mydb

    # Export selected tables to DuckDB
    python mysql_connect.py --mode export --format duckdb \\
        --output magnetdb_mysql.duckdb --tables sites magnets \\
        --host myhost --user myuser --password mypw --database mydb

    # Poll every 10 s, matplotlib line plot
    python mysql_connect.py --mode poll --table measurements \\
        --fields timestamp Icoil Ucoil --x-field timestamp \\
        --interval 10 --limit 200 --plot matplotlib \\
        --plot-options '{"type":"line","figsize":[14,6],"colors":["steelblue","tomato"]}' \\
        --host myhost --user myuser --password mypw --database mydb

    # Same with plotly (opens browser, writes poll_output.html)
    python mysql_connect.py --mode poll --table measurements \\
        --fields timestamp Icoil Ucoil --x-field timestamp \\
        --interval 10 --limit 200 --plot plotly \\
        --plot-options '{"type":"scatter","colors":["navy","crimson"]}' \\
        --host myhost --user myuser --password mypw --database mydb

    # Two tables, shared x-axis — graph 1: Icoil/Ucoil, graph 2: tsb/teb (matplotlib)
    python mysql_connect.py --mode poll \\
        --table measurements --fields timestamp Icoil Ucoil --x-field timestamp \\
        --table2 temperatures --fields2 tsb teb \\
        --interval 10 --plot matplotlib \\
        --host myhost --user myuser --password mypw --database mydb

    # Same with Dash (interactive; zoom/pan is linked between the two graphs)
    python mysql_connect.py --mode poll \\
        --table measurements --fields timestamp Icoil Ucoil --x-field timestamp \\
        --table2 temperatures --fields2 tsb teb \\
        --interval 10 --plot dash \\
        --host myhost --user myuser --password mypw --database mydb

    # Two tables via raw SQL (tables not directly joinable)
    python mysql_connect.py --mode poll \\
        --query "SELECT t, Icoil, Ucoil FROM mysqldb.measurements ORDER BY t LIMIT 500" \\
        --fields Icoil Ucoil --x-field t \\
        --query2 "SELECT t, tsb, teb FROM mysqldb.temperatures ORDER BY t LIMIT 500" \\
        --fields2 tsb teb \\
        --interval 10 --plot dash \\
        --host myhost --user myuser --password mypw --database mydb

    # View most recent 50 rows of measurements as a table
    python mysql_connect.py --mode view --table measurements --limit 50 \\
        --host myhost --user myuser --password mypw --database mydb

    # View all rows matching a filter, no row cap
    python mysql_connect.py --mode view --table measurements \\
        --where "label='run42'" --limit 0 \\
        --host myhost --user myuser --password mypw --database mydb

    # List all databases on a host (--database not required)
    python mysql_connect.py --list-databases \\
        --host myhost --user myuser --password mypw

Connection parameters may also be supplied via environment variables
(MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB).
CLI flags take precedence over env vars; env vars take precedence over
built-in defaults.
"""

import argparse
import collections
import contextlib
import importlib.util
import json
import os
import sys
import time
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_HOST = "localhost"
_DEFAULT_PORT = 3306
_DEFAULT_OUTPUT = "magnetdb_mysql.duckdb"
_DEFAULT_OUTPUT_DIR = "."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def _build_dsn(
    host: str, port: int, user: str, password: str, database: str
) -> str:
    """Build a mysql_scanner DSN string."""
    return (
        f"host={host} port={port} user={user} "
        f"password={password} database={database}"
    )


def _load_mysql_extension(
    con: duckdb.DuckDBPyConnection, verbose: bool = False
) -> None:
    """Install and load the mysql_scanner DuckDB extension."""
    if verbose:
        print("Loading mysql_scanner extension...")
    con.execute("INSTALL mysql; LOAD mysql;")


def _attach_mysql(
    con: duckdb.DuckDBPyConnection, dsn: str, verbose: bool = False
) -> None:
    """Attach a MySQL database as the alias 'mysqldb'."""
    if verbose:
        print("Attaching MySQL database...")
    con.execute(f"ATTACH '{dsn}' AS mysqldb (TYPE mysql_scanner)")


def list_mysql_databases(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Return sorted list of database (schema) names visible on the MySQL server.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open connection with mysqldb attached (to information_schema or any db).

    Returns
    -------
    list[str]
        Schema names, sorted alphabetically.
    """
    rows = con.execute(
        "SELECT schema_name FROM mysqldb.information_schema.schemata "
        "ORDER BY schema_name"
    ).fetchall()
    return [r[0] for r in rows]


def list_mysql_tables(con: duckdb.DuckDBPyConnection, database: str) -> list[str]:
    """Return sorted list of table names in the attached MySQL database.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open connection with mysqldb already attached.
    database : str
        The MySQL database (schema) name to list tables from.

    Returns
    -------
    list[str]
        Table names, sorted alphabetically.
    """
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_catalog = 'mysqldb' AND table_schema = ? ORDER BY table_name",
        [database],
    ).fetchall()
    return [r[0] for r in rows]


def describe_table(
    con: duckdb.DuckDBPyConnection, table: str
) -> list[tuple]:
    """Return column descriptors for a single table.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open connection with mysqldb already attached.
    table : str
        Table name (unqualified).

    Returns
    -------
    list[tuple]
        Each tuple: (column_name, column_type, ...) as returned by DESCRIBE.
    """
    return con.execute(f"DESCRIBE mysqldb.{table}").fetchall()


# DuckDB type-name prefixes that are safe to plot as numeric y-values.
# MySQL types are translated by mysql_scanner: FLOAT→FLOAT, DOUBLE→DOUBLE,
# INT/BIGINT/…→INTEGER/BIGINT/…, DECIMAL→DECIMAL, etc.
_NUMERIC_PREFIXES = (
    "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "REAL",
    "INTEGER", "INT", "BIGINT", "SMALLINT", "TINYINT",
    "HUGEINT", "UBIGINT", "UINTEGER", "USMALLINT", "UTINYINT",
)

# DuckDB type-name prefixes that are suitable as an x-axis (time axis).
# MySQL TIMESTAMP / DATETIME → DuckDB TIMESTAMP; MySQL DATE → DuckDB DATE.
_TIMESTAMP_PREFIXES = ("TIMESTAMP", "TIMESTAMPTZ", "DATE", "DATETIME")


def _is_numeric_type(col_type: object) -> bool:
    """Return True if the DuckDB column type is numeric / plottable as y."""
    upper = str(col_type).upper().strip()
    return upper.startswith(_NUMERIC_PREFIXES)


def _is_timestamp_type(col_type: object) -> bool:
    """Return True if the DuckDB column type is temporal (suitable as x-axis)."""
    upper = str(col_type).upper().strip()
    return upper.startswith(_TIMESTAMP_PREFIXES)


def _is_id_field(name: str) -> bool:
    """Return True for columns that are ID keys and should not be plotted."""
    lower = name.lower()
    return lower == "id" or lower.endswith("_id")


def _safe_widget_id(name: str) -> str:
    """Return a CSS-safe identifier by replacing non-alphanumeric chars with '_'."""
    import re
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _auto_x_field(cols: Sequence[tuple[str, Any]]) -> str | None:
    """Return the name of the first TIMESTAMP column in *cols*, or None."""
    for name, col_type in cols:
        if _is_timestamp_type(col_type):
            return name
    return None


def list_numeric_columns(
    con: duckdb.DuckDBPyConnection, table: str
) -> list[tuple[str, str]]:
    """Return (name, type) pairs for numeric columns in *table*.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open connection with mysqldb already attached.
    table : str
        Table name (unqualified).

    Returns
    -------
    list[tuple[str, str]]
        ``[(column_name, column_type), ...]`` for all numeric columns,
        in schema order.
    """
    return [
        (c[0], c[1])
        for c in describe_table(con, table)
        if _is_numeric_type(c[1])
    ]


# ---------------------------------------------------------------------------
# Mode: live
# ---------------------------------------------------------------------------


def mode_live(args: argparse.Namespace) -> None:
    """Attach MySQL and print the full schema summary.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    dsn = _build_dsn(args.host, args.port, args.user, args.password, args.database)
    con = duckdb.connect()
    _load_mysql_extension(con, args.verbose)
    _attach_mysql(con, dsn, args.verbose)

    tables = list_mysql_tables(con, args.database)
    print(f"Connected to {args.database}@{args.host}:{args.port}")
    print(f"{len(tables)} table(s) found:\n")

    for table in tables:
        cols = describe_table(con, table)
        print(f"  {table}")
        for col in cols:
            col_name, col_type = col[0], col[1]
            if _is_numeric_type(col_type):
                marker = " *"
            elif _is_timestamp_type(col_type):
                marker = " @"
            else:
                marker = ""
            print(f"    {col_name:<30}  {col_type}{marker}")
        print()
    print("(* = numeric / y-axis  @ = timestamp / x-axis)")


# ---------------------------------------------------------------------------
# Mode: export
# ---------------------------------------------------------------------------


def _build_export_select(
    table: str,
    fields: list[str],
    time_field: str | None,
    start: str | None,
    end: str | None,
) -> str:
    """Build a SELECT for exporting *table* with optional column and time filtering."""
    field_list = ", ".join(fields) if fields else "*"
    sql = f"SELECT {field_list} FROM mysqldb.{table}"
    conditions: list[str] = []
    if time_field and start:
        conditions.append(f"{time_field} >= CAST('{start}' AS TIMESTAMP)")
    if time_field and end:
        conditions.append(f"{time_field} <= CAST('{end}' AS TIMESTAMP)")
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    return sql


def mode_export(args: argparse.Namespace) -> None:
    """Export MySQL tables to a local file (CSV, Parquet, or DuckDB).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    dsn = _build_dsn(args.host, args.port, args.user, args.password, args.database)

    if args.fmt == "duckdb":
        con = duckdb.connect(args.output)
        if args.verbose:
            print(f"Output database: {args.output}")
    else:
        con = duckdb.connect()
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    _load_mysql_extension(con, args.verbose)
    _attach_mysql(con, dsn, args.verbose)

    all_tables = list_mysql_tables(con, args.database)
    if args.tables:
        tables = args.tables
    elif getattr(args, "table", None):
        tables = [args.table]
    else:
        tables = all_tables

    unknown = set(tables) - set(all_tables)
    if unknown:
        print(
            f"Error: unknown table(s): {', '.join(sorted(unknown))}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── field / time-range filtering (requires a single table) ──────────────
    filtered = bool(args.export_fields or args.time_field or args.start or args.end)

    if filtered and len(tables) != 1:
        print(
            "Error: --export-fields / --time-field / --start / --end require "
            "exactly one table (use --tables TABLE).",
            file=sys.stderr,
        )
        sys.exit(1)

    time_field = args.time_field
    export_fields: list[str] = args.export_fields or []

    if filtered:
        table_name = tables[0]
        all_cols_raw = describe_table(con, table_name)
        all_col_names = [c[0] for c in all_cols_raw]

        # validate requested columns
        if export_fields:
            unknown_cols = set(export_fields) - set(all_col_names)
            if unknown_cols:
                print(
                    f"Error: unknown column(s) in '{table_name}': "
                    f"{', '.join(sorted(unknown_cols))}",
                    file=sys.stderr,
                )
                sys.exit(1)

        # auto-detect time field if not given
        if not time_field and (args.start or args.end):
            all_cols_typed = [(c[0], c[1]) for c in all_cols_raw]
            time_field = _auto_x_field(all_cols_typed)
            if time_field:
                if args.verbose:
                    print(f"Auto-selected time field: {time_field}")
            else:
                print(
                    f"Error: no TIMESTAMP column found in '{table_name}'. "
                    "Use --time-field to specify one.",
                    file=sys.stderr,
                )
                sys.exit(1)

        # ensure time_field is included in the SELECT when fields are restricted
        if export_fields and time_field and time_field not in export_fields:
            export_fields = [time_field] + export_fields

    if args.verbose:
        print(f"Exporting {len(tables)} table(s): {', '.join(tables)}")

    for table in tables:
        if filtered:
            select_sql = _build_export_select(
                table, export_fields, time_field, args.start, args.end
            )
        else:
            select_sql = f"SELECT * FROM mysqldb.{table}"

        if args.verbose:
            print(f"  SQL: {select_sql}")

        if args.fmt == "duckdb":
            con.execute(f"CREATE OR REPLACE TABLE {table} AS {select_sql}")
            if args.verbose:
                _row = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                n = _row[0] if _row is not None else 0
                print(f"  {table}: {n} row(s)")

        elif args.fmt == "csv":
            dest = Path(args.output_dir) / f"{table}.csv"
            con.execute(f"COPY ({select_sql}) TO '{dest}' (HEADER, DELIMITER ',')")
            if args.verbose:
                print(f"  {table} → {dest}")

        elif args.fmt == "parquet":
            dest = Path(args.output_dir) / f"{table}.parquet"
            con.execute(f"COPY ({select_sql}) TO '{dest}' (FORMAT PARQUET)")
            if args.verbose:
                print(f"  {table} → {dest}")

        elif args.fmt == "excel":
            if importlib.util.find_spec("pandas") is None or importlib.util.find_spec("openpyxl") is None:
                print(
                    "Error: pandas and openpyxl are required for --format excel."
                    "  pip install pandas openpyxl",
                    file=sys.stderr,
                )
                sys.exit(1)
            dest = Path(args.output_dir) / f"{table}.xlsx"
            df = con.execute(select_sql).df()
            df.to_excel(dest, index=False, engine="openpyxl")
            if args.verbose:
                print(f"  {table} → {dest}")

    print(f"Export complete: {len(tables)} table(s).")


# ---------------------------------------------------------------------------
# Mode: view
# ---------------------------------------------------------------------------


def mode_view(args: argparse.Namespace) -> None:
    """Display query results as a rich table in the terminal.

    Uses ``--table`` / ``--query``, ``--fields``, ``--where``, and ``--limit``
    from the CLI.  ``--limit 0`` means no row cap (show everything).
    """
    try:
        from rich.console import Console
        from rich.table import Table as RichTable
    except ImportError:
        print(
            "Error: rich is required for --mode view.  pip install rich",
            file=sys.stderr,
        )
        sys.exit(1)

    dsn = _build_dsn(args.host, args.port, args.user, args.password, args.database)
    con = duckdb.connect()
    _load_mysql_extension(con, args.verbose)
    _attach_mysql(con, dsn, args.verbose)

    if args.query:
        sql = args.query
    else:
        field_list = ", ".join(args.fields) if args.fields else "*"
        sql = f"SELECT {field_list} FROM mysqldb.{args.table}"
        if args.where:
            sql += f" WHERE {args.where}"
        view_limit = args.limit if args.limit is not None else 200
        if view_limit > 0:
            sql += f" LIMIT {view_limit}"

    if args.verbose:
        print(f"SQL: {sql}")

    cols, rows = _fetch_poll(con, sql)
    con.close()

    console = Console()
    rich_table = RichTable(show_header=True, header_style="bold cyan", show_lines=False)
    for col in cols:
        rich_table.add_column(col, overflow="fold")
    for row in rows:
        rich_table.add_row(*["" if v is None else str(v) for v in row])

    console.print(rich_table)
    console.print(f"[dim]{len(rows)} row(s)[/dim]")


# ---------------------------------------------------------------------------
# Mode: poll
# ---------------------------------------------------------------------------

_POLL_DEFAULTS: dict[str, Any] = {
    "type": "line",
    "figsize": [12, 6],
    "colors": [],
    # layout: "subplots" (one axes per field), "overlay" (all on one axes),
    #         "groups"   (user-defined groups; set "groups": [[f1,f2],[f3]] too)
    "layout": "subplots",
    "groups": [],
    "font": None,       # font family string, e.g. "DejaVu Sans" / "Arial"
    "fontsize": None,   # base font size in points, e.g. 12
}


def _resolve_groups(y_fields: list[str], opts: dict[str, Any]) -> list[list[str]]:
    """Return a list of field groups based on opts['layout'] / opts['groups'].

    overlay  → [[field1, field2, ...]]          (one group = all fields)
    subplots → [[field1], [field2], ...]         (one group per field)
    groups   → opts['groups'] if provided, else falls back to subplots
    """
    layout = opts.get("layout", "subplots")
    if layout == "overlay":
        return [list(y_fields)]
    if layout == "groups":
        user_groups = opts.get("groups", [])
        if user_groups:
            # Validate: every named field must be in y_fields
            flat = [f for g in user_groups for f in g]
            unknown = set(flat) - set(y_fields)
            if unknown:
                print(
                    f"Error: --plot-options groups reference unknown field(s): "
                    f"{', '.join(sorted(unknown))}",
                    file=sys.stderr,
                )
                sys.exit(1)
            return [list(g) for g in user_groups]
    # default: subplots
    return [[f] for f in y_fields]


def _parse_plot_options(raw: str | None) -> dict[str, Any]:
    """Merge user-supplied JSON plot options with defaults."""
    opts = dict(_POLL_DEFAULTS)
    if raw:
        try:
            user = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"Error: --plot-options is not valid JSON: {exc}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(user, dict):
            print("Error: --plot-options must be a JSON object.", file=sys.stderr)
            sys.exit(1)
        opts.update(user)
    return opts


def _build_poll_query(
    table: str,
    fields: list[str],
    where: str | None,
    limit: int,
    order_by: str | None,
    start_time: str | None = None,
) -> str:
    field_list = ", ".join(fields) if fields else "*"
    sql = f"SELECT {field_list} FROM mysqldb.{table}"
    conditions: list[str] = []
    if where:
        conditions.append(f"({where})")
    if start_time is not None and order_by:
        conditions.append(f"{order_by} >= CAST('{start_time}' AS TIMESTAMP)")
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    if order_by:
        sql += f" ORDER BY {order_by}"
    if limit > 0:
        sql += f" LIMIT {limit}"
    return sql


def _fetch_poll(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[list[str], list[tuple]]:
    rel = con.execute(sql)
    cols = [desc[0] for desc in rel.description]
    return cols, rel.fetchall()


def _fmt_x_latest(val: Any) -> str:
    """Format the last x-axis value for display in a chart title."""
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m-%d %H:%M:%S")
    return str(val)


def _apply_date_format(ax: Any, x_data: list) -> None:
    """Apply auto date locator/formatter to *ax* when *x_data* holds datetimes."""
    if not x_data or not any(hasattr(v, "strftime") for v in x_data if v is not None):
        return
    import matplotlib.dates as mdates
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))


# ── matplotlib backend ──────────────────────────────────────────────────────

def _poll_matplotlib(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    x_field: str | None,
    y_fields: list[str],
    opts: dict[str, Any],
    interval: float,
    count: int | None,
    verbose: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for --plot matplotlib.  pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    figsize = tuple(opts["figsize"])
    colors = opts["colors"] or [None] * len(y_fields)
    plot_type = opts["type"]
    groups = _resolve_groups(y_fields, opts)
    field_color = {f: i for i, f in enumerate(y_fields)}

    if opts["font"]:
        plt.rcParams["font.family"] = opts["font"]
    if opts["fontsize"]:
        plt.rcParams["font.size"] = opts["fontsize"]

    plt.ion()
    fig, axes = plt.subplots(len(groups), 1, figsize=figsize, sharex=True, squeeze=False)

    buffers: dict[str, collections.deque] = {
        f: collections.deque()
        for f in (y_fields + ([x_field] if x_field else []))
    }

    fields_label = ", ".join(y_fields)
    suptitle_obj = fig.suptitle(fields_label)
    poll_n = 0
    try:
        while count is None or poll_n < count:
            cols, rows = _fetch_poll(con, sql)
            if verbose:
                print(f"[poll {poll_n + 1}] {len(rows)} row(s) fetched")

            for f in buffers:
                buffers[f].clear()
            for row in rows:
                row_dict = dict(zip(cols, row, strict=False))
                for f in buffers:
                    if f in row_dict:
                        buffers[f].append(row_dict[f])

            x_data = (
                list(buffers[x_field])
                if x_field and x_field in buffers
                else list(range(len(rows)))
            )

            if x_data:
                suptitle_obj.set_text(f"{fields_label} ({_fmt_x_latest(x_data[-1])})")

            for ax, group in zip(axes[:, 0], groups, strict=False):
                ax.cla()
                for y_field in group:
                    y_data = list(buffers[y_field])
                    cidx = field_color[y_field]
                    color = colors[cidx] if cidx < len(colors) else None
                    kw = {"color": color, "label": y_field} if color else {"label": y_field}
                    if plot_type == "scatter":
                        ax.scatter(x_data, y_data, s=10, **kw)
                    elif plot_type == "bar":
                        ax.bar(x_data, y_data, **kw)
                    else:
                        ax.plot(x_data, y_data, **kw)
                # show y-label: single field name, or legend for multiple
                if len(group) == 1:
                    ax.set_ylabel(group[0])
                else:
                    ax.legend(loc="upper left", fontsize="small")
                ax.grid(True, linestyle="--", alpha=0.5)

            if x_field:
                axes[-1, 0].set_xlabel(x_field)

            _apply_date_format(axes[-1, 0], x_data)
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.draw()
            plt.pause(interval)
            poll_n += 1

    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.show()


# ── static backends (plot mode) ────────────────────────────────────────────


def _plot_static_matplotlib(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    x_field: str | None,
    y_fields: list[str],
    opts: dict[str, Any],
    verbose: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for --plot matplotlib.  pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    figsize = tuple(opts["figsize"])
    colors = opts["colors"] or [None] * len(y_fields)
    plot_type = opts["type"]
    groups = _resolve_groups(y_fields, opts)
    field_color = {f: i for i, f in enumerate(y_fields)}

    if opts["font"]:
        plt.rcParams["font.family"] = opts["font"]
    if opts["fontsize"]:
        plt.rcParams["font.size"] = opts["fontsize"]

    cols, rows = _fetch_poll(con, sql)
    if verbose:
        print(f"{len(rows)} row(s) fetched")

    row_dicts = [dict(zip(cols, r, strict=False)) for r in rows]
    x_data = (
        [r[x_field] for r in row_dicts]
        if x_field and row_dicts and x_field in row_dicts[0]
        else list(range(len(rows)))
    )

    fig, axes = plt.subplots(len(groups), 1, figsize=figsize, sharex=True, squeeze=False)

    for ax, group in zip(axes[:, 0], groups, strict=False):
        for y_field in group:
            y_data = [r.get(y_field) for r in row_dicts]
            cidx = field_color[y_field]
            color = colors[cidx] if cidx < len(colors) else None
            kw = {"color": color, "label": y_field} if color else {"label": y_field}
            if plot_type == "scatter":
                ax.scatter(x_data, y_data, s=10, **kw)
            elif plot_type == "bar":
                ax.bar(x_data, y_data, **kw)
            else:
                ax.plot(x_data, y_data, **kw)
        if len(group) == 1:
            ax.set_ylabel(group[0])
        else:
            ax.legend(loc="upper left", fontsize="small")
        ax.grid(True, linestyle="--", alpha=0.5)

    if x_field:
        axes[-1, 0].set_xlabel(x_field)

    _apply_date_format(axes[-1, 0], x_data)
    fig.autofmt_xdate()
    fig.suptitle(f"MySQL — {sql[:60]}…" if len(sql) > 60 else f"MySQL — {sql}")
    plt.tight_layout()
    plt.show()


def _plot_static_plotly(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    x_field: str | None,
    y_fields: list[str],
    opts: dict[str, Any],
    output_html: str,
    verbose: bool,
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Error: plotly is required for --plot plotly.  pip install plotly", file=sys.stderr)
        sys.exit(1)

    colors = opts["colors"] or [None] * len(y_fields)
    plot_type = opts["type"]
    groups = _resolve_groups(y_fields, opts)
    field_color = {f: i for i, f in enumerate(y_fields)}

    cols, rows = _fetch_poll(con, sql)
    if verbose:
        print(f"{len(rows)} row(s) fetched")

    row_dicts = [dict(zip(cols, r, strict=False)) for r in rows]
    x_data = (
        [r[x_field] for r in row_dicts]
        if x_field and row_dicts and x_field in row_dicts[0]
        else list(range(len(rows)))
    )

    subplot_titles = [", ".join(g) for g in groups]
    fig = make_subplots(
        rows=len(groups), cols=1, shared_xaxes=True,
        vertical_spacing=0.04, subplot_titles=subplot_titles,
    )

    for row_idx, group in enumerate(groups, start=1):
        for y_field in group:
            y_data = [r.get(y_field) for r in row_dicts]
            cidx = field_color[y_field]
            color = colors[cidx] if cidx < len(colors) else None
            marker = {"color": color} if color else {}

            if plot_type == "scatter":
                trace = go.Scatter(x=x_data, y=y_data, mode="markers", name=y_field, marker=marker)
            elif plot_type == "bar":
                trace = go.Bar(x=x_data, y=y_data, name=y_field, marker=marker)
            else:
                trace = go.Scatter(x=x_data, y=y_data, mode="lines", name=y_field, line=marker)
            fig.add_trace(trace, row=row_idx, col=1)

        y_title = group[0] if len(group) == 1 else ""
        fig.update_yaxes(title_text=y_title, row=row_idx, col=1)

    if x_field:
        fig.update_xaxes(title_text=x_field, row=len(groups), col=1)

    font_dict: dict[str, Any] = {}
    if opts["font"]:
        font_dict["family"] = opts["font"]
    if opts["fontsize"]:
        font_dict["size"] = opts["fontsize"]

    _layout_kw: dict[str, Any] = {
        "title": f"MySQL — {sql[:80]}…" if len(sql) > 80 else f"MySQL — {sql}",
        "height": max(300, 280 * len(groups)),
    }
    if font_dict:
        _layout_kw["font"] = font_dict
    fig.update_layout(**_layout_kw)

    html_path = Path(output_html).resolve()
    html_path.write_text(fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")
    webbrowser.open(html_path.as_uri())
    print(f"Plotly output written to {html_path}")


def _plot_static_dash(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    x_field: str | None,
    y_fields: list[str],
    opts: dict[str, Any],
    host: str,
    port: int,
    verbose: bool,
) -> None:
    try:
        import plotly.graph_objects as go
        from dash import Dash, dcc, html
    except ImportError:
        print("Error: dash is required for --plot dash.  pip install dash", file=sys.stderr)
        sys.exit(1)

    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    colors = opts["colors"] or [None] * len(y_fields)
    plot_type = opts["type"]
    groups = _resolve_groups(y_fields, opts)
    field_color = {f: i for i, f in enumerate(y_fields)}

    font_dict: dict[str, Any] = {}
    if opts["font"]:
        font_dict["family"] = opts["font"]
    if opts["fontsize"]:
        font_dict["size"] = opts["fontsize"]

    cols, rows = _fetch_poll(con, sql)
    if verbose:
        print(f"{len(rows)} row(s) fetched")

    row_dicts = [dict(zip(cols, r, strict=False)) for r in rows]
    x_data = (
        [r[x_field] for r in row_dicts]
        if x_field and row_dicts and x_field in row_dicts[0]
        else list(range(len(rows)))
    )

    figs = []
    for group in groups:
        fig = go.Figure()
        for y_field in group:
            y_data = [r.get(y_field) for r in row_dicts]
            cidx = field_color[y_field]
            color = colors[cidx] if cidx < len(colors) else None
            marker = {"color": color} if color else {}

            if plot_type == "scatter":
                trace = go.Scatter(x=x_data, y=y_data, mode="markers", name=y_field, marker=marker)
            elif plot_type == "bar":
                trace = go.Bar(x=x_data, y=y_data, name=y_field, marker=marker)
            else:
                trace = go.Scatter(x=x_data, y=y_data, mode="lines", name=y_field, line=marker)
            fig.add_trace(trace)

        _layout_kw: dict[str, Any] = {
            "title": ", ".join(group) if len(groups) > 1 else "",
            "height": 280,
            "xaxis_title": x_field or "",
            "yaxis_title": group[0] if len(group) == 1 else "",
            "margin": {"t": 40, "b": 40, "l": 60, "r": 20},
        }
        if font_dict:
            _layout_kw["font"] = font_dict
        fig.update_layout(**_layout_kw)
        figs.append(fig)

    heading = f"MySQL — {sql[:80]}…" if len(sql) > 80 else f"MySQL — {sql}"
    app = Dash(__name__, title="MySQL Plot")
    app.layout = html.Div(
        style={"fontFamily": opts["font"] or "sans-serif", "padding": "16px"},
        children=[
            html.H4(heading, style={"marginBottom": "8px"}),
            html.P(
                f"{len(rows)} row(s)",
                style={"color": "gray", "fontSize": "0.85em", "marginBottom": "12px"},
            ),
            html.Div([
                dcc.Graph(figure=fig, config={"displayModeBar": True})
                for fig in figs
            ]),
        ],
    )

    url = f"http://{host}:{port}"
    print(f"Dash server at {url}  (Ctrl+C to stop)")
    webbrowser.open(url)
    app.run(host=host, port=port, debug=False, use_reloader=False)


# ── plotly backend ──────────────────────────────────────────────────────────

def _poll_plotly(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    x_field: str | None,
    y_fields: list[str],
    opts: dict[str, Any],
    interval: float,
    count: int | None,
    output_html: str,
    verbose: bool,
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Error: plotly is required for --plot plotly.  pip install plotly", file=sys.stderr)
        sys.exit(1)

    colors = opts["colors"] or [None] * len(y_fields)
    plot_type = opts["type"]
    groups = _resolve_groups(y_fields, opts)
    field_color = {f: i for i, f in enumerate(y_fields)}

    html_path = Path(output_html).resolve()
    browser_opened = False
    fields_label = ", ".join(y_fields)
    title = fields_label
    poll_n = 0

    try:
        while count is None or poll_n < count:
            cols, rows = _fetch_poll(con, sql)
            if verbose:
                print(f"[poll {poll_n + 1}] {len(rows)} row(s) fetched")

            row_dicts = [dict(zip(cols, r, strict=False)) for r in rows]
            x_data = (
                [r[x_field] for r in row_dicts]
                if x_field and row_dicts and x_field in row_dicts[0]
                else list(range(len(rows)))
            )

            if x_data:
                title = f"{fields_label} ({_fmt_x_latest(x_data[-1])})"

            subplot_titles = [", ".join(g) for g in groups]
            fig = make_subplots(
                rows=len(groups),
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.04,
                subplot_titles=subplot_titles,
            )

            for row_idx, group in enumerate(groups, start=1):
                for y_field in group:
                    y_data = [r.get(y_field) for r in row_dicts]
                    cidx = field_color[y_field]
                    color = colors[cidx] if cidx < len(colors) else None
                    marker = {"color": color} if color else {}

                    if plot_type == "scatter":
                        trace = go.Scatter(x=x_data, y=y_data, mode="markers", name=y_field, marker=marker)
                    elif plot_type == "bar":
                        trace = go.Bar(x=x_data, y=y_data, name=y_field, marker=marker)
                    else:
                        trace = go.Scatter(x=x_data, y=y_data, mode="lines", name=y_field, line=marker)

                    fig.add_trace(trace, row=row_idx, col=1)

                # y-axis label: single field or blank (legend handles multi-field groups)
                y_title = group[0] if len(group) == 1 else ""
                fig.update_yaxes(title_text=y_title, row=row_idx, col=1)

            if x_field:
                fig.update_xaxes(title_text=x_field, row=len(groups), col=1)

            font_dict: dict[str, Any] = {}
            if opts["font"]:
                font_dict["family"] = opts["font"]
            if opts["fontsize"]:
                font_dict["size"] = opts["fontsize"]

            _layout_kw: dict[str, Any] = {
                "title": title,
                "height": max(300, 280 * len(groups)),
            }
            if font_dict:
                _layout_kw["font"] = font_dict
            fig.update_layout(**_layout_kw)

            # Inject meta-refresh so the browser reloads automatically
            html_body = fig.to_html(full_html=True, include_plotlyjs="cdn")
            refresh_tag = f'<meta http-equiv="refresh" content="{int(interval)}">'
            html_body = html_body.replace("<head>", f"<head>\n  {refresh_tag}", 1)
            html_path.write_text(html_body, encoding="utf-8")

            if not browser_opened:
                webbrowser.open(html_path.as_uri())
                browser_opened = True

            poll_n += 1
            if count is None or poll_n < count:
                time.sleep(interval)

    except KeyboardInterrupt:
        pass

    print(f"Plotly output written to {html_path}")


# ── textual backend ────────────────────────────────────────────────────────


def _poll_textual(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    x_field: str | None,
    y_fields: list[str],
    opts: dict[str, Any],
    interval: float,
    count: int | None,
    verbose: bool,
) -> None:
    try:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer, Header, Static
    except ImportError:
        print(
            "Error: textual is required for --plot textual.  pip install textual",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import plotext as _plt
    except ImportError:
        print(
            "Error: plotext is required for --plot textual.  pip install plotext",
            file=sys.stderr,
        )
        sys.exit(1)

    import datetime as _dt

    colors = opts["colors"] or []
    groups = _resolve_groups(y_fields, opts)

    def _to_plotext_x(vals: list) -> tuple[list, list, list]:
        """Convert x-values for plotext.

        Returns (x_values, xtick_positions, xtick_labels).
        Datetimes become float Unix timestamps so plotext never sees date
        strings and date_form() is never needed (it resets unpredictably
        across clf/subplot calls).
        """
        if not vals:
            return [], [], []
        clean = [v for v in vals if v is not None and hasattr(v, "timestamp")]
        if clean:
            x_num = [
                v.timestamp() if (v is not None and hasattr(v, "timestamp"))
                else float("nan")
                for v in vals
            ]
            pairs = [(xn, v) for xn, v in zip(x_num, vals, strict=False) if xn == xn]  # skip NaN
            if pairs:
                step = max(1, len(pairs) // 6)
                ticks = pairs[::step][:6]
                tick_pos = [p[0] for p in ticks]
                tick_lbl = [p[1].strftime("%H:%M:%S") for p in ticks]
            else:
                tick_pos, tick_lbl = [], []
            return x_num, tick_pos, tick_lbl
        return list(vals), [], []

    # plotext uses a module-level singleton figure.  Using one widget that owns
    # the entire render pass avoids concurrent clf()/scatter()/build() calls
    # from multiple widgets corrupting each other's output.
    class PollWidget(Static):  # type: ignore[type-arg]
        DEFAULT_CSS = """
        PollWidget {
            height: 1fr;
            padding: 0;
        }
        """

        def __init__(self, **kwargs: Any) -> None:
            super().__init__("Connecting…", **kwargs)
            self._x: list = []
            self._ys: dict[str, list[float]] = {f: [] for f in y_fields}

        def refresh_data(self, x: list, ys: dict[str, list[float]]) -> None:
            self._x = x
            self._ys = ys
            self._redraw()

        def on_resize(self, _event: object) -> None:
            self._redraw()

        def _redraw(self) -> None:
            w = self.size.width
            h = self.size.height
            if w < 10 or h < 4:
                return

            n = len(groups)
            x_vals, tick_pos, tick_lbl = _to_plotext_x(self._x)
            any_plotted = False

            try:
                _plt.clf()
                if n > 1:
                    _plt.subplots(n, 1)
                _plt.plotsize(w, h)
                _plt.theme("dark")

                for gi, group in enumerate(groups):
                    if n > 1:
                        _plt.subplot(gi + 1, 1)
                    _plt.title(", ".join(group))
                    if x_field:
                        _plt.xlabel(x_field)
                    if tick_pos:
                        _plt.xticks(tick_pos, tick_lbl)

                    for field in group:
                        y = self._ys.get(field, [])
                        if x_vals and y and len(x_vals) == len(y):
                            fi = y_fields.index(field)
                            kw: dict[str, Any] = {"label": field}
                            c = colors[fi] if fi < len(colors) else None
                            if c:
                                kw["color"] = c
                            _plt.scatter(x_vals, y, **kw)
                            any_plotted = True

                if any_plotted:
                    self.update(_plt.build())
                else:
                    self.update("Waiting for data…")
            except (ValueError, RuntimeError, TypeError, AttributeError) as exc:
                self.update(f"Plot error: {exc}")

    class PollApp(App):  # type: ignore[type-arg]
        THEME = "textual-dark"
        CSS = """
        Screen { background: $surface; }
        #plot  { height: 1fr; }
        #status {
            height: 1;
            background: $primary-darken-3;
            color: $text-muted;
            padding: 0 1;
        }
        """

        BINDINGS = [
            ("q", "quit", "Quit"),
            ("p", "toggle_pause", "Pause / Resume"),
        ]

        def __init__(self) -> None:
            super().__init__()
            self._fields_label = ", ".join(y_fields)
            self.title = self._fields_label
            self._poll_n = 0
            self._paused = False

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield PollWidget(id="plot")
            yield Static("Connecting…", id="status")
            yield Footer()

        def on_mount(self) -> None:
            self.set_interval(interval, self._do_poll)

        async def _do_poll(self) -> None:
            if self._paused:
                return
            if count is not None and self._poll_n >= count:
                self.exit()
                return

            try:
                cols, rows = _fetch_poll(con, sql)
            except duckdb.Error as exc:
                self.query_one("#status", Static).update(f"Error: {exc}")
                return

            if x_field and rows:
                x_latest = dict(zip(cols, rows[-1], strict=False)).get(x_field)
                if x_latest is not None:
                    self.title = f"{self._fields_label} ({_fmt_x_latest(x_latest)})"

            x_data: list = []
            ys: dict[str, list[float]] = {f: [] for f in y_fields}
            for row in rows:
                rd = dict(zip(cols, row, strict=False))
                if x_field:
                    x_data.append(rd.get(x_field))
                for f in y_fields:
                    val = rd.get(f)
                    try:
                        ys[f].append(float(val) if val is not None else float("nan"))
                    except (TypeError, ValueError):
                        ys[f].append(float("nan"))

            if not x_field:
                x_data = list(range(len(rows)))

            with contextlib.suppress(Exception):
                self.query_one("#plot", PollWidget).refresh_data(x_data, ys)

            self._poll_n += 1
            now = _dt.datetime.now().strftime("%H:%M:%S")
            paused_tag = "  [PAUSED]" if self._paused else ""
            self.query_one("#status", Static).update(
                f"Poll #{self._poll_n}  rows={len(rows)}"
                f"  interval={interval}s  updated={now}{paused_tag}"
            )

        def action_toggle_pause(self) -> None:
            self._paused = not self._paused

    PollApp().run()


# ── terminal table backend ─────────────────────────────────────────────────


def _poll_table(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    x_field: str | None,
    y_fields: list[str],
    opts: dict[str, Any],
    interval: float,
    count: int | None,
    verbose: bool,
) -> None:
    try:
        from rich.console import Console, Group
        from rich.live import Live
        from rich.table import Table as RichTable
        from rich.text import Text
    except ImportError:
        print(
            "Error: rich is required for --plot table.  pip install rich",
            file=sys.stderr,
        )
        sys.exit(1)

    import datetime as _dt

    fields_label = ", ".join(y_fields)
    poll_n = 0
    console = Console()

    def _build_rich_table(cols: list[str], rows: list[tuple]) -> RichTable:
        t = RichTable(
            show_header=True, header_style="bold cyan",
            show_lines=False, expand=False,
        )
        for col in cols:
            t.add_column(col, overflow="fold", no_wrap=(col == x_field))
        for row in rows:
            t.add_row(*["—" if v is None else str(v) for v in row])
        return t

    try:
        with Live(console=console, refresh_per_second=4, screen=False) as live:
            while count is None or poll_n < count:
                try:
                    cols, rows = _fetch_poll(con, sql)
                except duckdb.Error as exc:
                    live.update(Text(f"Poll error: {exc}", style="red"))
                    time.sleep(interval)
                    continue

                now = _dt.datetime.now().strftime("%H:%M:%S")
                x_info = ""
                if x_field and rows:
                    last_val = dict(zip(cols, rows[-1], strict=False)).get(x_field)
                    if last_val is not None:
                        x_info = f"  ·  {x_field}: {_fmt_x_latest(last_val)}"

                status = Text(
                    f"[Poll #{poll_n + 1}]  {len(rows)} row(s)"
                    f"  ·  updated {now}{x_info}"
                    f"  ·  interval {interval}s  ·  Ctrl-C to stop",
                    style="dim",
                )
                live.update(Group(status, _build_rich_table(cols, rows)))

                if verbose:
                    print(f"[poll {poll_n + 1}] {len(rows)} row(s)")

                poll_n += 1
                if count is None or poll_n < count:
                    time.sleep(interval)

    except KeyboardInterrupt:
        pass


# ── dash backend ───────────────────────────────────────────────────────────


def _poll_dash(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    x_field: str | None,
    y_fields: list[str],
    opts: dict[str, Any],
    interval: float,
    count: int | None,
    host: str,
    port: int,
    verbose: bool,
) -> None:
    try:
        import plotly.graph_objects as go
        from dash import Dash, Input, Output, State, dcc, html, no_update
        from dash.exceptions import PreventUpdate
    except ImportError:
        print(
            "Error: dash is required for --plot dash.  pip install dash",
            file=sys.stderr,
        )
        sys.exit(1)

    import datetime as _dt
    import logging

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    colors = opts["colors"] or [None] * len(y_fields)
    plot_type = opts["type"]
    groups = _resolve_groups(y_fields, opts)
    field_color = {f: i for i, f in enumerate(y_fields)}

    font_dict: dict[str, Any] = {}
    if opts["font"]:
        font_dict["family"] = opts["font"]
    if opts["fontsize"]:
        font_dict["size"] = opts["fontsize"]

    app = Dash(__name__, title="MySQL Poll", suppress_callback_exceptions=True)
    graph_ids = [f"graph-{i}" for i in range(len(groups))]
    interval_ms = int(interval * 1000)
    fields_label = ", ".join(y_fields)

    app.layout = html.Div(
        style={"fontFamily": opts["font"] or "sans-serif", "padding": "16px"},
        children=[
            html.H4(id="heading", children=fields_label, style={"marginBottom": "8px"}),
            html.Div(
                style={"display": "flex", "alignItems": "center",
                       "gap": "16px", "marginBottom": "12px"},
                children=[
                    html.Button("⏸ Pause / Resume", id="pause-btn", n_clicks=0),
                    html.Span(id="status-text",
                              style={"color": "gray", "fontSize": "0.85em"}),
                ],
            ),
            dcc.Interval(id="interval", interval=interval_ms,
                         n_intervals=0, disabled=False),
            dcc.Store(id="paused", data=False),
            dcc.Store(id="poll-n", data=0),
            html.Div([
                dcc.Graph(id=gid, config={"displayModeBar": True})
                for gid in graph_ids
            ]),
        ],
    )

    @app.callback(
        Output("paused", "data"),
        Output("interval", "disabled"),
        Input("pause-btn", "n_clicks"),
        State("paused", "data"),
        prevent_initial_call=True,
    )
    def toggle_pause(n_clicks, paused):
        new_state = not paused
        return new_state, new_state

    @app.callback(
        [Output(gid, "figure") for gid in graph_ids]
        + [Output("status-text", "children"), Output("heading", "children"),
           Output("poll-n", "data")],
        Input("interval", "n_intervals"),
        State("poll-n", "data"),
    )
    def refresh(n_intervals, poll_n):
        if count is not None and poll_n >= count:
            raise PreventUpdate

        try:
            cols, rows = _fetch_poll(con, sql)
        except duckdb.Error as exc:
            if verbose:
                print(f"Poll error: {exc}", file=sys.stderr)
            raise PreventUpdate from exc

        row_dicts = [dict(zip(cols, r, strict=False)) for r in rows]
        x_data = (
            [r[x_field] for r in row_dicts]
            if x_field and row_dicts and x_field in row_dicts[0]
            else list(range(len(rows)))
        )

        new_heading = f"{fields_label} ({_fmt_x_latest(x_data[-1])})" if x_data else no_update

        figs = []
        for group in groups:
            fig = go.Figure()
            for y_field in group:
                y_data = [r.get(y_field) for r in row_dicts]
                cidx = field_color[y_field]
                color = colors[cidx] if cidx < len(colors) else None
                marker = {"color": color} if color else {}

                if plot_type == "scatter":
                    trace = go.Scatter(x=x_data, y=y_data, mode="markers",
                                       name=y_field, marker=marker)
                elif plot_type == "bar":
                    trace = go.Bar(x=x_data, y=y_data, name=y_field,
                                   marker=marker)
                else:
                    trace = go.Scatter(x=x_data, y=y_data, mode="lines",
                                       name=y_field, line=marker)
                fig.add_trace(trace)

            _layout_kw: dict[str, Any] = {
                "title": ", ".join(group) if len(groups) > 1 else "",
                "height": 280,
                "xaxis_title": x_field or "",
                "yaxis_title": group[0] if len(group) == 1 else "",
                "margin": {"t": 40, "b": 40, "l": 60, "r": 20},
            }
            if font_dict:
                _layout_kw["font"] = font_dict
            fig.update_layout(**_layout_kw)
            figs.append(fig)

        poll_n += 1
        now = _dt.datetime.now().strftime("%H:%M:%S")
        status = f"Poll #{poll_n}  ·  {len(rows)} row(s)  ·  updated {now}"
        if verbose:
            print(status)

        return figs + [status, new_heading, poll_n]

    url = f"http://{host}:{port}"
    print(f"Dash server at {url}  (Ctrl+C to stop)")
    webbrowser.open(url)
    app.run(host=host, port=port, debug=False, use_reloader=False)


# ── multi-source backends (two tables, shared x-axis) ─────────────────────


def _poll_matplotlib_multi(
    con: duckdb.DuckDBPyConnection,
    sql1: str,
    sql2: str,
    x_field: str | None,
    y_fields1: list[str],
    y_fields2: list[str],
    opts: dict[str, Any],
    interval: float,
    count: int | None,
    verbose: bool,
) -> None:
    """Live matplotlib poll: y_fields1 in subplot 1, y_fields2 in subplot 2, shared x."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for --plot matplotlib.  pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    figsize = tuple(opts["figsize"])
    colors = opts["colors"] or []
    plot_type = opts["type"]

    if opts["font"]:
        plt.rcParams["font.family"] = opts["font"]
    if opts["fontsize"]:
        plt.rcParams["font.size"] = opts["fontsize"]

    label1 = ", ".join(y_fields1)
    label2 = ", ".join(y_fields2)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    suptitle_obj = fig.suptitle(f"{label1}  |  {label2}")

    poll_n = 0
    try:
        while count is None or poll_n < count:
            cols1, rows1 = _fetch_poll(con, sql1)
            cols2, rows2 = _fetch_poll(con, sql2)
            if verbose:
                print(f"[poll {poll_n + 1}] src1={len(rows1)} row(s), src2={len(rows2)} row(s)")

            rd1 = [dict(zip(cols1, r, strict=False)) for r in rows1]
            rd2 = [dict(zip(cols2, r, strict=False)) for r in rows2]

            x1 = (
                [r[x_field] for r in rd1]
                if x_field and rd1 and x_field in rd1[0]
                else list(range(len(rows1)))
            )
            x2 = (
                [r[x_field] for r in rd2]
                if x_field and rd2 and x_field in rd2[0]
                else list(range(len(rows2)))
            )

            ax1.cla()
            ax2.cla()

            for i, yf in enumerate(y_fields1):
                y = [r.get(yf) for r in rd1]
                color = colors[i] if i < len(colors) else None
                kw: dict[str, Any] = {"label": yf}
                if color:
                    kw["color"] = color
                if plot_type == "scatter":
                    ax1.scatter(x1, y, s=10, **kw)
                elif plot_type == "bar":
                    ax1.bar(x1, y, **kw)
                else:
                    ax1.plot(x1, y, **kw)
            if len(y_fields1) == 1:
                ax1.set_ylabel(label1)
            else:
                ax1.legend(loc="upper left", fontsize="small")
            ax1.grid(True, linestyle="--", alpha=0.5)

            n_colors1 = len(y_fields1)
            for j, yf in enumerate(y_fields2):
                cidx = n_colors1 + j
                y = [r.get(yf) for r in rd2]
                color = colors[cidx] if cidx < len(colors) else None
                kw = {"label": yf}
                if color:
                    kw["color"] = color
                if plot_type == "scatter":
                    ax2.scatter(x2, y, s=10, **kw)
                elif plot_type == "bar":
                    ax2.bar(x2, y, **kw)
                else:
                    ax2.plot(x2, y, **kw)
            if len(y_fields2) == 1:
                ax2.set_ylabel(label2)
            else:
                ax2.legend(loc="upper left", fontsize="small")
            ax2.grid(True, linestyle="--", alpha=0.5)

            if x_field:
                ax2.set_xlabel(x_field)
                if x1:
                    suptitle_obj.set_text(
                        f"{label1}  |  {label2}  ({_fmt_x_latest(x1[-1])})"
                    )

            _apply_date_format(ax2, x1 or x2)
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.draw()
            plt.pause(interval)
            poll_n += 1

    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.show()


def _poll_plotly_multi(
    con: duckdb.DuckDBPyConnection,
    sql1: str,
    sql2: str,
    x_field: str | None,
    y_fields1: list[str],
    y_fields2: list[str],
    opts: dict[str, Any],
    interval: float,
    count: int | None,
    output_html: str,
    verbose: bool,
) -> None:
    """Two-source plotly polling with two rows sharing one x-axis."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Error: plotly is required for --plot plotly.  pip install plotly", file=sys.stderr)
        sys.exit(1)

    colors = opts["colors"] or []
    plot_type = opts["type"]
    html_path = Path(output_html).resolve()
    browser_opened = False
    label1 = ", ".join(y_fields1)
    label2 = ", ".join(y_fields2)
    title = f"{label1}  |  {label2}"
    poll_n = 0

    font_dict: dict[str, Any] = {}
    if opts["font"]:
        font_dict["family"] = opts["font"]
    if opts["fontsize"]:
        font_dict["size"] = opts["fontsize"]

    try:
        while count is None or poll_n < count:
            cols1, rows1 = _fetch_poll(con, sql1)
            cols2, rows2 = _fetch_poll(con, sql2)
            if verbose:
                print(f"[poll {poll_n + 1}] src1={len(rows1)} row(s), src2={len(rows2)} row(s)")

            rd1 = [dict(zip(cols1, r, strict=False)) for r in rows1]
            rd2 = [dict(zip(cols2, r, strict=False)) for r in rows2]

            x1 = (
                [r[x_field] for r in rd1]
                if x_field and rd1 and x_field in rd1[0]
                else list(range(len(rows1)))
            )
            x2 = (
                [r[x_field] for r in rd2]
                if x_field and rd2 and x_field in rd2[0]
                else list(range(len(rows2)))
            )

            if x1:
                title = f"{label1}  |  {label2}  ({_fmt_x_latest(x1[-1])})"

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                subplot_titles=[label1, label2],
            )

            for i, yf in enumerate(y_fields1):
                y = [r.get(yf) for r in rd1]
                color = colors[i] if i < len(colors) else None
                marker = {"color": color} if color else {}
                if plot_type == "scatter":
                    trace = go.Scatter(x=x1, y=y, mode="markers", name=yf, marker=marker)
                elif plot_type == "bar":
                    trace = go.Bar(x=x1, y=y, name=yf, marker=marker)
                else:
                    trace = go.Scatter(x=x1, y=y, mode="lines", name=yf, line=marker)
                fig.add_trace(trace, row=1, col=1)

            n_colors1 = len(y_fields1)
            for j, yf in enumerate(y_fields2):
                cidx = n_colors1 + j
                y = [r.get(yf) for r in rd2]
                color = colors[cidx] if cidx < len(colors) else None
                marker = {"color": color} if color else {}
                if plot_type == "scatter":
                    trace = go.Scatter(x=x2, y=y, mode="markers", name=yf, marker=marker)
                elif plot_type == "bar":
                    trace = go.Bar(x=x2, y=y, name=yf, marker=marker)
                else:
                    trace = go.Scatter(x=x2, y=y, mode="lines", name=yf, line=marker)
                fig.add_trace(trace, row=2, col=1)

            if x_field:
                fig.update_xaxes(title_text=x_field, row=2, col=1)
            fig.update_yaxes(title_text=label1 if len(y_fields1) == 1 else "", row=1, col=1)
            fig.update_yaxes(title_text=label2 if len(y_fields2) == 1 else "", row=2, col=1)
            _layout_kw: dict[str, Any] = {"title": title, "height": 560}
            if font_dict:
                _layout_kw["font"] = font_dict
            fig.update_layout(**_layout_kw)

            html_body = fig.to_html(full_html=True, include_plotlyjs="cdn")
            refresh_tag = f'<meta http-equiv="refresh" content="{int(interval)}">'
            html_body = html_body.replace("<head>", f"<head>\n  {refresh_tag}", 1)
            html_path.write_text(html_body, encoding="utf-8")

            if not browser_opened:
                webbrowser.open(html_path.as_uri())
                browser_opened = True

            poll_n += 1
            if count is None or poll_n < count:
                time.sleep(interval)

    except KeyboardInterrupt:
        pass

    print(f"Plotly output written to {html_path}")


def _poll_dash_multi(
    con: duckdb.DuckDBPyConnection,
    sql1: str,
    sql2: str,
    x_field: str | None,
    y_fields1: list[str],
    y_fields2: list[str],
    opts: dict[str, Any],
    interval: float,
    count: int | None,
    host: str,
    port: int,
    verbose: bool,
) -> None:
    """Two-source Dash polling: two rows in one make_subplots figure with shared x-axis."""
    try:
        import plotly.graph_objects as go
        from dash import Dash, Input, Output, State, dcc, html, no_update
        from dash.exceptions import PreventUpdate
        from plotly.subplots import make_subplots
    except ImportError:
        print("Error: dash is required for --plot dash.  pip install dash", file=sys.stderr)
        sys.exit(1)

    import datetime as _dt
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    colors = opts["colors"] or []
    plot_type = opts["type"]
    interval_ms = int(interval * 1000)
    label1 = ", ".join(y_fields1)
    label2 = ", ".join(y_fields2)
    heading_base = f"{label1}  |  {label2}"

    font_dict: dict[str, Any] = {}
    if opts["font"]:
        font_dict["family"] = opts["font"]
    if opts["fontsize"]:
        font_dict["size"] = opts["fontsize"]

    app = Dash(__name__, title="MySQL Poll (dual)", suppress_callback_exceptions=True)

    app.layout = html.Div(
        style={"fontFamily": opts["font"] or "sans-serif", "padding": "16px"},
        children=[
            html.H4(id="heading", children=heading_base, style={"marginBottom": "8px"}),
            html.Div(
                style={"display": "flex", "alignItems": "center",
                       "gap": "16px", "marginBottom": "12px"},
                children=[
                    html.Button("⏸ Pause / Resume", id="pause-btn", n_clicks=0),
                    html.Span(id="status-text",
                              style={"color": "gray", "fontSize": "0.85em"}),
                ],
            ),
            dcc.Interval(id="interval", interval=interval_ms,
                         n_intervals=0, disabled=False),
            dcc.Store(id="paused", data=False),
            dcc.Store(id="poll-n", data=0),
            dcc.Graph(id="dual-graph", config={"displayModeBar": True}),
        ],
    )

    @app.callback(
        Output("paused", "data"),
        Output("interval", "disabled"),
        Input("pause-btn", "n_clicks"),
        State("paused", "data"),
        prevent_initial_call=True,
    )
    def toggle_pause(n_clicks, paused):
        new_state = not paused
        return new_state, new_state

    @app.callback(
        Output("dual-graph", "figure"),
        Output("status-text", "children"),
        Output("heading", "children"),
        Output("poll-n", "data"),
        Input("interval", "n_intervals"),
        State("poll-n", "data"),
    )
    def refresh(n_intervals, poll_n):
        if count is not None and poll_n >= count:
            raise PreventUpdate

        try:
            cols1, rows1 = _fetch_poll(con, sql1)
            cols2, rows2 = _fetch_poll(con, sql2)
        except duckdb.Error as exc:
            if verbose:
                print(f"Poll error: {exc}", file=sys.stderr)
            raise PreventUpdate from exc

        rd1 = [dict(zip(cols1, r, strict=False)) for r in rows1]
        rd2 = [dict(zip(cols2, r, strict=False)) for r in rows2]

        x1 = (
            [r[x_field] for r in rd1]
            if x_field and rd1 and x_field in rd1[0]
            else list(range(len(rows1)))
        )
        x2 = (
            [r[x_field] for r in rd2]
            if x_field and rd2 and x_field in rd2[0]
            else list(range(len(rows2)))
        )

        new_heading = no_update
        if x1:
            new_heading = f"{heading_base}  ({_fmt_x_latest(x1[-1])})"

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=[label1, label2],
        )

        for i, yf in enumerate(y_fields1):
            y = [r.get(yf) for r in rd1]
            color = colors[i] if i < len(colors) else None
            marker = {"color": color} if color else {}
            if plot_type == "scatter":
                trace = go.Scatter(x=x1, y=y, mode="markers", name=yf, marker=marker)
            elif plot_type == "bar":
                trace = go.Bar(x=x1, y=y, name=yf, marker=marker)
            else:
                trace = go.Scatter(x=x1, y=y, mode="lines", name=yf, line=marker)
            fig.add_trace(trace, row=1, col=1)

        n_colors1 = len(y_fields1)
        for j, yf in enumerate(y_fields2):
            cidx = n_colors1 + j
            y = [r.get(yf) for r in rd2]
            color = colors[cidx] if cidx < len(colors) else None
            marker = {"color": color} if color else {}
            if plot_type == "scatter":
                trace = go.Scatter(x=x2, y=y, mode="markers", name=yf, marker=marker)
            elif plot_type == "bar":
                trace = go.Bar(x=x2, y=y, name=yf, marker=marker)
            else:
                trace = go.Scatter(x=x2, y=y, mode="lines", name=yf, line=marker)
            fig.add_trace(trace, row=2, col=1)

        if x_field:
            fig.update_xaxes(title_text=x_field, row=2, col=1)
        fig.update_yaxes(title_text=label1 if len(y_fields1) == 1 else "", row=1, col=1)
        fig.update_yaxes(title_text=label2 if len(y_fields2) == 1 else "", row=2, col=1)
        _layout_kw: dict[str, Any] = {"height": 560}
        if font_dict:
            _layout_kw["font"] = font_dict
        fig.update_layout(**_layout_kw)

        poll_n += 1
        now = _dt.datetime.now().strftime("%H:%M:%S")
        status = (
            f"Poll #{poll_n}  ·  src1: {len(rows1)} row(s)  ·  "
            f"src2: {len(rows2)} row(s)  ·  updated {now}"
        )
        if verbose:
            print(status)

        return fig, status, new_heading, poll_n

    url = f"http://{host}:{port}"
    print(f"Dash server at {url}  (Ctrl+C to stop)")
    webbrowser.open(url)
    app.run(host=host, port=port, debug=False, use_reloader=False)


# ── mode entry point ────────────────────────────────────────────────────────

def mode_poll(args: argparse.Namespace) -> None:
    """Poll a MySQL table and display selected fields as a live chart."""
    import datetime as _dt

    opts = _parse_plot_options(args.plot_options)
    dsn = _build_dsn(args.host, args.port, args.user, args.password, args.database)

    con = duckdb.connect()
    _load_mysql_extension(con, args.verbose)
    _attach_mysql(con, dsn, args.verbose)

    if args.list_tables:
        tables = list_mysql_tables(con, args.database)
        print(f"Tables in '{args.database}' ({len(tables)}):")
        for t in tables:
            print(f"  {t}")
        return

    # These are needed for second-source resolution in the table path.
    limit: int = 0
    start_time: str | None = None

    if args.query:
        # ── raw-query path ──────────────────────────────────────────────────
        sql = args.query

        # Run the query once to discover result-column names and types.
        rel = con.execute(sql)
        result_cols = [(desc[0], desc[1]) for desc in rel.description]
        result_col_names = [c[0] for c in result_cols]

        if args.list_fields:
            print("Columns returned by query:")
            for name, col_type in result_cols:
                if _is_numeric_type(col_type):
                    marker = " *"
                elif _is_timestamp_type(col_type):
                    marker = " @"
                else:
                    marker = ""
                print(f"  {name:<30}  {col_type}{marker}")
            print("(* = numeric / y-axis  @ = timestamp / x-axis)")
            return

        # Resolve x-axis: explicit flag → first TIMESTAMP column → None
        x_field = args.x_field or _auto_x_field(result_cols)
        if x_field and x_field not in result_col_names:
            print(f"Error: x-field '{x_field}' not in query result columns.", file=sys.stderr)
            sys.exit(1)
        if x_field and not args.x_field and args.verbose:
            print(f"Auto-selected x-axis: {x_field}")

        requested = args.fields or []
        if requested:
            unknown = set(requested) - set(result_col_names)
            if unknown:
                print(f"Error: unknown column(s) in query result: {', '.join(sorted(unknown))}", file=sys.stderr)
                sys.exit(1)
            y_fields = [f for f in requested if f != x_field]
        else:
            numeric_names = [c[0] for c in result_cols if _is_numeric_type(c[1]) and not _is_id_field(c[0])]
            y_fields = [f for f in numeric_names if f != x_field]
            if not y_fields:
                print("Error: no numeric columns in query result. Use --fields to specify columns explicitly.", file=sys.stderr)
                sys.exit(1)
            if args.verbose:
                print(f"Auto-selected y-fields: {', '.join(y_fields)}")

    else:
        # ── single-table path ───────────────────────────────────────────────
        all_cols_raw = describe_table(con, args.table)
        all_col_names = [c[0] for c in all_cols_raw]
        all_cols_typed = [(c[0], c[1]) for c in all_cols_raw]
        numeric_cols = list_numeric_columns(con, args.table)

        if args.list_fields:
            print(f"Columns in '{args.table}':")
            for name, col_type in all_cols_typed:
                if _is_numeric_type(col_type):
                    marker = " *"
                elif _is_timestamp_type(col_type):
                    marker = " @"
                else:
                    marker = ""
                print(f"  {name:<30}  {col_type}{marker}")
            print("(* = numeric / y-axis  @ = timestamp / x-axis)")
            return

        # Resolve x-axis: explicit flag → first TIMESTAMP column → None
        x_field = args.x_field or _auto_x_field(all_cols_typed)
        if x_field and not args.x_field and args.verbose:
            print(f"Auto-selected x-axis: {x_field}")

        requested = args.fields or []
        if requested:
            unknown = set(requested) - set(all_col_names)
            if unknown:
                print(f"Error: unknown column(s) in {args.table}: {', '.join(sorted(unknown))}", file=sys.stderr)
                sys.exit(1)
            fields = [f for f in requested if f != x_field]
        else:
            fields = [name for name, _ in numeric_cols if name != x_field and not _is_id_field(name)]
            if not fields:
                print(f"Error: no numeric columns found in '{args.table}'. Use --fields to specify columns explicitly.", file=sys.stderr)
                sys.exit(1)
            if args.verbose:
                print(f"Auto-selected y-fields: {', '.join(fields)}")

        select_fields = [x_field] + fields if x_field else fields

        y_fields = fields

        if not y_fields:
            print("Error: no y-fields to plot.", file=sys.stderr)
            sys.exit(1)

        # When a time axis is available: filter to data from launch time onwards so
        # the plot starts "now" rather than at the beginning of the table.
        # Default to no row cap in this mode so rows accumulate as they arrive.
        if x_field and args.limit is None:
            start_time = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            limit = 0
        else:
            start_time = None
            limit = args.limit if args.limit is not None else 200
        sql = _build_poll_query(args.table, select_fields, args.where, limit, x_field, start_time=start_time)

    # ── resolve second source (--table2 / --query2) ─────────────────────────
    sql2: str | None = None
    y_fields2: list[str] = []

    has_table2 = bool(getattr(args, "table2", None))
    has_query2 = bool(getattr(args, "query2", None))

    if has_query2:
        sql2 = args.query2
        assert sql2 is not None
        rel2 = con.execute(sql2)
        result_cols2 = [(desc[0], desc[1]) for desc in rel2.description]
        requested2 = getattr(args, "fields2", None) or []
        if requested2:
            y_fields2 = [f for f in requested2 if f != x_field]
        else:
            y_fields2 = [
                c[0] for c in result_cols2
                if _is_numeric_type(c[1]) and not _is_id_field(c[0]) and c[0] != x_field
            ]
        if not y_fields2:
            print("Error: no y-fields for second source. Use --fields2.", file=sys.stderr)
            sys.exit(1)
        if args.verbose:
            print(f"Second source (query): {sql2}  fields: {', '.join(y_fields2)}")

    elif has_table2:
        all_cols2_raw = describe_table(con, args.table2)
        all_cols2_typed = [(c[0], c[1]) for c in all_cols2_raw]
        requested2 = getattr(args, "fields2", None) or []
        if requested2:
            f2 = [f for f in requested2 if f != x_field]
        else:
            f2 = [
                name for name, t in all_cols2_typed
                if _is_numeric_type(t) and not _is_id_field(name) and name != x_field
            ]
        if not f2:
            print(f"Error: no y-fields for '{args.table2}'. Use --fields2.", file=sys.stderr)
            sys.exit(1)
        y_fields2 = f2
        sel2 = ([x_field] + f2) if x_field else f2
        where2 = getattr(args, "where2", None)
        sql2 = _build_poll_query(args.table2, sel2, where2, limit, x_field, start_time=start_time)
        if args.verbose:
            print(f"Second source (table): {sql2}  fields: {', '.join(y_fields2)}")

    if args.verbose:
        print(f"Query: {sql}")

    count = args.count if args.count and args.count > 0 else None

    if args.plot == "table":
        if sql2:
            print(
                "Warning: --table2/--query2 is not supported with --plot table; "
                "ignoring second source.",
                file=sys.stderr,
            )
        _poll_table(
            con, sql, x_field, y_fields, opts,
            args.interval, count, args.verbose,
        )
    elif args.plot == "plotly":
        if sql2:
            _poll_plotly_multi(
                con, sql, sql2, x_field, y_fields, y_fields2, opts,
                args.interval, count, args.output_html, args.verbose,
            )
        else:
            _poll_plotly(
                con, sql, x_field, y_fields, opts,
                args.interval, count, args.output_html, args.verbose,
            )
    elif args.plot == "textual":
        if sql2:
            print(
                "Warning: --table2/--query2 is not supported with --plot textual; "
                "ignoring second source.",
                file=sys.stderr,
            )
        _poll_textual(
            con, sql, x_field, y_fields, opts,
            args.interval, count, args.verbose,
        )
    elif args.plot == "dash":
        if sql2:
            _poll_dash_multi(
                con, sql, sql2, x_field, y_fields, y_fields2, opts,
                args.interval, count, args.dash_host, args.dash_port, args.verbose,
            )
        else:
            _poll_dash(
                con, sql, x_field, y_fields, opts,
                args.interval, count, args.dash_host, args.dash_port, args.verbose,
            )
    else:
        if sql2:
            _poll_matplotlib_multi(
                con, sql, sql2, x_field, y_fields, y_fields2, opts,
                args.interval, count, args.verbose,
            )
        else:
            _poll_matplotlib(
                con, sql, x_field, y_fields, opts,
                args.interval, count, args.verbose,
            )


def mode_plot(args: argparse.Namespace) -> None:
    """Fetch MySQL data once and display selected fields as a static chart."""
    opts = _parse_plot_options(args.plot_options)

    if args.plot == "textual":
        print("Error: --plot textual is not supported in plot mode.", file=sys.stderr)
        sys.exit(1)

    dsn = _build_dsn(args.host, args.port, args.user, args.password, args.database)

    con = duckdb.connect()
    _load_mysql_extension(con, args.verbose)
    _attach_mysql(con, dsn, args.verbose)

    if args.list_tables:
        tables = list_mysql_tables(con, args.database)
        print(f"Tables in '{args.database}' ({len(tables)}):")
        for t in tables:
            print(f"  {t}")
        return

    if args.query:
        # ── raw-query path ──────────────────────────────────────────────────
        sql = args.query

        rel = con.execute(sql)
        result_cols = [(desc[0], desc[1]) for desc in rel.description]
        result_col_names = [c[0] for c in result_cols]

        if args.list_fields:
            print("Columns returned by query:")
            for name, col_type in result_cols:
                if _is_numeric_type(col_type):
                    marker = " *"
                elif _is_timestamp_type(col_type):
                    marker = " @"
                else:
                    marker = ""
                print(f"  {name:<30}  {col_type}{marker}")
            print("(* = numeric / y-axis  @ = timestamp / x-axis)")
            return

        x_field = args.x_field or _auto_x_field(result_cols)
        if x_field and x_field not in result_col_names:
            print(f"Error: x-field '{x_field}' not in query result columns.", file=sys.stderr)
            sys.exit(1)
        if x_field and not args.x_field and args.verbose:
            print(f"Auto-selected x-axis: {x_field}")

        requested = args.fields or []
        if requested:
            unknown = set(requested) - set(result_col_names)
            if unknown:
                print(f"Error: unknown column(s) in query result: {', '.join(sorted(unknown))}", file=sys.stderr)
                sys.exit(1)
            y_fields = [f for f in requested if f != x_field]
        else:
            numeric_names = [c[0] for c in result_cols if _is_numeric_type(c[1]) and not _is_id_field(c[0])]
            y_fields = [f for f in numeric_names if f != x_field]
            if not y_fields:
                print("Error: no numeric columns in query result. Use --fields to specify columns explicitly.", file=sys.stderr)
                sys.exit(1)
            if args.verbose:
                print(f"Auto-selected y-fields: {', '.join(y_fields)}")

    else:
        # ── single-table path ───────────────────────────────────────────────
        all_cols_raw = describe_table(con, args.table)
        all_col_names = [c[0] for c in all_cols_raw]
        all_cols_typed = [(c[0], c[1]) for c in all_cols_raw]
        numeric_cols = list_numeric_columns(con, args.table)

        if args.list_fields:
            print(f"Columns in '{args.table}':")
            for name, col_type in all_cols_typed:
                if _is_numeric_type(col_type):
                    marker = " *"
                elif _is_timestamp_type(col_type):
                    marker = " @"
                else:
                    marker = ""
                print(f"  {name:<30}  {col_type}{marker}")
            print("(* = numeric / y-axis  @ = timestamp / x-axis)")
            return

        x_field = args.x_field or _auto_x_field(all_cols_typed)
        if x_field and not args.x_field and args.verbose:
            print(f"Auto-selected x-axis: {x_field}")

        requested = args.fields or []
        if requested:
            unknown = set(requested) - set(all_col_names)
            if unknown:
                print(f"Error: unknown column(s) in {args.table}: {', '.join(sorted(unknown))}", file=sys.stderr)
                sys.exit(1)
            fields = [f for f in requested if f != x_field]
        else:
            fields = [name for name, _ in numeric_cols if name != x_field and not _is_id_field(name)]
            if not fields:
                print(f"Error: no numeric columns found in '{args.table}'. Use --fields to specify columns explicitly.", file=sys.stderr)
                sys.exit(1)
            if args.verbose:
                print(f"Auto-selected y-fields: {', '.join(fields)}")

        select_fields = [x_field] + fields if x_field else fields

        y_fields = fields

        if not y_fields:
            print("Error: no y-fields to plot.", file=sys.stderr)
            sys.exit(1)

        limit = args.limit if args.limit is not None else 0
        sql = _build_poll_query(args.table, select_fields, args.where, limit, x_field)

    if args.verbose:
        print(f"Query: {sql}")

    if args.plot == "plotly":
        _plot_static_plotly(
            con, sql, x_field, y_fields, opts,
            args.output_html, args.verbose,
        )
    elif args.plot == "dash":
        _plot_static_dash(
            con, sql, x_field, y_fields, opts,
            args.dash_host, args.dash_port, args.verbose,
        )
    else:
        _plot_static_matplotlib(
            con, sql, x_field, y_fields, opts, args.verbose,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments.

    Returns
    -------
    argparse.Namespace
        Validated arguments with connection parameters resolved from
        CLI flags → env vars → built-in defaults (highest priority first).
    """
    parser = argparse.ArgumentParser(
        description="Connect DuckDB to a remote MySQL server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Parameters can also be set via environment variables (e.g. direnv):\n"
            "  MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB\n"
            "  MYSQL_OUTPUT, MYSQL_OUTPUT_DIR\n\n"
            "Examples:\n"
            "  %(prog)s --mode live --host db.example.com --user admin"
            " --password secret --database magnetdb\n"
            "  %(prog)s --mode export --format csv --output-dir ./out"
            " --host db.example.com --user admin --password secret"
            " --database magnetdb\n"
            "  %(prog)s --mode poll --table measurements --fields t Icoil Ucoil"
            " --x-field t --interval 5 --plot matplotlib"
            " --host db.example.com --user admin --password secret"
            " --database magnetdb\n"
            "  %(prog)s --mode view --table measurements --limit 50"
            " --host db.example.com --user admin --password secret"
            " --database magnetdb\n"
        ),
    )

    parser.add_argument(
        "--mode",
        choices=["live", "export", "poll", "plot", "view"],
        default="live",
        help=(
            "live: print schema (default); export: copy tables; "
            "poll: live chart; plot: one-shot static chart; view: tabular table display"
        ),
    )
    parser.add_argument(
        "--list-databases",
        action="store_true",
        dest="list_databases",
        help="list all databases on MYSQL_HOST and exit (--database is not required)",
    )

    # Connection parameters
    conn = parser.add_argument_group("MySQL connection")
    conn.add_argument(
        "--host",
        default=_env("MYSQL_HOST", _DEFAULT_HOST),
        metavar="HOST",
        help=f"MySQL host (env: MYSQL_HOST, default: {_DEFAULT_HOST})",
    )
    conn.add_argument(
        "--port",
        type=int,
        default=int(_env("MYSQL_PORT", str(_DEFAULT_PORT)) or str(_DEFAULT_PORT)),
        metavar="PORT",
        help=f"MySQL port (env: MYSQL_PORT, default: {_DEFAULT_PORT})",
    )
    conn.add_argument(
        "--user",
        default=_env("MYSQL_USER"),
        metavar="USER",
        help="MySQL user (env: MYSQL_USER, required)",
    )
    conn.add_argument(
        "--password",
        default=_env("MYSQL_PASSWORD"),
        metavar="PASSWORD",
        help="MySQL password (env: MYSQL_PASSWORD, required)",
    )
    conn.add_argument(
        "--database",
        default=_env("MYSQL_DB"),
        metavar="DATABASE",
        help="MySQL database name (env: MYSQL_DB, required)",
    )

    # Export options
    exp = parser.add_argument_group("export options (--mode export only)")
    exp.add_argument(
        "--format",
        choices=["csv", "parquet", "duckdb", "excel"],
        default="csv",
        dest="fmt",
        metavar="{csv,parquet,duckdb,excel}",
        help="output format (default: csv)",
    )
    exp.add_argument(
        "--output",
        default=_env("MYSQL_OUTPUT", _DEFAULT_OUTPUT),
        metavar="FILE",
        help=f"DuckDB output file for --format duckdb "
             f"(env: MYSQL_OUTPUT, default: {_DEFAULT_OUTPUT})",
    )
    exp.add_argument(
        "--output-dir",
        default=_env("MYSQL_OUTPUT_DIR", _DEFAULT_OUTPUT_DIR),
        metavar="DIR",
        help="output directory for CSV/Parquet files "
             "(env: MYSQL_OUTPUT_DIR, default: current dir)",
    )
    exp.add_argument(
        "--tables",
        nargs="+",
        metavar="TABLE",
        help="subset of tables to export (default: all)",
    )
    exp.add_argument(
        "--export-fields",
        nargs="+",
        metavar="COL",
        dest="export_fields",
        help="columns to include in the export; requires exactly one table in --tables",
    )
    exp.add_argument(
        "--time-field",
        metavar="COL",
        dest="time_field",
        help=(
            "TIMESTAMP column used for --start/--end filtering "
            "(auto-detected from schema if omitted)"
        ),
    )
    exp.add_argument(
        "--start",
        metavar="DATETIME",
        help="start of time range, ISO 8601, e.g. '2024-01-15 08:00:00'",
    )
    exp.add_argument(
        "--end",
        metavar="DATETIME",
        help="end of time range, ISO 8601, e.g. '2024-01-15 20:00:00'",
    )

    # Poll / plot options
    poll = parser.add_argument_group("poll/plot options (--mode poll or --mode plot)")
    poll.add_argument(
        "--table",
        metavar="TABLE",
        help="MySQL table to poll. Mutually exclusive with --query.",
    )
    poll.add_argument(
        "--query",
        metavar="SQL",
        help=(
            "Raw SELECT query to run instead of --table/--fields. "
            "Must reference tables as mysqldb.<table>. "
            "Use --fields to choose which result columns to plot, "
            "--x-field for the x-axis column. "
            "Example: \"SELECT t, m.Icoil, s.temp "
            "FROM mysqldb.meas m JOIN mysqldb.sensors s ON m.id=s.mid "
            "ORDER BY t LIMIT 500\""
        ),
    )
    poll.add_argument(
        "--list-tables",
        action="store_true",
        dest="list_tables",
        help="print all tables in the database and exit, no polling",
    )
    poll.add_argument(
        "--list-fields",
        action="store_true",
        dest="list_fields",
        help="print numeric (plottable) columns of --table and exit, no polling",
    )
    poll.add_argument(
        "--fields",
        nargs="+",
        metavar="COL",
        help="columns to extract and plot (default: all columns)",
    )
    poll.add_argument(
        "--x-field",
        metavar="COL",
        help="column to use as x-axis (default: row index)",
    )
    poll.add_argument(
        "--where",
        metavar="EXPR",
        help="SQL WHERE clause filter (e.g. \"status='active'\")",
    )
    poll.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="max rows fetched (poll default: 200; plot default: 0 = all rows; 0 = no cap)",
    )
    poll.add_argument(
        "--interval",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="seconds between polls (default: 5)",
    )
    poll.add_argument(
        "--count",
        type=int,
        default=0,
        metavar="N",
        help="number of polls to run (default: 0 = run until Ctrl+C)",
    )
    poll.add_argument(
        "--plot",
        choices=["table", "matplotlib", "plotly", "textual", "dash"],
        default="matplotlib",
        help=(
            "poll backend: matplotlib (default, live chart), "
            "table (rich terminal table), plotly (auto-refresh HTML), "
            "textual (TUI sparklines), dash (web app)"
        ),
    )
    poll.add_argument(
        "--table2",
        metavar="TABLE",
        dest="table2",
        help=(
            "Second MySQL table to poll simultaneously. "
            "Its fields are shown in a separate subplot sharing the same x-axis. "
            "Mutually exclusive with --query2. "
            "Supported backends: matplotlib, plotly, dash."
        ),
    )
    poll.add_argument(
        "--query2",
        metavar="SQL",
        dest="query2",
        help=(
            "Raw SELECT query for the second subplot. "
            "Must reference tables as mysqldb.<table>. "
            "Mutually exclusive with --table2."
        ),
    )
    poll.add_argument(
        "--fields2",
        nargs="+",
        metavar="COL",
        dest="fields2",
        help="Columns to plot from the second table/query (default: all numeric).",
    )
    poll.add_argument(
        "--where2",
        metavar="EXPR",
        dest="where2",
        help="SQL WHERE clause for the second table (--table2 only).",
    )
    poll.add_argument(
        "--plot-options",
        metavar="JSON",
        dest="plot_options",
        help=(
            'JSON object with plot style options. '
            'Keys: '
            'type (line|scatter|bar, default line); '
            'layout (subplots|overlay|groups, default subplots); '
            'groups (list of field-name lists, used when layout=groups, '
            'e.g. [["Icoil","Ucoil"],["tsb","teb"]]); '
            'figsize ([w,h] inches, matplotlib only, default [12,6]); '
            'colors (list of color strings, one per y-field); '
            'font (font family, e.g. "Arial"); '
            'fontsize (base font size in points, e.g. 12). '
            'Example: \'{"layout":"groups","groups":[["Icoil","Ucoil"],["tsb"]],'
            '"font":"Arial","fontsize":11}\''
        ),
    )
    poll.add_argument(
        "--output-html",
        metavar="FILE",
        default="poll_output.html",
        dest="output_html",
        help="HTML output path for --plot plotly (default: poll_output.html)",
    )
    poll.add_argument(
        "--dash-host",
        metavar="HOST",
        default="127.0.0.1",
        dest="dash_host",
        help="host for the Dash web server (default: 127.0.0.1)",
    )
    poll.add_argument(
        "--dash-port",
        type=int,
        metavar="PORT",
        default=8050,
        dest="dash_port",
        help="port for the Dash web server (default: 8050)",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")

    args = parser.parse_args()

    required_params = [
        ("--user / MYSQL_USER", args.user),
        ("--password / MYSQL_PASSWORD", args.password),
    ]
    if not args.list_databases:
        required_params.append(("--database / MYSQL_DB", args.database))

    missing = [name for name, val in required_params if not val]
    if missing:
        parser.error(
            "Missing required connection parameter(s): "
            + ", ".join(missing)
        )

    if args.list_databases and not args.database:
        args.database = "information_schema"

    if args.mode == "export" and args.tables and getattr(args, "table", None):
        parser.error("--table and --tables are mutually exclusive in export mode")

    if args.mode in ("poll", "plot", "view") and args.table and args.query:
        parser.error("--table and --query are mutually exclusive")

    if args.mode == "poll":
        if not args.table and not args.query and not args.list_tables:
            parser.error("--mode poll requires --table TABLE or --query SQL (or --list-tables)")
        table2 = getattr(args, "table2", None)
        query2 = getattr(args, "query2", None)
        if table2 and query2:
            parser.error("--table2 and --query2 are mutually exclusive")
        if (table2 or query2) and not (args.table or args.query):
            parser.error("--table2/--query2 require --table or --query for the first source")

    if args.mode == "plot" and not args.table and not args.query and not args.list_tables:
        parser.error("--mode plot requires --table TABLE or --query SQL (or --list-tables)")

    if args.mode == "view" and not args.table and not args.query:
        parser.error("--mode view requires --table TABLE or --query SQL")

    return args


def main() -> None:
    args = parse_args()
    try:
        if args.list_databases:
            dsn = _build_dsn(args.host, args.port, args.user, args.password, args.database)
            con = duckdb.connect()
            _load_mysql_extension(con, args.verbose)
            _attach_mysql(con, dsn, args.verbose)
            databases = list_mysql_databases(con)
            print(f"Databases on {args.host}:{args.port} ({len(databases)}):")
            for db in databases:
                print(f"  {db}")
            return
        if args.mode == "live":
            mode_live(args)
        elif args.mode == "export":
            mode_export(args)
        elif args.mode == "view":
            mode_view(args)
        elif args.mode == "plot":
            mode_plot(args)
        else:
            mode_poll(args)
    except duckdb.Error as e:
        print(f"DuckDB error: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"I/O error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
