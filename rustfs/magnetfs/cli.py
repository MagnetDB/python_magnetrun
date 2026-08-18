import argparse
import os

from .client import get_s3_client, list_bucket_files
from .conversion import list_local_files, run_conversion
from .plotting import run_plot
from .storage import read_parquet


def cmd_files(args):
    exts = (
        {e if e.startswith(".") else f".{e}" for e in args.ext} if args.ext else None
    )
    list_local_files(args.datadir, exts)


def cmd_convert(args):
    source_path = os.path.join(args.datadir, args.source_txt)
    run_conversion(source_path)


def cmd_list(args):
    s3 = get_s3_client()
    list_bucket_files(s3)


def cmd_read(args):
    read_parquet(args.target_file)


def cmd_plot(args):
    run_plot(args.filename, args.x, args.y, args.keys, args.output)


def main():
    parser = argparse.ArgumentParser(
        description="magnetfs — manage magnetrun data in RustFS S3 storage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datadir",
        default="/mnt/sshfs_data",
        help="Local data directory",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # files: list local data files
    files_parser = subparsers.add_parser("files", help="List local data files")
    files_parser.add_argument(
        "--ext", nargs="+", metavar="EXT", help="Filter by extension(s), e.g. txt csv"
    )
    files_parser.set_defaults(func=cmd_files)

    # convert: convert .txt → Parquet and upload
    convert_parser = subparsers.add_parser(
        "convert", help="Convert a .txt file to Parquet and upload to RustFS"
    )
    convert_parser.add_argument(
        "source_txt", help="Filename to convert (relative to --datadir)"
    )
    convert_parser.set_defaults(func=cmd_convert)

    # list: list Parquet files in bucket
    list_parser = subparsers.add_parser("list", help="List Parquet files in the bucket")
    list_parser.set_defaults(func=cmd_list)

    # read: preview a Parquet file from bucket
    read_parser = subparsers.add_parser(
        "read", help="Download and preview a Parquet file from the bucket"
    )
    read_parser.add_argument("target_file", help="Parquet file key in the bucket")
    read_parser.set_defaults(func=cmd_read)

    # plot: plot columns from a Parquet file
    plot_parser = subparsers.add_parser(
        "plot", help="Plot columns from a Parquet file in the bucket"
    )
    plot_parser.add_argument(
        "filename", nargs="?", help="Parquet file key in the bucket (interactive if omitted)"
    )
    plot_parser.add_argument("--x", default=None, help="X-axis column name")
    plot_parser.add_argument("--y", default=None, help="Y-axis column name")
    plot_parser.add_argument(
        "--keys", action="store_true", help="List available columns and exit"
    )
    plot_parser.add_argument(
        "--output", default=None, help="Save plot to PNG file instead of displaying"
    )
    plot_parser.set_defaults(func=cmd_plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
