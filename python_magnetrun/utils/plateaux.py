#! /usr/bin/python3

from ..magnetdata import MagnetData
from ..processing.stats import nplateaus


def tuple_type(strings: str) -> tuple:
    strings = strings.replace("(", "").replace(")", "")
    mapped_str = map(str, strings.split(","))
    return tuple(mapped_str)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_files", help="input txt file (ex. HL31_2018.04.13.txt)", nargs="*"
    )
    parser.add_argument(
        "--difference", help="specify difference", type=float, default=2.0e-2
    )
    parser.add_argument(
        "--min_num_points",
        help="specify minimum number of points",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--xField",
        help="specify xField (name, unit)",
        type=tuple_type,
        default="(t,s)",
    )
    parser.add_argument(
        "--yField",
        help="specify yField (name, unit)",
        type=tuple_type,
        default="(Field,T)",
    )

    parser.add_argument(
        "--show",
        help="display graphs (default save in png format)",
        action="store_true",
    )
    args = parser.parse_args()

    print(f"input_files: {args.input_files}")
    xField = args.xField
    yField = args.yField

    show = args.show
    save = not show
    threshold = args.difference
    num_points_threshold = args.min_num_points

    for name in args.input_files:
        Data = MagnetData.fromtxt(name)
        nplateaus(Data, xField, yField, threshold, num_points_threshold, show, save)


if __name__ == "__main__":
    main()
