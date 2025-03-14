"""
Demonstrator

Get data from User DataBase
Attached exp. Acronym to magnet records
Perform some stats
Try to classify records per PRoposalType and Research Area
"""

import pandas as pd
import os
import sys

from ..MagnetRun import MagnetRun
from ..magnetdata import MagnetData
from ..processing.stats import stats
from ..utils.plateaux import plateaus, nplateaus

import matplotlib
import matplotlib.pyplot as plt

# hack to avoid pandas warning
pd.options.mode.copy_on_write = True


def getTimestamp(file: str, debug: bool = False):
    """
    extract timestamp from file
    """
    # print(f"getTime({file}):", flush=True)
    from datetime import datetime

    filename = ""
    if "/" in file:
        filename = file.split("/")
    res = filename[-1].split("_")
    print(f"getTime({file})={res}", flush=True)

    (site, date_string) = res
    date_string = date_string.replace(".txt", "")
    tformat = "%Y.%m.%d---%H:%M:%S"
    timestamp = datetime.strptime(date_string, tformat)
    if debug:
        print(f"{site}: timestamp={timestamp}")
    return timestamp


def load_record(file: str) -> MagnetData:
    """Load record."""
    # print(f'load_record: {file}')

    filename = os.path.basename(file)
    (housing, timestamp) = filename.split("_")
    site = "blbl"

    mrun = MagnetRun.fromtxt(housing, site, file)
    data = mrun.MagnetData
    if not isinstance(data, MagnetData):
        raise RuntimeError(f"{file}: cannot load data as MagnetData")
    return data


def main():
    """Console script."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csvfile", help="specify csv file")
    parser.add_argument(
        "--mdatadir",
        help="specify datadir for record files with txt extension",
        type=str,
        default="srvdata",
    )
    parser.add_argument(
        "--show",
        help="display graphs (requires X11 server active)",
        action="store_true",
    )
    parser.add_argument("--save", help="save graphs (png format)", action="store_true")
    parser.add_argument(
        "--thresold",
        help="specify thresold for regime detection",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--bthresold",
        help="specify b thresold for regime detection",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--dthresold",
        help="specify duration thresold for regime detection",
        type=float,
        default=10,
    )
    parser.add_argument(
        "--window", help="stopping criteria for nlopt", type=int, default=10
    )
    parser.add_argument("--debug", help="acticate debug", action="store_true")
    args = parser.parse_args()

    print(f"wd: {os.getcwd()}")

    ignored_keys = [
        "Affiliation",
        "User",
        "Local",
        "form",
        "PR",
        "requested",
        "accepted",
        "Shots",
        "selection",
        "Selcom",
        "Progress",
        "Letters",
        "title",
    ]

    csvfile = args.csvfile  # eg "python_magnetrun/examples/proposals.csv"
    if not os.path.isfile(csvfile):
        raise RuntimeError(f"{csvfile}: no such file")

    proposals = pd.read_csv(args.csvfile)
    Keys = proposals.columns.values.tolist()
    drop_columns = []
    rename_columns = {"Debut": "Start", "Fin": "Stop"}
    for key in Keys:
        ignored = False
        print(key, end=" ")
        for ignore in ignored_keys:
            if ignore in key:
                drop_columns.append(key)
                print("*", end=" ")
                ignored = True
                pass

        if not ignored and " " in key:
            # split instead and replace first char of each item by an uppercase
            nkey = ""
            for substr in key.split(" "):
                nkey += substr.capitalize()
            # nkey = key.replace(" ", "")
            print(f"-> {nkey}", end="")
            rename_columns[key] = nkey
        print()

    # make sure to anonymise data
    if len(drop_columns) != 0:
        print(f"remove {drop_columns}")
        proposals.drop(drop_columns, axis=1, inplace=True)

    print(f"rename_columns: {rename_columns}")
    proposals.rename(columns=rename_columns, inplace=True)
    print(f"new keys: {proposals.columns.values.tolist()}")

    # save anonymized data
    proposals.to_csv("anonymized_proposals.csv")

    # select exp done in Grenoble
    # print(proposals['Facility'])
    df = proposals.query("Facility == 'Grenoble'")
    # print(df.head())

    # remove Supra Proposaltype
    # df.query("Proposaltype != 'Supra'", inplace=True)

    # select Done or InProgress
    df.query("ExperimentState in ['Done', 'InProgress']", inplace=True)

    print(len(df))
    print(df.head())

    # need to "Researcharea"
    # NB review keys Researcharea or ResearchArea, Units??
    selected_df = df[
        [
            "ProjectID",
            "Acronym",
            "ProposalType",
            "Site",
            "EnergyUsed",
            "Start",
            "Stop",
        ]
    ]

    # drop rows where site=nan
    selected_df = selected_df[selected_df["Site"].notna()]

    # filter out
    _df = selected_df.query(
        'Start == "0000-00-00 00:00:00" or Stop == "0000-00-00 00:00:00"'
    )
    if not _df.empty:
        print(f"drop: {_df}")
        selected_df.query(
            'Start != "0000-00-00 00:00:00" and Stop != "0000-00-00 00:00:00"',
            inplace=True,
        )

    # see: https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
    # pd.to_datetime('13000101', format='%Y%m%d', errors='coerce')
    try:
        selected_df["Start"] = pd.to_datetime(
            selected_df.Start, format="%Y-%m-%d %H:%M:%S"
        )
        selected_df["Stop"] = pd.to_datetime(
            selected_df.Stop, format="%Y-%m-%d %H:%M:%S"
        )
    except:
        print("failed to convert Start/Stop to real timestamp", flush=True)
        """
        print(f"types: {selected_df.dtypes}")
        from datetime import datetime

        for i in selected_df.itertuples():
            project = i.ProjectID
            site = i.Site
            start = i.Start
            stop = i.Stop
            # print(
            #     f"project: {project}, site={site}, start={start}, stop={stop}",
            #     flush=True,
            # )
            start_t = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
            # print(f"start_t: {start_t}")
        exit(1)
        """

    # print(selected_df.head())

    # find records attached to each line
    # recover list of records stored
    from glob import glob
    from datetime import datetime

    mdatadir = args.mdatadir  # eg srvdata
    print(f"Get records from {mdatadir}", flush=True)
    files = glob(f"{mdatadir}/*.txt")
    housing_data = []
    files_data = []
    experiment_start = []
    for file in files:
        # print(f"file: {file}", end=", ", flush=True)
        try:
            (housing, experiment_date) = file.replace(f"{mdatadir}/", "").split("_")
            # print(f'housing: {housing}, date: {experiment_date.replace(".txt","")}')
            housing_data.append(housing)

            datetime_str = experiment_date.replace(".txt", "")
            datetime_object = datetime.strptime(datetime_str, "%Y.%m.%d---%H:%M:%S")
            experiment_start.append(datetime_object)
            files_data.append(file)

        except:
            # print(" - ignored")
            pass

    files_data = sorted(
        files_data, key=lambda x: getTimestamp(x, args.debug), reverse=False
    )
    print(f"record files ({len(files_data)})")

    # create a dataframe holding: start, housing, recordfilename
    experiments = pd.DataFrame(
        data={
            "housing": housing_data,
            "filename": files_data,
            "start": experiment_start,
        }
    )
    # print(experiments.head())

    # loop over ProjetID and get records in between start and stop
    # NB: pd.itertuples return a namedtuple
    print("attach records to project")
    attached_records = []
    for i in selected_df.itertuples():
        project = i.ProjectID
        site = i.Site
        start = i.Start
        stop = i.Stop
        print(
            f"project: {project}, site={site}, start={start}, stop={stop}",
            end="",
            flush=True,
        )

        try:
            runs = experiments.loc[experiments["start"].between(start, stop)]
            housing = str(site)
            mask = runs["housing"] == housing[:-1]
            print(f"\n{runs.loc[mask]}")
            records = runs.loc[mask]["filename"].to_list()
            attached_records.append(records)
            print(f"project: {project}, site={site}, attached_records: {len(records)}")
        except:
            print("- no records")
            attached_records.append([])

    nrecords = 0
    for records in attached_records:
        nrecords += len(records)
    print(f"attached_records: {nrecords} / {len(files)} total records")

    selected_df["records"] = attached_records
    print(selected_df.head())
    # export selected_df
    selected_df.to_csv("project_records.csv")

    # get orphaned records?
    # link with MagnetDB?

    # create a dataset for classification vs researcharea, proposaltype
    print("stats per project (ignore run that does not exceed 60 s)")
    for i in selected_df.itertuples():
        project = i.ProjectID
        site = i.Site
        start = i.Start
        stop = i.Stop
        records = i.records
        # TODO: add researcharea and proposaltype
        print(
            f"project: {project}, site={site}, start={start}, stop={stop}, records={len(records)}",
            flush=True,
        )
        for record in records:
            ignored = False
            try:
                data = load_record(record)
            except Exception as e:
                print(f"caught {type(e)}: e", flush=True)
                print(f"{record}: fail to load -  ignored", flush=True)
                ignored = True
                pass

            if not ignored:
                data.Units()
                print(f"load {record} (duration={data.getDuration()} s)")
                if data.getDuration() >= 60:
                    stats(data)

                    # see utils/plateaux.py
                    pdata = nplateaus(
                        data,
                        xField=("t", "s"),
                        yField=("Field", "T"),
                        threshold=args.thresold,
                        num_points_threshold=600,
                        show=args.debug,
                        save=args.save,
                        verbose=False,
                    )
                    for plateau in pdata:
                        if plateau["value"] > 1.0:
                            print(f"{plateau}", flush=True)
                    # exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
