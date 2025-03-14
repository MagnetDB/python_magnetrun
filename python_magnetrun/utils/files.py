from __future__ import unicode_literals
import datetime
import pandas as pd

# TODO use MagnetData instead of files


def concat_files(input_files: list, keys: list = [], debug: bool = False):
    if debug:
        print(f"concat_files: input_files={input_files}, keys={keys}")

    df_f = []
    for i, f in enumerate(input_files):
        _df = pd.DataFrame()
        if debug:
            print(f"concat_files: process {f}")
        try:
            if f.endswith(".txt"):
                _df = pd.read_csv(f, sep=r"\s+", engine="python", skiprows=1)
            else:
                _df = pd.read_csv(f, sep="str(',')", engine="python", skiprows=0)

        except:
            raise RuntimeError(f"load_files: failed to load {f} with pandas")

        # print(f'load_files: {f}')
        if keys:
            df_f.append(_df[keys])
        else:
            df_f.append(_df)

    df = pd.concat(df_f, axis=0)

    # Drop empty columns
    df = df.loc[:, (df != 0.0).any(axis=0)]

    try:
        # Add a time column
        tformat = "%Y.%m.%d %H:%M:%S"
        start_date = df["Date"].iloc[0]
        start_time = df["Time"].iloc[0]
        end_time = df["Time"].iloc[-1]
        print("start_time=", start_time, "start_date=", start_date)

        t0 = datetime.datetime.strptime(
            df["Date"].iloc[0] + " " + df["Time"].iloc[0], tformat
        )
        df["t"] = df.apply(
            lambda row: (
                datetime.datetime.strptime(row.Date + " " + row.Time, tformat) - t0
            ).total_seconds(),
            axis=1,
        )
        # # del df['Date']
        # # del df['Time']
    except:
        pass

    return df
