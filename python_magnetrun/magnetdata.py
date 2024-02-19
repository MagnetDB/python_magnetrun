"""MagnetData"""

import os
import sys
import datetime
import matplotlib.pyplot as plt
from nptdms import TdmsFile
import pandas as pd
import numpy as np
import matplotlib

# print("matplotlib=", matplotlib.rcParams.keys())
matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams['text.latex.unicode'] = True key not available


class MagnetData:
    """
    Magnet Data

    FileName: name of the input file
    Data: pandas dataframe or tdms file
    Keys:
    Groups:
    Type: 0 for Pandas data, 1 for Tdms
    """

    def __init__(
        self,
        filename: str,
        Groups: dict,
        Keys: list,
        Type: int = 0,
        Data: pd.DataFrame | TdmsFile | None = None,
    ) -> None:
        """default constructor"""
        self.FileName = filename
        self.Groups = Groups
        self.Keys = Keys
        self.Type = Type  # 0 for Pandas, 1 for Tdms, 2: for Ensight
        if Data is not None:
            self.Data = Data
        self.units = dict()

    @classmethod
    def fromtdms(cls, name: str):
        """create from a tdms file"""
        Keys = []
        Groups = {}
        Data = None
        with open(name, "r"):
            f_extension = os.path.splitext(name)[-1]
            # print("f_extension: % s" % f_extension)
            if f_extension == ".tdms":
                Data = TdmsFile.open(name)
                for group in Data.groups():
                    for channel in group.channels():
                        Keys.append(channel.name)
                        Groups[channel.name] = group.name
            else:
                raise RuntimeError(f"fromtdms: expect a tdms filename - got {name}")
        return cls(name, Groups, Keys, 1, Data)

    @classmethod
    def fromtxt(cls, name: str):
        """create from a txt file"""
        with open(name, "r") as f:
            f_extension = os.path.splitext(name)[-1]
            # print("f_extension: % s" % f_extension)
            if f_extension == ".txt":
                Data = pd.read_csv(f, sep=r"\s+", engine="python", skiprows=1)
                Keys = Data.columns.values.tolist()
            else:
                raise RuntimeError(f"fromtxt: expect a txt filename - got {name}")
        # print("MagnetData.fromtxt: ", Data)
        return cls(name, {}, Keys, 0, Data)

    @classmethod
    def fromensight(cls, name: str):
        """create from a cvs ensight file"""
        with open(name, "r") as f:
            # f_extension=os.path.splitext(name)[-1]
            # f_extension += "-ensight"
            Data = pd.read_csv(f, sep=",", engine="python", skiprows=2)
            Keys = Data.columns.values.tolist()
        return cls(name, {}, Keys, 2, Data)

    @classmethod
    def fromcsv(cls, name: str):
        """create from a cvs file"""
        with open(name, "r") as f:
            # get file extension
            f_extension = os.path.splitext(name)[-1]
            Data = pd.read_csv(f, sep=str(","), engine="python", skiprows=0)
            Keys = Data.columns.values.tolist()
        return cls(name, {}, Keys, 0, Data)

    @classmethod
    def fromStringIO(cls, name: str, sep: str = r"\s+", skiprows: int = 1):
        """create from a stringIO"""
        from io import StringIO

        Data = pd.DataFrame()
        Keys = []
        try:
            Data = pd.read_csv(
                StringIO(name), sep=sep, engine="python", skiprows=skiprows
            )
            Keys = Data.columns.values.tolist()
        except:
            print("magnetdata.fromStringIO: trouble loading data")
            with open("wrongdata.txt", "w", newline="\n") as fo:
                fo.write(name)
            pass

        return cls("stringIO", {}, Keys, 0, Data)

    def __repr__(self):
        return "%s(Type=%r, Groups=%r, Keys=%r, Data=%r)" % (
            self.__class__.__name__,
            self.Type,
            self.Groups,
            self.Keys,
            self.Data,
        )

    def getType(self):
        """returns Data Type"""
        return self.Type

    def getData(self, key: str = None):
        """return Data"""
        if not key:
            return self.Data
        else:
            if key in self.Keys:
                if isinstance(self.Data, pd.DataFrame):
                    return self.Data[key]
                elif isinstance(self.Data, TdmsFile):
                    return self.Data[self.Groups[key]]
            else:
                raise Exception(f"cannot get data for key {key}: no such key")

    def Units(self):
        """
        set units and symbols for data in record

        NB: to print unit use '[{:~P}]'.format(self.units[key][1])
        """
        from pint import UnitRegistry

        ureg = UnitRegistry()

        # if self.Type == 0:
        for key in self.Keys:
            print(key)
            if key.startwith("I"):
                self.units[key] = ("I", ureg.ampere)
            elif key.startwith("U"):
                self.units[key] = ("U", ureg.volt)
            elif key.startwith("T") or key.startwith("teb") or key.startwith("tsb"):
                self.units[key] = ("T", ureg.degC)
            elif key == "t":
                self.units[key] = ("t", ureg.second)
            elif key.startwith("Rpm"):
                self.units[key] = ("Rpm", "rpm")
            elif key.startwith("DR"):
                self.units[key] = ("%", "")
            elif key.startwith("Flo"):
                self.units[key] = ("Q", ureg.liter / ureg.second)
            elif key.startwith("debit"):
                self.units[key] = ("Q", ureg.meter**3 / ureg.second)
            elif key.startwith("Fie"):
                self.units[key] = ("B", ureg.tesla)
            elif key.startwith("HP") or key.startwith("BP"):
                self.units[key] = ("P", ureg.bar)
            elif key == "Pmagnet" or key == "Ptot":
                self.units[key] = ("Power", ureg.megawatt)
            elif key == "Q":
                # TODO define a specific 'var' unit for this field
                self.units[key] = ("Preac", ureg.megawatt)

    def getKeys(self):
        """return list of Data keys"""
        # print("type: ", type(self.Keys))
        return self.Keys

    def cleanupData(self):
        """removes empty columns from Data"""

        # print(f"Clean up Data")
        if isinstance(self.Data, pd.DataFrame):
            import re

            # print(f'self.Keys = {self.Keys}') # Data.columns.values.tolist()}')
            Ikeys = [_key for _key in self.Keys if re.match(r"Icoil\d+", _key)]
            Fkeys = [_key for _key in self.Keys if re.match(r"Flow\w+", _key)]
            Fkeys += [_key for _key in self.Keys if re.match(r"\w+_ref", _key)]
            # print(f'IKeys = {Ikeys}')
            # print(f'FKeys = {Fkeys}')

            # drop duplicates
            _df = self.Data.T.drop_duplicates().T
            # print(f'uniq Keys = {_df.columns.values.tolist()}')

            # TODO remove empty column except that with a name that starts with Icoil*
            empty_cols = [
                col
                for col in _df.columns
                if _df[col].isnull().all()
                and not col.startswith("Icoil")
                and not col.startswith("Flow")
            ]
            if empty_cols:
                # print(f'empty cols: {empty_cols}')
                _df.drop(empty_cols, axis=1, inplace=True)
                # print(f'uniq Keys wo empty cols = {_df.columns.values.tolist()}')

            # Always add latest Ikeys if not already in _df
            if Ikeys[-1] not in _df.columns.values.tolist():
                _df = pd.concat([_df, self.Data[Ikeys[-1]]], axis=1)

            # Kepp Fkeys if not already in _df
            _df_keys = _df.columns.values.tolist()
            for key in Fkeys:
                if key not in _df_keys:
                    _df = pd.concat([_df, self.Data[key]], axis=1)

            self.Data = _df
            self.Keys = self.Data.columns.values.tolist()
            # print(f'--> self.Keys = {self.Keys}') # Data.columns.values.tolist()}')
        return 0

    def removeData(self, keys: list):
        """remove a column to Data"""
        if isinstance(self.Data, pd.DataFrame):
            for key in keys:
                if key in self.Keys:
                    # print(f"Remove {key}")
                    del self.Data[key]
                else:
                    print(f"cannot remove {key}: no such key - skip operation")
            self.Keys = self.Data.columns.values.tolist()
        return 0

    def renameData(self, columns: dict):
        """
        rename columns
        """
        if isinstance(self.Data, pd.DataFrame):
            self.Data.rename(columns=columns, inplace=True)
            self.Keys = self.Data.columns.values.tolist()
        else:
            raise RuntimeError(f"cannot rename {columns.keys()}: no such columns")

    def addData(self, key, formula):
        """
        add a new column to Data from  a formula

        see:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.eval.html
        https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html#enhancingperf-eval
        """

        # print("addData: %s = %s" % (key, formula) )
        if isinstance(self.Data, pd.DataFrame):
            if key in self.Keys:
                print(f"Key {key} already exists in DataFrame")
            else:
                # check formula using pyparsing

                self.Data.eval(formula, inplace=True)
                self.Keys = self.Data.columns.values.tolist()
        return 0

    def getStartDate(self) -> tuple:
        """get start timestamps"""
        res = ()
        if isinstance(self.Data, pd.DataFrame):
            # print("keys=", self.Keys)
            if "Date" in self.Keys and "Time" in self.Keys:
                tformat = "%Y.%m.%d %H:%M:%S"
                start_date = self.Data["Date"].iloc[0]
                start_time = self.Data["Time"].iloc[0]
                end_date = self.Data["Date"].iloc[-1]
                end_time = self.Data["Time"].iloc[-1]
                res = (start_date, start_time, end_date, end_time)
        return res

    def getDuration(self):
        """compute duration of the run in seconds"""
        # print("magnetdata.getDuration")
        duration = None
        if "timestamp" in self.Keys:
            start_time = self.Data["timestamp"].iloc[0]
            end_time = self.Data["timestamp"].iloc[-1]
            dt = end_time - start_time
            # print(f'dt={dt}')
            duration = dt.seconds
            # print(f'duration={duration}')
        else:
            print("magnetdata.getDuration: no timestamp key")
            print(f"available keys are: {self.Keys}")
        return duration

    def addTime(self):
        """add a Time column to Data"""
        # print("magnetdata.AddTime")

        if isinstance(self.Data, pd.DataFrame):
            if "Date" in self.Keys and "Time" in self.Keys:
                tformat = "%Y.%m.%d %H:%M:%S"
                t0 = datetime.datetime.strptime(
                    self.Data["Date"].iloc[0] + " " + self.Data["Time"].iloc[0], tformat
                )
                self.Data["t"] = self.Data.apply(
                    lambda row: (
                        datetime.datetime.strptime(row.Date + " " + row.Time, tformat)
                        - t0
                    ).total_seconds(),
                    axis=1,
                )
                self.Data["timestamp"] = self.Data.apply(
                    lambda row: datetime.datetime.strptime(
                        row.Date + " " + row.Time, tformat
                    ),
                    axis=1,
                )
                # print("magnetdata.AddTime: add t and timestamp")
                # remove Date and Time ??
                self.Data.drop(["Date", "Time"], axis=1, inplace=True)
                # print("magnetdata.AddTime: drop done")
                # regenerate keys
                self.Keys = self.Data.columns.values.tolist()
                # print("magnetdata.AddTime: regenerate keys")

            else:
                raise Exception("cannot add t[s] columnn: no Date or Time column")
        return 0

    def extractData(self, keys) -> pd.DataFrame:
        """extract columns keys to Data"""

        if self.Type == 0:
            for key in keys:
                if key not in self.Keys:
                    raise Exception(
                        f"{self.__class__.__name__}.{sys._getframe().f_code.co_name}: no {key} key"
                    )

            return pd.concat([self.Data[key] for key in keys], axis=1)

        else:
            raise RuntimeError(f"extractData: magnetdata type ({self.Type})unsupported")

    def extractDataThreshold(self, key, threshold):
        """extra data above a given threshold for field"""
        if isinstance(self.Data, pd.DataFrame):
            if key not in self.Keys:
                raise Exception(
                    f"extractData: key={key} - no such keys in dataframe (valid keys are: {self.Keys()}"
                )

            return self.Data.loc[self.Data[key] >= threshold]
        else:
            raise Exception(f"extractData: not implement for {type(self.Data)}")

    def extractTimeData(self, timerange) -> pd.DataFrame:
        """extract column to Data"""

        if isinstance(self.Data, pd.DataFrame):
            trange = timerange.split(";")
            print(f"Select data from {trange[0]} to {trange[1]}")

            return self.Data[
                self.Data["Time"].between(trange[0], trange[1], inclusive="both")
            ]
        else:
            raise RuntimeError(
                f"extractTimeData: magnetdata type ({self.Type})unsupported"
            )

    def saveData(self, keys, filename):
        """save Data to csv format"""
        if isinstance(self.Data, pd.DataFrame):
            self.Data[keys].to_csv(filename, sep=str("\t"), index=False, header=True)
            return 0

    def plotData(self, x, y, ax):
        """plot x vs y"""

        # print("plotData Type:", self.Type, f"x={x}, y={y}" )
        if x not in self.Keys:
            if isinstance(self.Data, pd.DataFrame):
                raise Exception(
                    f"{self.__class__.__name__}.{sys._getframe().f_code.co_name}: no x={x} key"
                )
            else:
                if x != "Time":
                    Exception(
                        f"{self.__class__.__name__}.{sys._getframe().f_code.co_name}: no {x} key"
                    )

        if y in self.Keys:
            if isinstance(self.Data, pd.DataFrame):
                self.Data.plot(x=x, y=y, ax=ax, grid=True)
            else:
                group = self.Data[self.Groups[y]]
                channel = group[y]
                samples = channel.properties["wf_samples"]

                if x == "Time":
                    increment = channel.properties["wf_increment"]
                    time_steps = np.array([i * increment for i in range(0, samples)])

                    plt.plot(time_steps, self.Data[self.Groups[y]][y], label=y)
                    plt.ylabel(" [" + channel.properties["unit_string"] + "]")
                    plt.xlabel("t [s]")
                else:
                    group = self.Data[self.Groups[x]]
                    xchannel = group[x]

                    plt.plot(
                        self.Data[self.Groups[x]][x],
                        self.Data[self.Groups[y]][y],
                        label=y,
                    )
                    plt.ylabel(" [" + channel.properties["unit_string"] + "]")
                    plt.xlabel(
                        xchannel.name + " [" + xchannel.properties["unit_string"] + "]"
                    )

                plt.grid(b=True)
                ax.legend()

        else:
            raise Exception(
                f"{self.__class__.__name__}.{sys._getframe().f_code.co_name}: no {y} key"
            )

    def stats(self, key: str = None):
        """returns stats for key"""
        print("magnetdata.stats")
        if isinstance(self.Data, pd.DataFrame):
            # print(f'keys: {self.Keys}')
            if key is not None:
                if key in self.Keys:
                    print(f"stats[{key}]: {self.Data[key].mean()}")
                    return self.Data[key].describe()
                else:
                    raise Exception(
                        f"{self.__class__.__name__}.{sys._getframe().f_code.co_name}: no {y} key"
                    )
            else:
                for _key in self.Keys:
                    if _key not in ["timestamp"]:
                        print(
                            f'stats[{_key}]: {self.Data[_key].describe(include="all")}'
                        )
        else:
            raise Exception("data is not a panda dataframe: cannot get stats")
