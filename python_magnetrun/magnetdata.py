"""MagnetData"""

from logging import raiseExceptions
from natsort import natsorted
from datetime import datetime

import os
import sys

import pandas as pd
import numpy as np
from scipy import stats


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
        Keys: list[str],
        Type: int = 0,
        Data: pd.DataFrame | dict = None,
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
        """create from a pigbrother file

        :param name: filename with a tdms extension
        :type name: str
        :raises RuntimeError: _description_
        :return: magnetdata object
        :rtype: magnetdata
        """
        from nptdms import TdmsFile

        Keys = []
        Groups = {}
        Data = {}
        with open(name, "r"):
            f_extension = os.path.splitext(name)[-1]
            # print("f_extension: % s" % f_extension)

            if f_extension != ".tdms":
                raise RuntimeError(f"fromtdms: expect a tdms filename - got {name}")

            rawData = TdmsFile.open(name)
            # print(f"rawData: {rawData.properties}", flush=True)
            for group in rawData.groups():
                gname = group.name.replace(" ", "_")
                gname = gname.replace("_et_Ref.", "")
                Groups[gname] = {}
                if gname != "Infos":
                    Data[gname] = {}

                    # print(group.channels, type(group.channels))
                    for channel in group.channels():
                        cname = channel.name.replace(" ", "_")
                        Keys.append(f"{gname}/{cname}")
                        Groups[gname][cname] = channel.properties
                        # print(f"properties: {channel.properties}", flush=True)

                    Data[gname] = group.as_dataframe(
                        time_index=False,
                        absolute_time=False,
                        scaled_data=True,
                    )

                    # rename columns to avoid space in columns name
                    Data[gname].rename(
                        columns={
                            col: col.replace(" ", "_") for col in Data[gname].columns
                        },
                        inplace=True,
                    )

                    # print(f"Data[{gname}]\n{Data[gname].head()}")
                else:
                    Groups[gname]["Infos"] = group
            # print(f"keys: {Keys}")

        # Add reference for GR1, GR2
        if "Référence_A1" in Data["Courants_Alimentations"]:
            Data["Courants_Alimentations"]["Référence_GR1"] = (
                Data["Courants_Alimentations"]["Référence_A1"]
                + Data["Courants_Alimentations"]["Référence_A2"]
            )
            Keys.append("Courants_Alimentations/Référence_GR1")
            Groups["Courants_Alimentations"]["Référence_GR1"] = Groups[
                "Courants_Alimentations"
            ][
                "Référence_A1"
            ]  # "Added Référence_A1+Référence_A2"
        if "Référence_A3" in Data["Courants_Alimentations"]:
            Data["Courants_Alimentations"]["Référence_GR2"] = (
                Data["Courants_Alimentations"]["Référence_A3"]
                + Data["Courants_Alimentations"]["Référence_A4"]
            )
            Keys.append("Courants_Alimentations/Référence_GR2")
            Groups["Courants_Alimentations"]["Référence_GR2"] = Groups[
                "Courants_Alimentations"
            ][
                "Référence_A3"
            ]  # "Added Référence_A3+Référence_A4"

        # print(f"magnetdata/fromtdms: Groups={Groups}", flush=True)
        return cls(name, Groups, Keys, 1, Data)

    @classmethod
    def fromtxt(cls, name: str):
        """create from a pupitre file

        :param name: filename with a txt extension
        :type name: str
        :raises RuntimeError: _description_
        :return: magnetdata object
        :rtype: magnetdata
        """
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
            # f_extension = os.path.splitext(name)[-1]
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
        except Exception:
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

    def getPandasData(self, key: list[str] | str | None) -> pd.DataFrame:
        """return Data or a selection using key"""
        # print(f"MagnetData/Data({key}): {self.FileName}", flush=True)

        if key is None:
            if not isinstance(self.Data, pd.DataFrame):
                raise Exception(
                    f"MagnetData/Data: {self.FileName} - expect Data to be a pandas dataframe"
                )
            return self.Data
        else:
            selected_keys = []
            if isinstance(key, list):
                selected_keys = key
            elif isinstance(key, str):
                selected_keys = [key]

            for item in selected_keys:
                if item not in self.Keys:
                    raise Exception(
                        f"MagnetData/Data({key}): {self.FileName}: cannot get data for key={item}: no such key"
                    )

        # print(f"selected_keys={selected_keys}", flush=True)
        _df = self.Data[selected_keys]
        _df_keys = _df.columns.values.tolist()
        # print(f"_df_keys = {_df_keys}", flush=True)
        return _df

    def getTdmsData(self, group: str, channel: str | list[str] | None) -> pd.DataFrame:
        """_summary_

        :param group: _description_
        :type group: str
        :param channel: _description_
        :type channel: str
        """

        if not isinstance(self.Data, dict):
            raise Exception(
                f"MagnetData/getTdmsData: {self.FileName} - expect Data to be a dict"
            )

        if channel is None:
            return self.Data[group]
        else:
            # print(f"getTdms: group={group}, channel={channel}")
            if isinstance(channel, list):
                return self.Data[group][channel]
            else:
                return self.Data[group][channel]

    def getData(self, key: list[str] | str = None) -> pd.DataFrame:
        """_summary_

        :param group: _description_, defaults to None
        :type group: str, optional
        :param channels: _description_, defaults to None
        :type channels: list[str] | str, optional
        :return: _description_
        :rtype: _type_
        """
        if self.Type == 0:
            return self.getPandasData(key)
        elif self.Type == 1:
            channels = []
            groups = []
            if isinstance(key, str):
                (group, channel) = key.split("/")
                channels.append(channel)
                groups.append(group)
            elif isinstance(key, list):
                channels = []

                # print(f"magnetda.getData: key={key}", flush=True)
                for item in key:
                    (group, channel) = item.split("/")
                    # print(f"group{group}, channel={channel}", flush=True)
                    channels.append(channel)
                    if group not in groups:
                        groups.append(group)
                # print(f"groups={groups}")
                # print(f"channels={channels}")

            if len(groups) > 1 or len(groups) == 0:
                raise RuntimeError(
                    f"magnetata:getData for tdms - expect only one group - got {len(groups)}"
                )

            return self.getTdmsData(groups[0], channels)
        else:
            raise RuntimeError(
                f"getData not implemented for MagneData.Data type={self.Type}"
            )

    def PigBrotherUnits(self, key: str, debug: bool = False) -> tuple:
        from pint import UnitRegistry

        # print(f"PigBrotherUnits: key={key}")
        ureg = UnitRegistry()

        PigBrotherUnits = {
            "Courant": ("I", ureg.ampere),
            "Tension": ("U", ureg.volt),
            "Puissance": ("Power", ureg.watt),
            "Champ_magn": ("B", ureg.gauss),
        }

        for entry in PigBrotherUnits:
            if entry in key:
                return PigBrotherUnits[entry]

        return ()

    def Units(self, debug: bool = False):
        """
        set units and symbols for data in record

        NB: to print unit use '[{:~P}]'.format(self.units[key][1])
        """
        from pint import UnitRegistry

        # print(f'magnetdata: Units, Filename:{self.FileName}')

        ureg = UnitRegistry()
        ureg.define("percent = 1 / 100 = %")
        ureg.define("ppm = 1e-6 = ppm")
        ureg.define("var = 1")

        if self.Type == 1:
            # print("self.Data=", type(self.Data))
            for entry in self.Data:
                # print("entry=", entry, type(entry))
                if entry == "t":
                    self.units["t"] = ("t", ureg.second)
                else:
                    group = entry
                    if "/" in entry:
                        (group, channel) = entry.split("/")
                        if channel == "t":
                            self.units[entry] = ("t", ureg.second)
                        # if channel == "timestamp":
                        #    self.units[entry] = ("t", ureg.second)
                    self.units[entry] = self.PigBrotherUnits(group)

            pass

        elif self.Type > 1:
            print(f"magnetdata/getUnits: ignore for type={self.Type}")
            pass

        for key in self.Keys:
            if key == "timestamp":
                # TODO define a specific 'var' unit for this field
                self.units[key] = ("time", None)
            elif key == "t":
                self.units[key] = ("t", ureg.second)
            elif key == "Field":
                self.units[key] = ("B", ureg.tesla)
            elif key.startswith("I"):
                self.units[key] = ("I", ureg.ampere)
            elif key.startswith("U"):
                self.units[key] = ("U", ureg.volt)
            elif key.startswith("T") or key == "teb" or key == "tsb":
                self.units[key] = ("T", ureg.degC)
            elif key.startswith("Rpm"):
                self.units[key] = ("Rpm", ureg.rpm)
            elif key.startswith("DR"):
                self.units[key] = ("%", ureg.percent)
            elif key.startswith("Flo"):
                self.units[key] = ("Q", ureg.liter / ureg.second)
            elif key == "debitbrut":
                self.units[key] = ("Q", ureg.meter**3 / ureg.hour)
            elif key.startswith("HP") or key.startswith("BP"):
                self.units[key] = ("P", ureg.bar)
            elif key == "Pmagnet" or key == "Ptot":
                self.units[key] = ("Power", ureg.megawatt)
            elif key.startswith("Power"):
                self.units[key] = ("Power", ureg.megawatt)
            elif key == "Q":
                # TODO define a specific 'var' unit for this field
                self.units[key] = ("Preac", ureg.megavar)

        # if self.Type == 1 # aka data from tdms
        # load units from tdms but check if coherent
        if debug:
            print(f"Units: {self.Keys}")
            for key, values in self.units.items():
                symbol = values[0]
                unit = values[1]
                print(f"{key}: symbol={symbol}, unit={unit:~P}", flush=True)

    def getUnitKey(self, key: str) -> tuple:
        # print(f'getUnitKey: {key}')
        if key not in self.Keys:
            from pint import UnitRegistry

            # print(f"getUnitKey: {key} not in {self.Keys}")
            ureg = UnitRegistry()
            if key == "t":
                return ("t", ureg.second)
            elif key == "timestamp":
                return ("time", None)
            else:
                raise RuntimeError(
                    f"{key} not defined in data - available keys are {self.Keys}"
                )

        if self.Type == 0:
            return self.units[key]
        elif self.Type == 1:
            (group, channel) = key.split("/")
            # !!! do not use tdms units - they are wrong !!!
            # unit_name = self.Data[group][channel].properties["NI_UnitDescription"]
            # unit_string = self.Data[group][channel].properties["unit_string"]
            # print(f"getUnitKey: {group}/{channel} -> {self.PigBrotherUnits(group)}")
            return self.PigBrotherUnits(group)
        return ()

    def getKeys(self):
        """return list of Data keys"""
        # print("type: ", type(self.Keys))
        return self.Keys

    def cleanupData(self, debug: bool = False):
        """removes empty columns from Data

        Apply only to pupitre files

        :param debug:activate debug mode, defaults to False
        :type debug: bool, optional
        :raises RuntimeError: _description_
        :return: _description_
        :rtype: _type_
        """

        if debug:
            print(
                f"Clean up Data: filename={self.FileName}, keys={self.Keys}", flush=True
            )
        if self.Type == 0:  # isinstance(self.Data, pd.DataFrame):
            import re

            # print(f'self.Keys = {self.Keys}') # Data.columns.values.tolist()}')
            init_Ikeys = natsorted(
                [_key for _key in self.Keys if re.match(r"Icoil\d+", _key)]
            )
            if debug:
                print(f"init_Ikeys: {init_Ikeys}")
            Fkeys = [_key for _key in self.Keys if re.match(r"Flow\w+", _key)]
            Fkeys += [_key for _key in self.Keys if re.match(r"Rpm\w+", _key)]
            Fkeys += [_key for _key in self.Keys if re.match(r"HP\w+", _key)]
            Fkeys += [_key for _key in self.Keys if re.match(r"\w+_ref", _key)]
            Fkeys += [_key for _key in self.Keys if re.match(r"Pmagnet", _key)]
            Fkeys += [_key for _key in self.Keys if re.match(r"Ptot", _key)]

            # print(f'FKeys = {Fkeys}')

            # drop duplicates
            def getDuplicateColumns(df):
                # Create an empty set
                duplicateColumnNames = set()

                # Iterate through all the columns of dataframe
                for x in range(df.shape[1]):
                    # Take column at xth index.
                    col = df.iloc[:, x]

                    for y in range(x + 1, df.shape[1]):
                        # Take column at yth index.
                        otherCol = df.iloc[:, y]

                        if col.equals(otherCol):
                            duplicateColumnNames.add(df.columns.values[y])

                # Return list of unique column names whose contents are duplicates.
                return list(duplicateColumnNames)

            if debug:
                print(
                    f"zero columns: {natsorted(self.Data.columns[(self.Data == 0).all()].values.tolist())}",
                    flush=True,
                )

            # TODO remove empty column except that with a name that starts with Icoil*
            empty_cols = [
                col
                for col in self.Data.columns[(self.Data == 0).all()].values.tolist()
                if not col.startswith("Flow") and not col.startswith("Field")
            ]
            empty_Ikeys = natsorted(
                [_key for _key in empty_cols if re.match(r"Icoil\d+", _key)]
            )
            # print(f"empty cols: {natsorted(empty_cols)}")
            # print(f"empty Ikeys: {empty_Ikeys}")
            _df = self.Data
            if empty_cols:
                _df = self.Data.drop(empty_cols, axis=1)
                # print(f'uniq Keys wo empty cols = {_df.columns.values.tolist()}')

            dropped_columns = getDuplicateColumns(_df)
            # print(
            #    f"duplicated columns: {natsorted(dropped_columns)}",
            #    flush=True,
            # )
            # faster but drop every detected duplicates: _df = _df.T.drop_duplicates().T
            # instead detect remove colums which are duplicated
            # do not remove Ucoil duplicates
            really_dropped_columns = natsorted(
                [col for col in dropped_columns if not col.startswith("Ucoil")]
            )
            # print(
            #    f"really duplicated columns: {really_dropped_columns}",
            #    flush=True,
            # )
            _df.drop(really_dropped_columns, axis=1, inplace=True)
            # print(
            #    f"_df uniq Keys = {natsorted(_df.columns.values.tolist())}", flush=True
            # )

            # Find Ucoil for Helices and Bitters
            Ukeys = natsorted(
                [
                    str(_key)
                    for _key in _df.columns.values.tolist()
                    if re.match(r"Ucoil\d+", _key)
                ]
            )
            # print(f"UKeys = {Ukeys}")

            from itertools import groupby

            Uindex = [int(ukey.replace("Ucoil", "")) for ukey in Ukeys]
            # print(f"Uindex = {Uindex}")

            # Enumerate and get differences between counter—integer pairs
            # Group by differences (consecutive integers have equal differences)
            gb = groupby(enumerate(Uindex), key=lambda x: x[0] - x[1])

            # Repack elements from each group into list
            all_groups = ([i[1] for i in g] for _, g in gb)

            # Filter out one element lists
            Uprobes = list(filter(lambda x: len(x) > 1, all_groups))
            # print(f"{self.FileName} Uprobes: {Uprobes}")
            if not Uprobes:
                raise RuntimeError(f"{self.FileName}: CleanUpData no Uprobes found")
                # (keys={self.Data.keys()}, non_empty_keys={_df.keys()})"

            UH = [f"Ucoil{i}" for i in Uprobes[0]]
            _df["UH"] = _df[UH].sum(axis=1)
            if debug:
                print(f"UH: {UH}")
            if len(Uprobes) > 1:
                UB = [f"Ucoil{i}" for i in Uprobes[1]]
                _df["UB"] = _df[UB].sum(axis=1)
                if debug:
                    print(f"UB: {UB}")

            # Always add latest Ikeys if not already in _df
            Ikeys = natsorted(
                [
                    _key
                    for _key in _df.columns.values.tolist()
                    if re.match(r"Icoil\d+", _key)
                ]
            )
            if debug:
                print(f"IKeys = {Ikeys} ({len(Ikeys)})")
            if Ikeys:
                # print(
                #     f"nonnull Ikeys: {natsorted(set(init_Ikeys).difference(set(empty_Ikeys)))}"
                # )
                if len(Ikeys) == 1:
                    if debug:
                        print(
                            f"{self.FileName}: check if {init_Ikeys[-1]} or {init_Ikeys[-2]} in _df"
                        )
                    if init_Ikeys[-1] not in Ikeys and init_Ikeys[-2] not in Ikeys:
                        # IH only
                        # print(f"add {init_Ikeys[-2]}")
                        _df = pd.concat([_df, self.Data[init_Ikeys[-2]]], axis=1)
                    else:
                        # IB only: first Icoil item of natsorted(init_Ikeys) not in really_dropped_columns
                        # print(f"add {init_Ikeys[0]}")
                        _df = pd.concat([_df, self.Data[init_Ikeys[0]]], axis=1)

                    Ikeys = natsorted(
                        [
                            _key
                            for _key in _df.columns.values.tolist()
                            if re.match(r"Icoil\d+", _key)
                        ]
                    )

                elif len(Ikeys) == 2:
                    if debug:
                        print("need to check consistancy")

                else:
                    if debug:
                        print(
                            f"{self.FileName}:try to cure dataset - got {Ikeys} expect at most 2 values",
                            flush=True,
                        )
                    ikeys = self.Data[Ikeys]
                    """
                    print(
                        f"{self.FileName}: ikeys={ikeys.keys()}, Ikeys={Ikeys}",
                        flush=True,
                    )
                    """

                    remove_Ikeys = []
                    for i in range(len(Ikeys)):
                        for j in range(i + 1, len(Ikeys)):
                            diff = ikeys[Ikeys[i]] - ikeys[Ikeys[j]]
                            error = diff.mean()
                            stderror = diff.std()
                            if debug:
                                print(
                                    f"diff[{Ikeys[i]}_{Ikeys[j]}: mean={error}, std={stderror}"
                                )
                            if abs(error) <= 1.0e-2:
                                remove_Ikeys.append(Ikeys[j])
                                # print(f"remove_Ikeys({Ikeys[j]})")

                    if debug:
                        print(f"remove_Ikeys: {remove_Ikeys}")
                    if remove_Ikeys:
                        """
                        print(
                            f"{self.FileName}: ikeys={ikeys.keys()}, Ikeys={Ikeys}",
                            flush=True,
                        )
                        print(f"remove_Ikeys: {remove_Ikeys}")
                        """
                        _df.drop(remove_Ikeys, axis=1, inplace=True)

                    Ikeys = natsorted(
                        [
                            _key
                            for _key in _df.columns.values.tolist()
                            if re.match(r"Icoil\d+", _key)
                        ]
                    )

                    if len(Ikeys) == 1:
                        if debug:
                            print(
                                f"{self.FileName}: check if {init_Ikeys[-1]} or {init_Ikeys[-2]} in _df"
                            )
                        if (
                            init_Ikeys[-1] not in _df.columns.values.tolist()
                            and init_Ikeys[-2] not in _df.columns.values.tolist()
                        ):
                            # IH only
                            _df = pd.concat([_df, self.Data[init_Ikeys[-2]]], axis=1)
                        else:
                            # IB only
                            _df = pd.concat([_df, self.Data[init_Ikeys[0]]], axis=1)

                    elif len(Ikeys) > 2:
                        _df[Ikeys].to_csv(f"{self.FileName}.ikey")
                        raise RuntimeError(
                            f"{self.FileName}: strange number of Ikeys detected - got {Ikeys} expect at most 2 values"
                        )

            else:
                # TODO: not working as expected if Ucoils are not correct
                Ukeys = natsorted(
                    [
                        str(_key)
                        for _key in _df.columns.values.tolist()
                        if re.match(r"Ucoil\d+", _key)
                    ]
                )
                # print(
                #     f"{self.FileName}: empty Ikeys - try to recover Ikeys from Ukeys={Ukeys}"
                # )
                for i, key in enumerate(Ukeys):
                    Ukeys[i] = key.replace("U", "I")
                Ikeys = [Ukeys[0], Ukeys[-1]]
                _df = pd.concat([_df, self.Data[Ikeys[0]], self.Data[Ikeys[1]]], axis=1)

            # print(f"IKeys = {Ikeys}")

            # Keep Fkeys if not already in _df
            _df_keys = _df.columns.values.tolist()
            for key in Fkeys:
                if key not in _df_keys:
                    _df = pd.concat([_df, self.Data[key]], axis=1)

            self.Data = _df
            self.Keys = self.Data.columns.values.tolist()
            if debug:
                print(f"--> self.Keys = {self.Keys}")  # Data.columns.values.tolist()}')
        return 0

    def removeData(self, keys: list):
        """remove a column to Data"""
        if self.Type == 0:
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
        if self.Type == 0:
            self.Data.rename(columns=columns, inplace=True)
            self.Keys = self.Data.columns.values.tolist()

    def addData(self, key, formula, unit: str = None, debug: bool = False):
        """
        add a new column to Data from  a formula

        see:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.eval.html
        https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html#enhancingperf-eval
        """

        # print("addData: %s = %s" % (key, formula) )
        if self.Type == 0:
            if key in self.Keys:
                print(f"Key {key} already exists in DataFrame")
            else:
                # check formula using pyparsing
                self.Data.eval(formula, inplace=True)
                self.Keys = self.Data.columns.values.tolist()
                if unit:
                    self.units[key] = unit
                else:
                    self.Units(debug)
        elif self.Type == 1:
            (group, channel) = key.split("/")
            print(f"add: key={key} - group={group}, channel={channel}")

            nformula = formula.replace(f"{group}/", "")
            # print(f"formula: {nformula}")

            # looking for group/ in nformula
            import re

            match = re.findall(r"(\w+)/(\w+)", nformula)
            if match:
                for matched in match:
                    print(f"matched={matched[0]}/{matched[1]}")
                    self.Data[group][matched[1]] = self.Data[matched[0]][matched[1]]
                    nformula = nformula.replace(f"{matched[0]}/", "")
            print(f"formula: {nformula}")

            try:
                self.Data[group].eval(nformula, inplace=True)
                self.Keys.append(key)

                # Get wf_increment from tdms group data except if matched wf_increment is different
                first_key = list(self.Groups[group].keys())[0]
                self.Groups[group][channel] = {
                    "wf_increment": self.Groups[group][first_key]["wf_increment"]
                }

            except pd.errors.UndefinedVariableError as error:
                raise RuntimeError(
                    f"addData: {key}: {nformula} - failed for tdms {group} data - error={error}"
                )

            # self.Groups[group][channel] = self.Groups[group]
            # raise RuntimeError("addData: not implemented for pigbrother file")
        return 0

    def computeData(
        self,
        method,
        key: str,
        kparams: list,
        unit: tuple = None,
        debug: bool = False,
    ):
        """
        compute new column
        """
        if debug:
            print(f"computeData: Key={key}", end=" ")

        if key in self.Keys:
            print(f"Key {key} already exists in DataFrame")
            return 0

        if self.Type == 0:
            data = []
            for values in self.Data[kparams].values.tolist():
                data.append(method(*values))

            # print(data)
            self.Data[key] = data

            self.Keys = self.Data.columns.values.tolist()
            if unit:
                self.units[key] = unit
            else:
                self.Units(debug)

            if debug:
                print("done")

        elif self.Type == 1:
            raise RuntimeError(
                f"computeData: key={key} not implemented for pigbrother file"
            )

    def getStartDate(self, group: str = None) -> tuple:
        """get start timestamp"""
        res = ()
        if self.Type == 0:  # isinstance(self.Data, pd.DataFrame):
            # print("keys=", self.Keys)
            if "Date" in self.Keys and "Time" in self.Keys:
                tformat = "%Y.%m.%d %H:%M:%S"
                start_date = self.Data["Date"].iloc[0]
                start_time = self.Data["Time"].iloc[0]
                end_date = self.Data["Date"].iloc[-1]
                end_time = self.Data["Time"].iloc[-1]
                res = (start_date, start_time, end_date, end_time)
        elif self.Type == 1:
            start_t = self.Data[group]["timestamp"].iloc[0]
            # end_t = self.Data[group]["timestamp"].iloc[-1]

            dformat = "%Y.%m.%d"
            tformat = "%H:%M:%S"
            start_date = datetime.strptime(start_t, dformat)
            start_time = datetime.strptime(start_t, tformat)
            end_date = datetime.strptime(start_t, dformat)
            end_time = datetime.strptime(start_t, tformat)
            res = (start_date, start_time, end_date, end_time)
        return res

    def getDuration(self, group: str = None) -> float:
        """compute duration of the run in seconds"""
        # print("magnetdata.getDuration")
        duration = 0
        if self.Type == 0:
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
        elif self.Type == 1:
            if group is None:
                group = "Tensions_Aimant"
            # print(f"self.Groups[{group}]={self.Groups[group]}")
            channel = list(self.Groups[group].keys())[0]
            ordered_dict = self.Groups[group][channel]
            # print(f"self.Groups[{group}][{channel}]={self.Groups[group][channel]}")
            # print(f"self.Groups[{group}][{channel}]={type(self.Groups[group][channel])}")

            # start_time = self.Data[group]["t"].iloc[0]
            # end_time = self.Data[group]["t"].iloc[-1]
            dt = ordered_dict["wf_increment"]
            samples = ordered_dict["wf_samples"]
            duration = dt * samples
        return duration

    def addTime(self):
        """add a Time column to Data"""
        # print("magnetdata.AddTime")

        if self.Type == 0:  # isinstance(self.Data, pd.DataFrame):
            if "Date" in self.Keys and "Time" in self.Keys:
                #  tformat = "%Y.%m.%d %H:%M:%S"

                try:
                    self.Data["Date"] = pd.to_datetime(
                        self.Data.Date, cache=True, format="%Y.%m.%d"
                    )
                except Exception:
                    raise RuntimeError(
                        f"MagnetData/AddTime {self.FileName}: failed to convert Date"
                    )

                try:
                    self.Data["Time"] = pd.to_timedelta(self.Data.Time)
                except Exception:
                    raise RuntimeError(
                        f"MagnetData/AddTime {self.FileName}: failed to convert Time"
                    )

                try:
                    self.Data["timestamp"] = self.Data.Date + self.Data.Time
                except Exception:
                    raise RuntimeError(
                        f"MagnetData/AddTime {self.FileName}: failed to create timestamp column"
                    )
                else:
                    t0 = self.Data.iloc[0]["timestamp"]

                from .utils.duplicates import find_duplicates

                self.Data = find_duplicates(self.Data, self.FileName, "timestamp")

                self.Data["t"] = self.Data.apply(
                    lambda row: (row.timestamp - t0).total_seconds(),
                    axis=1,
                )

                # check interval in t column
                times = self.Data["t"].to_numpy()
                dt = np.diff(times)
                if dt.min() != dt.max():
                    print(
                        f"!!! {self.FileName}: dt from {dt.min()} to {dt.max()} !!!"
                    )  # stats={stats.describe(dt)}"

                # print("magnetdata.AddTime: add t and timestamp")
                # remove Date and Time ??
                self.Data.drop(["Date", "Time"], axis=1, inplace=True)
                # print("magnetdata.AddTime: drop done")
                # regenerate keys
                self.Keys = self.Data.columns.values.tolist()
                # print("magnetdata.AddTime: regenerate keys")

            else:
                raise RuntimeError(
                    f"MagnetData/AddTime {self.FileName}: cannot add t[s] columnn: no Date or Time columns"
                )
        return 0

    def extractData(self, keys: list[str]) -> pd.DataFrame:
        """extract columns keys to Data

        :param keys: list of selected keys
        :type keys: list[str]
        :raises Exception: RuntimeError
        :return: pd.DataFrame
        :rtype: pd.DataFrame
        """

        if self.Type == 0:
            for key in keys:
                if key not in self.Keys:
                    raise RuntimeError(
                        f"{self.__class__.__name__}.{sys._getframe().f_code.co_name}: no {key} key"
                    )
            return pd.concat([self.Data[key] for key in keys], axis=1)

        elif self.Type == 1:
            dfs = []
            for item in keys:
                (group, channel) = item.split("/")
                dfs.append(self.getTdmsData(group, channel))
            return pd.concat(dfs)

        # else:
        #    raise RuntimeError(f"extractData: magnetdata type ({self.Type})unsupported")

    def extractDataThreshold(self, key, threshold):
        """extra data above a given threshold for field"""
        if self.Type == 0:
            # if isinstance(self.Data, pd.DataFrame):
            if key not in self.Keys:
                raise RuntimeError(
                    f"extractData: key={key} - no such keys in dataframe (valid keys are: {self.Keys}"
                )
            return self.Data.loc[self.Data[key] >= threshold]

        elif self.Type == 1:
            (group, channel) = key.split("/")
            return self.Data[group][channel].loc[self.Data[group][channel] >= threshold]
        else:
            raise RuntimeError(f"extractData: not implement for {type(self.Data)}")

    def extractTimeData(self, timerange, group: str = None) -> pd.DataFrame:
        """extract column to Data"""

        trange = timerange.split(";")
        print(f"Select data from {trange[0]} to {trange[1]}")

        if self.Type == 0:  # isinstance(self.Data, pd.DataFrame):
            return self.Data[
                self.Data["Time"].between(trange[0], trange[1], inclusive="both")
            ]
        elif self.Type == 1:
            return self.Data[group]["timestamp"].between(
                trange[0], trange[1], inclusive="both"
            )
        else:
            raise RuntimeError(
                f"extractTimeData: magnetdata type ({self.Type})unsupported"
            )

    def saveData(self, keys: list[str], filename: str):
        """save Data to csv format

        :param keys:list of selected keys
        :type keys: list[str]
        :param filename: _description_
        :type filename: str
        :return: _description_
        :rtype: _type_
        """
        if self.Type == 0:
            self.Data[keys].to_csv(filename, sep=str("\t"), index=False, header=True)
        elif self.Type == 1:
            dfs = []
            for key in keys:
                (group, channel) = key.split("/")
                dfs.append(self.getTdmsData(group, channel))
            df = pd.concat(dfs)
            df.to_csv(filename, sep=str("\t"), index=False, header=True)

        return 0

    def plotData(
        self,
        x: str,
        y: str,
        ax,
        alpha: float = 1,
        label: str = None,
        normalize: bool = False,
    ):
        """plot x vs y

        :param x: _description_
        :type x: _type_
        :param y: _description_
        :type y: _type_
        :param ax: _description_
        :type ax: _type_
        :param label: _description_, defaults to None
        :type label: str, optional
        :param normalize: _description_, defaults to False
        :type normalize: bool, optional
        :raises RuntimeError: _description_
        :raises RuntimeError: _description_
        :raises Exception: _description_
        """

        import datetime
        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.rcParams["text.usetex"] = True

        # print("plotData Type:", self.Type, f"x={x}, y={y}" )
        if x != "t" and x != "timestamp" and x not in self.Keys:
            raise RuntimeError(
                f"{self.__class__.__name__}.{sys._getframe().f_code.co_name}: no x={x} key (valid keys= {self.Keys})"
            )

        if y in self.Keys:
            (ysymbol, yunit) = self.getUnitKey(y)

            # if isinstance(self.Data, pd.DataFrame):
            if self.Type == 0:
                if normalize:
                    df = self.Data.copy()
                    ymax = abs(df[y].max())
                    df[y] /= ymax
                    label = f"{y} (norm with {ymax:.3e} {yunit:~P})"
                    df.plot(
                        x=x,
                        y=y,
                        ax=ax,
                        alpha=alpha,
                        label=f"{y} (norm with {ymax:.3e} {yunit:~P})",
                        grid=True,
                    )
                    del df
                else:
                    self.Data.plot(x=x, y=y, ax=ax, alpha=alpha, grid=True)
            elif self.Type == 1:
                # print(x, y)

                (ygroup, ychannel) = y.split("/")
                if "/" in x:
                    (xgroup, xchannel) = x.split("/")
                else:
                    xgroup = ygroup
                    xchannel = x

                if xgroup == ygroup:
                    df = self.Data[xgroup].copy()
                    dt = self.Groups[ygroup][ychannel]["wf_increment"]

                    if xchannel == "t":
                        df[xchannel] = df.index * dt  # + dt/2. ??
                    elif xchannel == "timestamp":
                        t0 = self.Groups[ygroup][ychannel]["wf_start_time"]
                        # print(f"plotData ({self.FileName}: t0={t0})")
                        df[xchannel] = [
                            np.datetime64(t0).astype(datetime.datetime)
                            + datetime.timedelta(0, i * dt)
                            for i in df.index.to_list()
                        ]

                    if normalize:
                        ymax = abs(df[ychannel].max())
                        df[ychannel] /= ymax
                        df.plot(
                            x=xchannel,
                            y=ychannel,
                            ax=ax,
                            alpha=alpha,
                            label=f"{ychannel} (norm with {ymax:.3e} {yunit:~P})",
                            grid=True,
                        )
                        del df
                    else:
                        df.plot(x=xchannel, y=ychannel, ax=ax, grid=True)
                else:
                    raise RuntimeError(
                        f"{self.__class__.__name__}.{sys._getframe().f_code.co_name}: xgroup={xgroup} != {ygroup}"
                    )

            # add xlabel, ylabel from units
            if yunit is not None:
                plt.ylabel(f"{ysymbol} [{yunit:~P}]")

            (xsymbol, xunit) = self.getUnitKey(x)
            if xunit is not None:
                plt.xlabel(f"{xsymbol} [{xunit:~P}]")

        else:
            raise Exception(
                f"{self.__class__.__name__}.{sys._getframe().f_code.co_name}: no {y} key (valid keys: {self.Keys})"
            )

    def stats(self, key: str = None):
        """display basic statistics

        :param key: key to display, defaults to None
        :type key: str, optional
        :raises Exception: RuntimeError
        :return: pd.DataFrame with stats
        :rtype: pd.DataFrame
        """
        from tabulate import tabulate

        print("magnetdata.stats")

        # TODO: use tabulate to get more pretty output

        if self.Type == 0:
            if key is not None:
                if key in self.Keys:
                    # print(f"stats[{key}]: {self.Data[key].mean()}")
                    print(
                        tabulate(
                            self.Data[key].describe(), headers="keys", tablefmt="psql"
                        )
                    )
                else:
                    raise RuntimeError(
                        f"{self.__class__.__name__}.{sys._getframe().f_code.co_name}: no {key} key"
                    )
            else:
                df = self.Data.describe(include="all")
                print(
                    tabulate(
                        df,
                        headers="keys",
                        tablefmt="psql",
                    )
                )

        elif self.Type == 1:
            if key is not None:
                (group, channel) = key.split("/")
                if group in self.Data:
                    if channel in self.Data[group]:
                        print(
                            tabulate(
                                self.Data[group][channel].describe(),
                                headers="keys",
                                tablefmt="psql",
                            )
                        )
                        return self.Data[group][channel].describe()
                    else:
                        raise RuntimeError(
                            f"magnetdata/stats: cannot find channel {channel}"
                        )
                else:
                    raise RuntimeError(f"magnetdata/stats: cannot find group {group}")
            else:
                for group in self.Data:
                    print(f"stats[{group}]: ")
                    df = self.Data[group].describe(include="all")
                    print(
                        tabulate(
                            df,
                            headers="keys",
                            tablefmt="psql",
                        )
                    )

    def info(self):
        """magnetdata info"""

        print(f"magnetdata: {self.FileName}, Type={self.Type}")
        if self.Type == 0:
            print("keys:")
            for key in self.Keys:
                print(f"\t{key}")

        # TODO for tdms display Infos group
        if self.Type == 1:
            from tabulate import tabulate
            from collections import OrderedDict

            headers = [
                "Group",
                "Channel",
                "Samples",
                "Increment",
                "start_time",
                "start_offset",
            ]
            tables = []

            for group, values in self.Groups.items():
                # print(f"group={group}")
                for item in values:
                    if isinstance(values[item], dict) or isinstance(
                        values[item], OrderedDict
                    ):
                        table = [
                            group,
                            item,
                            values[item]["wf_samples"],
                            values[item]["wf_increment"],
                            values[item]["wf_start_time"],
                            values[item]["wf_start_offset"],
                        ]
                        tables.append(table)
                        """
                        for sitem in values[item]:
                            print(f"  {sitem}: {values[item][sitem]} **")
                        """
                    """
                    else:
                        print(f"  {item}: {values[item]}")
                    """

            print(tabulate(tables, headers, tablefmt="simple"))

        # print("stats:")
        # self.stats()
