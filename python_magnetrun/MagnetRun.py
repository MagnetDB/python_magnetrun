"""Main module."""

import re
import pandas as pd

from .magnetdata import MagnetData


def prepareData(data: MagnetData, housing: str, cleanup: bool = False):
    # get start/end
    (start_date, start_time, end_date, end_time) = data.getStartDate()
    # print(f'start_date={start_date}, start_time={start_time}, end_date={end_date}, end_time={end_time}')

    # add timestamp
    data.addTime()
    # print(f'addTime done')

    # get duration
    duration = data.getDuration()
    # print(f'duration={duration}')

    # TODO use a dict struct to simplify this?
    # shall check if key exist beforehand
    if housing == "M9":
        # print('M9 housing case')
        data.addData("IH_ref", "IH_ref = Idcct1 + Idcct2")
        data.addData("IB_ref", "IB_ref = Idcct3 + Idcct4")

        # FlowH = Flow1, FlowB = Flow2
        for field in ["Flow", "Rpm", "Tin", "HP"]:
            data.renameData(
                columns={f"{field}1": f"{field}H", f"{field}2": f"{field}B"}
            )

    elif housing in ["M8", "M10"]:
        # print('M8/M10 housing case')
        data.addData("IH_ref", "IH_ref = Idcct3 + Idcct4")
        data.addData("IB_ref", "IB_ref = Idcct1 + Idcct2")
        
        # FlowH = Flow2, FlowB = Flow1
        for field in ["Flow", "Rpm", "Tin", "HP"]:
            data.renameData(
                columns={f"{field}1": f"{field}B", f"{field}2": f"{field}H"}
            )
    # what about M1, M5 and M7???

    data.removeData(['Idcct1', 'Idcct2', 'Idcct3', 'Idcct4'])
    
    data.cleanupData()
    Ikey = [_key
        for _key in data.getKeys()
        if re.match(r"Icoil\d+", _key)]
    # print(f"Ikey={Ikey}")

    data.renameData(columns={f"{Ikey[0]}": "IH"})
    data.renameData(columns={f"{Ikey[-1]}": "IB"})


    # print(f"data.keys={data.getKeys()}")


class MagnetRun:
    """
    Magnet Run

    Housing: name of the housing
    Site: name of site (magnetdb sense)
    MagnetData: pandas dataframe or tdms file
    """

    def __init__(
        self,
        housing: str = "unknown",
        site: str = "",
        data: MagnetData | None = None,
    ):
        """default constructor"""
        self.Housing = housing
        self.Site = site
        self.MagnetData = data

    @classmethod
    def fromtxt(cls, housing, site, filename):
        """create from a txt file"""
        # print(f"MagnetRun/fromtxt: housing={housing}, site={site}, filename={filename}")
        with open(filename, "r") as f:
            insert = f.readline().split()[-1]
            data = MagnetData.fromtxt(filename)
            prepareData(data, housing)

        # print("magnetrun.fromtxt: data=", data)
        return cls(housing, site, data)

    @classmethod
    def fromcsv(cls, housing, site, filename):
        """create from a csv file"""
        data = MagnetData.fromcsv(filename)
        return cls(housing, site, data)

    @classmethod
    def fromStringIO(cls, housing, site, name):
        """create from a stringIO"""
        # print(f'MagnetRun/fromStringIO: housing={housing}, site={site}')
        from io import StringIO

        insert = "Unknown"
        data = MagnetData(filename="", Groups={}, Keys=[])
        try:
            ioname = StringIO(name)
            # TODO rework: get item 2 otherwise set to unknown
            headers = ioname.readline().split()
            if len(headers) >= 2:
                insert = headers[1]
            if not site.startswith(insert):
                print(f"MagnetRun:fromStringIO: site={site}, insert={insert}")
            data = MagnetData.fromStringIO(name)
            # print(f'data keys({len(data.getKeys())}): {data.getKeys()}')
            prepareData(data, housing)
            # print(f'prepareData: data keys({len(data.getKeys())}): {data.getKeys()}')

        except:
            with open("wrongdata.txt", "w", newline="\n") as fo:
                fo.write(name)
            print(f'cannot load data for {housing}, {insert} insert, {site} site"')
            # raise RuntimeError(f'cannot load data for {housing}, {insert} insert, {site} site"')
        return cls(housing, site, data)

    def __repr__(self):
        return "%s(Housing=%r, Site=%r, MagnetData=%r)" % (
            self.__class__.__name__,
            self.Housing,
            self.Site,
            self.MagnetData,
        )

    def getSite(self):
        """returns Site"""
        return self.Site

    def getHousing(self):
        """returns Housing"""
        return self.Housing

    def setSite(self, site):
        """set Site"""
        self.Site = site

    def getType(self):
        """returns Data Type"""
        if self.MagnetData is not None:
            return self.MagnetData.Type
        else:
            raise RuntimeError("MagnetRun.getType: no MagnetData associated")

    def getMData(self) -> MagnetData:
        """return Magnet Data object"""
        if self.MagnetData is not None:
            return self.MagnetData
        else:
            raise RuntimeError("no magnetdata attached to this magnetrun")

    def getData(self, key: str = ""):
        """return Data"""
        if self.MagnetData is not None:
            return self.MagnetData.getData(key)
        else:
            raise RuntimeError("MagnetRun.getType: no MagnetData associated")

    def getKeys(self):
        """return list of Data keys"""
        if self.MagnetData is not None:
            return self.MagnetData.Keys
        else:
            raise RuntimeError("MagnetRun.getType: no MagnetData associated")

    def getStats(self, field: str=None):
        """return basic stats"""
        if self.MagnetData is not None:
            return self.MagnetData.stats(field)
        else:
            raise RuntimeError("MagnetRun.getType: no MagnetData associated")
    
    def saveData(self, filename: str):
        """save Data to file"""
        if self.MagnetData is not None:
            if isinstance(self.MagnetData.Data, pd.DataFrame):
                self.MagnetData.Data.to_csv(
                    filename, sep=str("\t"), index=False, header=True
                )
            else:
                raise RuntimeError(
                    f"MagnetRun.save: unsupported type of Data ({type(self.MagnetData.Data)})"
                )
