"""Main module."""

import math
import os
import sys
import datetime
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# print("matplotlib=", matplotlib.rcParams.keys())
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True key not available
from .magnetdata import MagnetData

class MagnetRun:
    """
    Magnet Run

    Housing: name of the housing
    Site: name of site (magnetdb sense)
    MagnetData: pandas dataframe or tdms file
    """

    def __init__(self, housing="unknown", site="", data=None):
        """default constructor"""
        self.Housing = housing
        self.Site = site
        self.MagnetData = data

        # start_date = None
        # try:
        #     if "Date" in self.MagnetData.getKeys() and "Time" in self.MagnetData.getKeys():
        #         start_date=self.MagnetData.getData("Date").iloc[0]
        #         start_time=self.MagnetData.getData("Time").iloc[0]
        #         end_date=self.MagnetData.getData("Date").iloc[-1]
        #         end_time = self.MagnetData.getData('Time').iloc[-1]

        #         tformat="%Y.%m.%d %H:%M:%S"
        #         t0 = datetime.datetime.strptime(start_date+" "+start_time, tformat)
        #         t1 = datetime.datetime.strptime(end_date+" "+end_time, tformat)
        #         dt = (t1-t0)
        #         duration = dt / datetime.timedelta(seconds=1)
                
        # except:
        #     print("MagnetRun.__init__: trouble loading data")
        #     try:
        #         file_name = "%s_%s_%s-wrongdata.txt" % (self.Housing, self.Site,start_date)
        #         self.MagnetData.to_csv(file_name, sep=str('\t'), index=False, header=True)
        #     except:
        #         print("MagnetRun.__init__: trouble loading data - fail to save csv file")
        #         pass
        #     pass
            
    @classmethod
    def fromtxt(cls, housing, site, filename):
        """create from a txt file"""
        with open(filename, 'r') as f:
            insert=f.readline().split()[-1]
            data = MagnetData.fromtxt(filename)
            # cleanup data
            # remove duplicates
            # get start/end
            # add timestamp
            # get duration
            
            if housing == "M9":
                data.addData("IH_ref", "IH_ref = Idcct1 + Idcct2")
                data.addData("IB_ref", "IB_ref = Idcct3 + Idcct4")
                # flowH = flowxx, flowB = flowyy
            elif housing in ["M8", "M10"]:
                data.addData("IH_ref", "IH_ref = Idcct3 + Idcct4")
                data.addData("IB_ref", "IB_ref = Idcct1 + Idcct2")
                # flowH = flowxx, flowB = flowyy
            # what about M1, M5 and M7???

            # remove Icoil duplicate for Helices, rename Icoil1 -> IH
            # remove Icoil duplicate for Bitter, rename Icoil15 -> IB

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
        from io import StringIO

        # try:
        ioname = StringIO(name)
        # TODO rework: get item 2 otherwise set to unknown
        insert = "Unknown"
        headers = ioname.readline().split()
        if len(headers) >=2:
            insert = headers[1]
        data = MagnetData.fromStringIO(name)
        # except:
        #      print("cannot read data for %s insert, %s site" % (insert, site) )
        #      fo = open("wrongdata.txt", "w", newline='\n')
        #      fo.write(ioname)
        #      fo.close()
        #      sys.exit(1)
        return cls(housing, site, data)

    def __repr__(self):
        return "%s(Housing=%r, Site=%r, MagnetData=%r)" % \
             (self.__class__.__name__,
              self.Housing,
              self.Site,
              self.MagnetData)

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
        return self.MagnetData.Type

    def getMData(self):
        """return Magnet Data object"""
        return self.MagnetData

    def getData(self, key=""):
        """return Data"""
        return self.MagnetData.getData(key)

    def getKeys(self):
        """return list of Data keys"""
        return self.MagnetData.Keys
    
