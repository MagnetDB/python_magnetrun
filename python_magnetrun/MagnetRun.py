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

    Site: name of the site
    Insert: list of the MagnetIDs composing the insert
    MagnetData: pandas dataframe or tdms file
    """

    def __init__(self, site="unknown", insert="", data=None):
        """default constructor"""
        self.Site = site
        self.Insert = insert
        self.MagnetData = data

        start_date = None
        try:
            if "Date" in self.MagnetData.getKeys() and "Time" in self.MagnetData.getKeys():
                start_date=self.MagnetData.getData("Date").iloc[0]
                start_time=self.MagnetData.getData("Time").iloc[0]
                end_date=self.MagnetData.getData("Date").iloc[-1]
                end_time = self.MagnetData.getData('Time').iloc[-1]

                tformat="%Y.%m.%d %H:%M:%S"
                t0 = datetime.datetime.strptime(start_date+" "+start_time, tformat)
                t1 = datetime.datetime.strptime(end_date+" "+end_time, tformat)
                dt = (t1-t0)
                duration = dt / datetime.timedelta(seconds=1)

                print("* Site: %s, Insert: %s" % (self.Site, self.Insert),
                      "MagnetData.Type: %d" % self.MagnetData.Type,
                      "start_date=%s" % start_date,
                      "start_time=%s" % start_time,
                      "duration=%g s" % duration)
                
        except:
            print("MagnetRun.__init__: trouble loading data")
            try:
                file_name = "%s_%s_%s-wrongdata.txt" % (self.Site, self.Insert,start_date)
                self.MagnetData.to_csv(file_name, sep=str('\t'), index=False, header=True)
            except:
                print("MagnetRun.__init__: trouble loading data - fail to save csv file")
                pass
            pass
            
    @classmethod
    def fromtxt(cls, site, filename):
        """create from a txt file"""
        with open(filename, 'r') as f:
            insert=f.readline().split()[-1]
            data = MagnetData.fromtxt(filename)

            if site == "M9":
                data.addData("IH", "IH = Idcct1 + Idcct2")
                data.addData("IB", "IB = Idcct3 + Idcct4")
            elif site in ["M8", "M10"]:
                data.addData("IH", "IH = Idcct3 + Idcct4")
                data.addData("IB", "IB = Idcct1 + Idcct2")
            # what about M1, M5 and M7???

        # print("magnetrun.fromtxt: data=", data)
        return cls(site, insert, data)

    @classmethod
    def fromcsv(cls, site, insert, filename):
        """create from a csv file"""
        data = MagnetData.fromcsv(filename)
        return cls(site, insert, data)

    @classmethod
    def fromStringIO(cls, site, name):
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
        return cls(site, insert, data)

    def __repr__(self):
        return "%s(Site=%r, Insert=%r, MagnetData=%r)" % \
             (self.__class__.__name__,
              self.Site,
              self.Insert,
              self.MagnetData)

    def getSite(self):
        """returns Site"""
        return self.Site

    def getInsert(self):
        """returns Insert"""
        return self.Insert

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

    def getDuration(self):
        """compute duration of the run in seconds"""
        duration = None
        if "Date" in self.MagnetData.getKeys() and "Time" in self.MagnetData.getKeys():
            start_date=self.MagnetData.getData("Date").iloc[0]
            start_time=self.MagnetData.getData("Time").iloc[0]
            end_date=self.MagnetData.getData("Date").iloc[-1]
            end_time = self.MagnetData.getData('Time').iloc[-1]

            tformat="%Y.%m.%d %H:%M:%S"
            t0 = datetime.datetime.strptime(start_date+" "+start_time, tformat)
            t1 = datetime.datetime.strptime(end_date+" "+end_time, tformat)
            dt = (t1-t0)
            duration = dt / datetime.timedelta(seconds=1)
        return duration
    
