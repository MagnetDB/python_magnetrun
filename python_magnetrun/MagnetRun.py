"""Main module."""

import math
import os
import sys
import datetime
import re
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# print("matplotlib=", matplotlib.rcParams.keys())
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True key not available

from .magnetdata import MagnetData

def prepareData(data: MagnetData, housing: str):
            
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
        data.addData("IH_ref", "IH_ref = Idcct1 + Idcct2")
        data.addData("IB_ref", "IB_ref = Idcct3 + Idcct4")
        # FlowH = Flow1, FlowB = Flow2
        # RpmH = Rpm1, RpmB = Rpm2
                
    elif housing in ["M8", "M10"]:
        data.addData("IH_ref", "IH_ref = Idcct3 + Idcct4")
        data.addData("IB_ref", "IB_ref = Idcct1 + Idcct2")
        # FlowH = Flow2, FlowB = Flow1
        # RpmH = Rpm2, RpmB = Rpm1
    # what about M1, M5 and M7???

    # cleanup data: remove empty columns aka columns with 0
    data.cleanupData()
    #print(f'prepareData cleanup done: {data.getKeys()}')
    
    # remove duplicates: get keys for Icoil\d+, keep first one and eventually latest (aka for Bitters),
    Ikeys = [ _key for _key in data.getKeys() if re.match("Icoil\d+", _key)]
    # print(f'Icoil keys from {Ikeys[0]} to {Ikeys[-1]}')

    # rename Icoil1 -> IH
    # rename Icoil15 -> IB

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
            
    @classmethod
    def fromtxt(cls, housing, site, filename):
        """create from a txt file"""
        print(f'MagnetRun/fromtxt: housing={housing}, site={site}, filename={filename}')
        with open(filename, 'r') as f:
            insert=f.readline().split()[-1]
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

        try:
            ioname = StringIO(name)
            # TODO rework: get item 2 otherwise set to unknown
            insert = "Unknown"
            headers = ioname.readline().split()
            if len(headers) >=2:
                insert = headers[1]
            if not site.startswith(insert):
                print(f'MagnetRun:fromStringIO: site={site}, insert={insert}')
            data = MagnetData.fromStringIO(name)
            #print(f'data keys({len(data.getKeys())}): {data.getKeys()}')
            prepareData(data, housing)
            #print(f'prepareData: data keys({len(data.getKeys())}): {data.getKeys()}')
        
        except:
            with open("wrongdata.txt", "w", newline='\n') as fo:
                fo.write(name)
            print(f'cannot load data for {housing}, {insert} insert, {site} site"')
            # raise RuntimeError(f'cannot load data for {housing}, {insert} insert, {site} site"')
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
    
    def saveData(self, filename: str):
        """save Data to file"""
        with open(filename, "w", newline='\n') as f:
            f.write(self.MagnetData)
