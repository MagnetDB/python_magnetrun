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
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True key not available

import pint

class MagnetData:
    """
    Magnet Data

    FileName: name of the input file
    Data: pandas dataframe or tdms file
    Keys:
    Groups:
    Type: 0 for Pandas data, 1 for Tdms
    """
        
    def __init__(self, filename, Groups=dict(), Keys=[], Type=0, Data=None):
        """default constructor"""
        self.FileName = filename
        self.Groups = Groups
        self.Keys = Keys
        self.Type = Type # 0 for Pandas, 1 for Tdms, 2: for Ensight
        self.Data = Data
        self.units = dict()

    @classmethod
    def fromtdms(cls, name):
        """create from a tdms file"""
        Keys = []
        Groups = {}
        Data = None
        with open(name, 'r') as f:
            f_extension=os.path.splitext(name)[-1]
            # print("f_extension: % s" % f_extension)
            if f_extension != ".tdms":
                raise("fromtdms: expect a tdms filename - got %s" % name)
            
            rawData = TdmsFile.open(name, 'r')
            for group in rawData.groups():
                print(f'group: {group.name}', flush=True)

                for channel in group.channels():
                    print(f'channel: {channel.name}', flush=True)
                    Groups[channel.name] = group.name

            Data = rawData.as_dataframe(time_index=False, absolute_time=False, scaled_data=True, arrow_dtypes=False)
            Keys = Data.columns.values.tolist()
            
        print(f'magnetdata/fromtdms: Groups={Groups}', flush=True)
        return cls(name, Groups, Keys, 1, Data)

    @classmethod
    def fromtxt(cls, name):
        """create from a txt file"""
        with open(name, 'r') as f:
            f_extension=os.path.splitext(name)[-1]
            # print("f_extension: % s" % f_extension)
            if f_extension == ".txt":
                Data = pd.read_csv(f,
                                   sep=r'\s+',
                                   engine='python',
                                   skiprows=1)
                Keys = Data.columns.values.tolist()
            else:
                raise("fromtxt: expect a txt filename - got %s" % name)
        # print("MagnetData.fromtxt: ", Data)
        return cls(name, [], Keys, 0, Data)

    @classmethod
    def fromensight(cls, name):
        """create from a cvs ensight file"""
        with open(name, 'r') as f:
            # f_extension=os.path.splitext(name)[-1]
            # f_extension += "-ensight"
            Data = pd.read_csv(f,
                               sep=",",
                               engine='python',
                               skiprows=2)
            Keys = Data.columns.values.tolist()
        return cls(name, [], Keys, 2, Data)

    @classmethod
    def fromcsv(cls, name):
        """create from a cvs file"""
        with open(name, 'r') as f:
            # get file extension
            f_extension=os.path.splitext(name)[-1]
            Data = pd.read_csv(f,
                               sep=str(","),
                               engine='python',
                               skiprows=0)
            Keys = Data.columns.values.tolist()
        return cls(name, [], Keys, 0, Data)
    
    @classmethod
    def fromStringIO(cls, name, sep=r'\s+', skiprows=1):
        """create from a stringIO"""
        from io import StringIO
        
        Data = None
        Keys = None
        try:
            Data = pd.read_csv(StringIO(name),
                               sep=sep,
                               engine='python',
                               skiprows=skiprows)
            Keys = Data.columns.values.tolist()
        except:
            print("magnetdata.fromStringIO: trouble loading data")
            fo = open("wrongdata.txt", "w", newline='\n')
            fo.write(name)
            fo.close()
            #sys.exit(1)
            pass
            
        return cls("stringIO", [], Keys, 0, Data)

    def __repr__(self):
        return "%s(Type=%r, Groups=%r, Keys=%r, Data=%r)" % \
            (self.__class__.__name__,
             self.Type,
             self.Groups,
             self.Keys,
             self.Data)

    def getType(self):
        """returns Data Type"""
        return self.Type

    def getData(self, key=""):
        """return Data"""
        if not key:
            return self.Data
        else:
            if key in self.Keys:
                if self.Type == 0:
                    return self.Data[key]
                else:
                    return self.Data[self.Groups[key]]
            else:
                raise Exception("cannot get data for key %s: no such key" % key)

    def Units(self):
        """
        set units and symbols for data in record
        
        NB: to print unit use '[{:~P}]'.format(self.units[key][1])
        """
        for key in self.keys:
            print(key)
            if key.startwith('I'):
                self.units[key] = ('I', ureg.ampere)
            elif key.startwith('U'):
                self.units[key] = ('U', ureg.volt)
            elif key.startwith('T') or key.startwith('teb') or key.startwith('tsb') :
                self.units[key] = ('T', ureg.degC)
            elif key == 't' :
                self.units[key] = ('t', ureg.second)
            elif key.startwith('Rpm'):
                self.units[key] = ('Rpm', 'rpm')
            elif key.startwith('DR'):
                self.units[key] = ('%', '')
            elif key.startwith('Flo'):
                self.units[key] = ('Q', ureg.liter/ureg.second)
            elif key.startwith('debit'):
                self.units[key] = ('Q', ureg.meter**3/ureg.second)
            elif key.startwith('Fie'):
                self.units[key] = ('B', ureg.tesla)
            elif key.startwith('HP') or key.startwith('BP'):
                self.units[key] = ('P', ureg.bar)
            elif key == "Pmagnet" or key == "Ptot":
                self.units[key] = ('Power', ureg.megawatt)
            elif key == "Q":
                # TODO define a specific 'var' unit for this field
                self.units[key] = ('Preac', ureg.megawatt)

    def getKeys(self):
        """return list of Data keys"""
        #print("type: ", type(self.Keys))
        return self.Keys

    def cleanupData(self):
        """removes empty columns from Data"""
        if self.Type == 0 :
            print(f"Clean up Data")
            self.Data = self.Data.loc[:, (self.Data != 0.0).any(axis=0)]
            self.Keys = self.Data.columns.values.tolist()
        return 0

    def removeData(self, key):
        """remove a column to Data"""
        if self.Type == 0 :
            if key in self.Keys:
                print(f"Remove {key}")
                del self.Data[key]
                self.Keys = self.Data.columns.values.tolist()
        else:
            raise Exception("cannot remove %s: no such key" % key)
        return 0

    def addData(self, key, formula):
        """
        add a new column to Data from  a formula
        
        see: 
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.eval.html
        https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html#enhancingperf-eval
        """

        # print("addData: %s = %s" % (key, formula) )
        if self.Type == 0 :
            if key in self.Keys:
                print("Key %s already exists in DataFrame")
            else:
                # check formula using pyparsing
                
                self.Data.eval(formula, inplace=True)
                self.Keys = self.Data.columns.values.tolist()
        return 0

    def getStartDate(self):
        """get start timestamps"""
        res = ()
        if self.Type == 0 :
            # print("keys=", self.Keys)
            if "Date" in self.Keys and "Time" in self.Keys:
                tformat="%Y.%m.%d %H:%M:%S"
                start_date=self.Data["Date"].iloc[0]
                start_time=self.Data["Time"].iloc[0]
                end_time=self.Data["Time"].iloc[-1]
                res = (start_date, start_time, end_time)
        return res
    
    def addTime(self):
        """add a Time column to Data"""

        # print("magnetdata.AddTime")
        if self.Type == 0 :
            # print("keys=", self.Keys)
            if "Date" in self.Keys and "Time" in self.Keys:
                tformat="%Y.%m.%d %H:%M:%S"
                start_date=self.Data["Date"].iloc[0]
                start_time=self.Data["Time"].iloc[0]
                end_time=self.Data["Time"].iloc[-1]
                # print ("start_time=", start_time, "start_date=", start_date)

                t0 = datetime.datetime.strptime(self.Data['Date'].iloc[0]+" "+self.Data['Time'].iloc[0], tformat)
                self.Data["t"] = self.Data.apply(lambda row: (datetime.datetime.strptime(row.Date+" "+row.Time, tformat)-t0).total_seconds(), axis=1)
                self.Data["timestamp"] = self.Data.apply(lambda row: datetime.datetime.strptime(row.Date+" "+row.Time, tformat), axis=1)
                self.Keys = self.Data.columns.values.tolist()
            else:
                raise Exception("cannot add t[s] columnn: no Date or Time column")
        return 0

    def extractData(self, keys):
        """extract columns keys to Data"""

        newdf = None
        if self.Type == 0 :
            for key in keys:
                if not key in self.Keys:
                    raise Exception("%s.%s: no %s key" % (self.__class__.__name__, sys._getframe().f_code.co_name, key) )
                
            newdf = pd.concat([self.Data[key] for key in keys], axis=1)
                
        return newdf

    def extractTimeData(self, timerange):
        """extract column to Data"""

        selected_df = None
        if self.Type == 0 :
            trange = timerange.split(";")
            print ("Select data from %s to %s" % (trange[0],trange[1]) )

            selected_df=self.Data[self.Data['Time'].between(trange[0], trange[1], inclusive=True)]
        return selected_df

    def saveData(self, keys, filename):
        """save Data to csv format"""
        if self.Type == 0 :
            self.Data[keys].to_csv(filename, sep=str('\t'), index=False, header=True)
            return 0

    def plotData(self, x, y, ax):
        """plot x vs y"""
        
        # print("plotData Type:", self.Type, "x=%s, y=%s" % (x,y) )
        if not x in self.Keys:
            if self.Type == 0 :
                raise Exception("%s.%s: no x=%s key" % (self.__class__.__name__, sys._getframe().f_code.co_name, x) )
            else:
                if x != "Time" :
                    Exception("%s.%s: no %s key" % (self.__class__.__name__, sys._getframe().f_code.co_name, x) )

        if y in self.Keys:
            if self.Type == 0 :
                self.Data.plot(x=x, y=y, ax=ax, grid=True)
            else:
                group = self.Data[self.Groups[y]]
                channel = group[y]
                samples = channel.properties['wf_samples']

                if x == 'Time':
                    increment = channel.properties['wf_increment']
                    time_steps = np.array([i*increment for i in range(0,samples)])

                    plt.plot(time_steps, self.Data[self.Groups[y]][y], label=y)
                    plt.ylabel(" [" + channel.properties['unit_string'] + "]")
                    plt.xlabel("t [s]")
                else:
                    group = self.Data[self.Groups[x]]
                    xchannel = group[x]

                    plt.plot(self.Data[self.Groups[x]][x], self.Data[self.Groups[y]][y], label=y)
                    plt.ylabel(" [" + channel.properties['unit_string'] + "]")
                    plt.xlabel(xchannel.name + " [" + xchannel.properties['unit_string'] + "]")

                plt.grid(b=True)
                ax.legend()
                
        else:
            raise Exception("%s.%s: no y=%s key" % (self.__class__.__name__, sys._getframe().f_code.co_name, y) )

    def stats(self):
        """compute stats fro the actual run"""
        return 0

