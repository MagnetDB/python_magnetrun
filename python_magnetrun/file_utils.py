"""File utils"""

class FileType:
    """
    Defines specific fields according to file extension
    in order to simplify read_csv
    """

    def __init__(self, ftype, sep=",", engine='python', skiprows=0):
        self.Ftype = ftype
        self.Sep = sep
        self.Engine = engine
        self.SkipRows = skiprows

    def __repr__(self):
        return "%s(Ftype=%r, Sep=%r, Engine=%r, Skiprows=%r)" % \
        ( self.__class__.__name__,
          self.Ftype,
          self.Sep,
          self.Engine,
          self.SkipRows )

    def setFtype(self, ftype):
        """set FType: set file extension"""
        self.Ftype = ftype

    def getFType(self):
        """get FType: name of file extension"""
        return self.Ftype

    def setSep(self, sep):
        """set Sep: set file separator"""
        self.Sep = sep

    def getSep(self):
        """get Sep: get file separator"""
        return self.Sep

    def setSkipRows(self, skiprows):
        """set SkipRows: set number of lines to skip while reading"""
        self.SkipRows = skiprows

    def getSkipRows(self):
        """get SkipRows: get number of lines to skip while reading"""
        return self.SkipRows

    def setEngine(self, engine):
        """set Engine: set file engine"""
        self.Engine = engine

    def getEngine(self):
        """get Engine: get file engine"""
        return self.Engine
