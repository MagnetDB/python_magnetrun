"""Helix Object"""

class HMagnet:
    """
    cadref
    GObjects: list of Objects (aka rings, helices, current leads)
    records: txt files
    rapidrecords: tdms files
    MAGfile(s)
    status: Dead/Alive
    index
    """

    def __init__(self):
        """empty constructor"""
        self.cadref = 0
        self.GObjects = []
        self.records = []
        self.rapidrecords = []
        self.MAGfile = []
        self.status = "Unknown"
        self.index = 0

    def __init__(self, cadref, GObjects, records, rapidrecords, MAGfile, status):
        """defaut constructor"""
        self.cadref = cadref
        self.GObjects = GObjects
        self.records = records
        self.rapidrecords = rapidrecords
        self.MAGfile = MAGfile
        self.status = status
        self.index = index
        
    def __repr__(self):
        """
        representation of object
        """
        return "%s(geofile=%r, matID=%r)" % \
            (self.__class__.__name__,
             self.cadref = cadref,
             self.GObjects,
             self.records,
             self.rapidrecords,
             self.MAGfile,
             self.status)

    def setIndex(self, index):
        """set Index"""
        self.index

    def getIndex(self):
        """get index"""
        return self.index

    def setCadref(self, cadref):
        """set Cadref"""
        self.cadrefc= cadref

    def getCadref(self):
        """get cadref"""
        return self.cadref

    def setStatus(self, status):
        """set status"""
        self.status = status

    def getStatus(self):
        """get status"""
        return self.status

    def setRecords(self, records):
        """set records list (aka txt files)"""
        self.records = records

    def addRecord(self, record):
        """add a record to records list (aka txt files)"""
        self.records.append(record)

    def getRecords(self):
        """get records list (aka txt files)"""
        return self.records

    def setRapidRecords(self, records):
        """set records list (aka tdms files)"""
        self.rapidrecords = records

    def addRapidRecord(self, record):
        """add a rpid record to records list (aka tdms files)"""
        self.rapidrecords.append(record)

    def getRapidRecords(self):
        """get rapid records list (aka tdms files)"""
        return self.rapidrecords

    def setGObjects(self, GObjects):
        """set list of GObjects composing Magnets"""
        self.GObjects = GObjects

    def addGObject(self, GObject):
        """add GObject to list of GObjects composing Magnets"""
        self.GObjects.append(GObject)

    def getGObjects(self):
        """get list of GObjects composing Magnets"""
        return self.GObjects

    def getGObjects(self, category):
        """
        get list of GObjects composing Magnets
        from a given category
        """
        selected = [gobject from GObjects if gobject.getCategory() == category]
        return selected

    def getGObjects(self):
        """get list of GObjects composing Magnets"""
        return self.GObjects

    def setMAGfile(self, MAGfile):
        """set MAGfile configuration file(s)"""
        self.MAGfile = MAGfile

    def getMAGfile(self):
        """get MAGfile configuration file(s)"""
        return self.MAGfile

    
class GObject:
    """
    cadref
    geometry: yaml file
    materialID
    category: Helix, Ring, Current Lead, Bitter
    status: Dead/Alive
    """

    def __init__(self):
        """empty constructor"""
        self.cadref = cadref
        self.geofile = ""
        self.matID = "-"
        self.category = "Unknown"
        self.status = "Unknown"

    def __init__(self, cadref, geofile, matID, category, status):
        """default constructor"""
        self.cadref = cadref
        self.geofile = geofile
        self.matID = matID
        self.category = category
        self.status

    def __repr__(self):
        """
        representation of object
        """
        return "%s(cadref=%r, geofile=%r, matID=%r, category=%r)" % \
            (self.__class__.__name__,
             self.cadref,
             self.geofile,
             self.matID,
             self.category,
             self.status
            )
                                                    
    def setCadref(self, cadref):
        self.cadref = cadref

    def getCadref(self):
        return self.cadref

    def setStatus(self, status):
        self.status = status

    def getStatus(self):
        return self.status

    def setCategory(self, category):
        self.category = category

    def getCategory(self):
        return self.category

